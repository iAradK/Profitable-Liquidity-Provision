"""Swap-by-swap fees from the BigQuery Swap logs: E8 validation, and the replay of thin-liquidity pool-months.

A Swap event's data holds amount0 | amount1 | sqrtPriceX96 | liquidity | tick (32-byte words). The fee of a swap
is fee_rate times the input amount and is shared pro rata by the liquidity in range, so its fee growth per unit
of liquidity is fee / liquidity.

Thin-liquidity months. When a swap crosses ticks with almost no liquidity, feeGrowthGlobal jumps, and the hourly
accumulators imply far more fees than traders paid. Even the daily fee-weighted liquidity (sharing_liquidity) then
only bounds the credit of a position from above. Pool-months where the fees implied by the hourly accumulators and
the in-range liquidity exceed the fees paid by more than replay_implied_ratio, and where the position earns more
than replay_min_fee_yield, are computed swap by swap (attach_replay).
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import FeeCrediting
from experiments.data import DAY, MarketData, PoolData, SwapReplay
from experiments.outcomes import position_amounts, position_outcome, sharing_liquidity, swap_growth


WORD = 64  # hex characters per 32-byte ABI word
LOG_COLUMNS = ["block_timestamp", "block_number", "log_index", "pool", "data"]
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


def _signed(word: str) -> int:
    value = int(word, 16)
    return value - (1 << 256) if value >= 1 << 255 else value


def _decode(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, usecols=LOG_COLUMNS)
    words = [[d[2 + i * WORD: 2 + (i + 1) * WORD] for i in range(4)] for d in raw["data"]]
    return pd.DataFrame({
        "pool": raw["pool"].str.lower(),
        "ts": pd.to_datetime(raw["block_timestamp"], utc=True).dt.as_unit("s").astype("int64"),
        "block": raw["block_number"].astype(np.int64), "log_index": raw["log_index"].astype(np.int64),
        "amount0": [float(_signed(w[0])) for w in words], "amount1": [float(_signed(w[1])) for w in words],
        "sqrt_price": [int(w[2], 16) / 2**96 for w in words], "liquidity": [float(int(w[3], 16)) for w in words],
    })


def decode_swaps(csv_paths: tuple[Path, ...], cache_path: Path) -> pd.DataFrame:
    """Swap logs -> amount0, amount1 (raw, signed, pool perspective), sqrt_price and liquidity after the swap.

    Several exports (e.g. one query split by date) are merged; rows in more than one file are kept once.
    The decoded table is cached and rebuilt when any CSV is newer than the cache.
    """
    newest = max(p.stat().st_mtime for p in csv_paths)
    if cache_path.exists() and cache_path.stat().st_mtime >= newest:
        return pd.read_parquet(cache_path)
    swaps = (pd.concat([_decode(p) for p in csv_paths], ignore_index=True)
             .drop_duplicates(["block", "log_index"])
             .sort_values(["block", "log_index"], ignore_index=True))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    swaps.to_parquet(tmp, index=False)
    os.replace(tmp, cache_path)
    return swaps


def swap_fee_yield(swaps: pd.DataFrame, s0: float, s1: float, l0: float, fee_rate: float, sa: float, sb: float,
                   position_liquidity: float = 0.0, rule: str | None = None) -> float:
    """Fees per unit of liquidity earned in range [sa, sb] (sqrt prices) over `swaps`, at the closing price, over
    HODL; s0 and l0 are the price and in-range liquidity at the start. position_liquidity = 0 credits
    fee / liquidity per swap (no position effect, Eq. 2); a positive value applies `rule` per swap (swap_growth)."""
    g0, g1 = swap_growth(swaps["sqrt_price"].to_numpy(), swaps["liquidity"].to_numpy(), swaps["amount0"].to_numpy(),
                         swaps["amount1"].to_numpy(), s0, l0, sa, sb, fee_rate, position_liquidity, rule)
    x0, y0 = position_amounts(s0, sa, sb)
    return float((g0 * s1**2 + g1) / (x0 * s1**2 + y0))


# ---------------------------------------------------------------- thin-liquidity pool-months

@dataclass(frozen=True)
class ThinMonth:
    pool: str           # address
    label: str
    start_ts: int
    end_ts: int
    implied_over_paid: float
    fee_yield: float    # hourly fixed-volume fee yield of the main position (an upper bound here)


def implied_over_paid(pool: PoolData, i0: int, i1: int) -> float:
    """Fees implied by the hourly fee growth and the in-range liquidity at the ends of each hour, over the fees
    traders paid (fee growth times sharing_liquidity); far above 1 when swaps crossed nearly empty ticks."""
    grid = pool.grid
    usd = np.nan_to_num(np.diff(grid.fee_growth0[i0:i1 + 1]) * pool.usd0[i0 + 1:i1 + 1]
                        + np.diff(grid.fee_growth1[i0:i1 + 1]) * pool.usd1[i0 + 1:i1 + 1])
    paid = float(np.sum(usd * sharing_liquidity(pool, i0, i1)))
    implied = float(np.sum(usd * 0.5 * (grid.liquidity[i0:i1] + grid.liquidity[i0 + 1:i1 + 1])))
    return implied / paid if paid > 0 else np.nan


def thin_liquidity_months(market: MarketData, starts: np.ndarray, ends: np.ndarray, ratio: float,
                          min_fee_yield: float) -> list[ThinMonth]:
    crediting = FeeCrediting("fixed_volume", market.crediting.position_usd)
    months = []
    for pool in market.pools:
        i0s, i1s = pool.grid.index_at(starts), pool.grid.index_at(ends)
        for t in np.flatnonzero(i0s >= 0):
            i0, i1 = int(i0s[t]), int(i1s[t])
            excess = implied_over_paid(pool, i0, i1)
            if not excess > ratio:
                continue
            fee = position_outcome(replace(pool, replay=None), i0, i1, crediting=crediting).fee_yield
            if fee > min_fee_yield:
                months.append(ThinMonth(pool.meta.address, pool.meta.label, int(starts[t]), int(ends[t]), excess, fee))
    return months


def covers(pool_swaps: pd.DataFrame, start_ts: int, end_ts: int) -> bool:
    """The export holds the whole window: swaps within a day of both ends (a partial export would bias fees)."""
    inside = pool_swaps["ts"].between(start_ts, end_ts - 1)
    if not inside.any():
        return False
    ts = pool_swaps.loc[inside, "ts"]
    return int(ts.min()) < start_ts + DAY and int(ts.max()) > end_ts - DAY


def attach_replay(market: MarketData, swaps: pd.DataFrame, months: list[ThinMonth]
                  ) -> tuple[MarketData, list[ThinMonth]]:
    """Market whose pools carry the swaps of the covered thin months; also returns the months not covered."""
    by_pool: dict[str, list[ThinMonth]] = {}
    missing = []
    for month in months:
        pool_swaps = swaps[swaps["pool"] == month.pool]
        (by_pool.setdefault(month.pool, []) if covers(pool_swaps, month.start_ts, month.end_ts)
         else missing).append(month)
    pools = []
    for pool in market.pools:
        covered = by_pool.get(pool.meta.address)
        if not covered:
            pools.append(pool)
            continue
        windows = tuple(sorted((m.start_ts, m.end_ts) for m in covered))
        rows = swaps[(swaps["pool"] == pool.meta.address)]
        rows = rows[np.any([rows["ts"].between(w0, w1 - 1) for w0, w1 in windows], axis=0)]
        pools.append(replace(pool, replay=SwapReplay(windows, *(rows[c].to_numpy() for c in
                                                                ("ts", "amount0", "amount1", "sqrt_price",
                                                                 "liquidity")))))
    return replace(market, pools=tuple(pools)), missing


def months_table(months: list[ThinMonth], missing: list[ThinMonth]) -> pd.DataFrame:
    missing_keys = {(m.pool, m.start_ts) for m in missing}
    return pd.DataFrame([{"label": m.label, "pool": m.pool,
                          "period": pd.Timestamp(m.start_ts, unit="s").strftime("%Y-%m"),
                          "implied_over_paid": m.implied_over_paid, "hourly_fee_yield": m.fee_yield,
                          "replayed": (m.pool, m.start_ts) not in missing_keys} for m in months])


def replay_query(months: list[ThinMonth]) -> str:
    """BigQuery SQL that exports the Swap logs of `months` (same columns as the other swap exports)."""
    windows = "\n    OR ".join(
        f"(address = '{m.pool}' AND block_timestamp >= TIMESTAMP('{pd.Timestamp(m.start_ts, unit='s'):%Y-%m-%d}') "
        f"AND block_timestamp < TIMESTAMP('{pd.Timestamp(m.end_ts, unit='s'):%Y-%m-%d}'))  -- {m.label}"
        for m in sorted(months, key=lambda m: (m.label, m.start_ts)))
    return ("-- Swap logs of the thin-liquidity pool-months (written by run_experiments.py)\n"
            "SELECT block_timestamp, block_number, transaction_hash, log_index, address AS pool, data\n"
            "FROM `bigquery-public-data.crypto_ethereum.logs`\n"
            f"WHERE topics[SAFE_OFFSET(0)] = '{SWAP_TOPIC}'\n"
            f"  AND (\n    {windows}\n  )\n"
            "ORDER BY block_number, log_index;\n")
