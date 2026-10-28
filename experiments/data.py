"""Load the downloaded data into arrays: hourly pool state, daily token USD prices and daily gas costs.

Hourly grids are built once from pool_data/hourly/<pool>.csv and cached as .npz files; a cache is
rebuilt automatically when its CSV is newer (e.g. after the download is re-run for the last days).
"""

import logging
import os
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Pool
from subgraph.parallel import run_each

log = logging.getLogger("experiments")

Q96 = 2**96
Q128 = 2**128
HOUR = 3600
DAY = 86400
TICK_BASE = 1.0001
TICK_SPACING = {100: 1, 500: 10, 3000: 60, 10000: 200}  # Uniswap v3 factory defaults per fee tier

BIG_INT_COLUMNS = ("sqrtPrice", "liquidity", "feeGrowthGlobal0X128", "feeGrowthGlobal1X128")


@dataclass(frozen=True)
class PoolMeta:
    address: str
    label: str
    category: str
    fee_tier: int
    token0: str
    token1: str
    symbol0: str
    symbol1: str
    decimals0: int
    decimals1: int

    @property
    def fee_rate(self) -> float:
        return self.fee_tier / 1e6

    @property
    def family(self) -> str:
        """The token pair regardless of fee tier, e.g. 'USDC/WETH'."""
        return "/".join(sorted((self.symbol0, self.symbol1)))

    @property
    def decimal_shift(self) -> float:
        """Multiplies a raw price (token1 base units per token0 base unit) into token1 per token0."""
        return 10.0 ** (self.decimals0 - self.decimals1)


@dataclass(frozen=True)
class HourlyGrid:
    """Pool state at the end of every hour, forward-filled through hours without events.

    sqrt_price:   sqrtPriceX96 / 2**96, the square root of the raw price (token1 per token0 base units).
    fee_growth0/1: feeGrowthGlobal{0,1}X128 / 2**128, cumulative fees per unit of liquidity in raw token units.
    liquidity:    active (in-range) liquidity.
    observed:     True where the subgraph has a row for that hour.
    """
    t_end: np.ndarray
    sqrt_price: np.ndarray
    fee_growth0: np.ndarray
    fee_growth1: np.ndarray
    liquidity: np.ndarray
    tvl_usd: np.ndarray
    observed: np.ndarray

    @property
    def first_ts(self) -> int:
        return int(self.t_end[0])

    def index_at(self, ts: np.ndarray | int) -> np.ndarray:
        """Index of the state at time ts: the last hour that ended at or before ts, -1 before the first hour.

        Times after the last row map to the last row: no row means no event, so the state did not change.
        """
        idx = (np.asarray(ts, dtype=np.int64) - self.t_end[0]) // HOUR
        return np.clip(idx, -1, len(self.t_end) - 1)

    def fee_growth_at(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fee growth at arbitrary times, linearly interpolated within the hour."""
        ts = np.asarray(ts, dtype=float)
        before = ts < self.t_end[0]
        g0 = np.where(before, np.nan, np.interp(ts, self.t_end, self.fee_growth0))
        g1 = np.where(before, np.nan, np.interp(ts, self.t_end, self.fee_growth1))
        return g0, g1


@dataclass(frozen=True)
class PoolData:
    """Per grid state (hour): usd0/usd1, the USD value of one raw base unit of token0/token1 (NaN before prices
    exist); day_fees_usd, the fees all swappers paid in the pool over that hour's UTC day (NaN without daily data).
    """
    meta: PoolMeta
    grid: HourlyGrid
    usd0: np.ndarray
    usd1: np.ndarray
    day_fees_usd: np.ndarray


@dataclass(frozen=True)
class TokenPrices:
    """Daily USD closes; usd[i] is the price at day_end[i] (the end of that UTC day)."""
    day_end: np.ndarray
    usd: np.ndarray

    def at(self, ts: np.ndarray | int) -> np.ndarray:
        """USD price at time ts: the close of the last day that ended at or before ts."""
        idx = np.searchsorted(self.day_end, np.asarray(ts), side="right") - 1
        return np.where(idx >= 0, self.usd[np.clip(idx, 0, None)], np.nan)


@dataclass(frozen=True)
class GasCosts:
    """Daily median gas cost in ETH of opening (mint) and closing (decrease + collect) a position."""
    day_start: np.ndarray
    entry_eth: np.ndarray
    exit_eth: np.ndarray

    def cost_eth(self, ts: int, op: str) -> float:
        if len(self.day_start) == 0:
            return 0.0
        idx = max(int(np.searchsorted(self.day_start, ts, side="right")) - 1, 0)
        return float((self.entry_eth if op == "entry" else self.exit_eth)[idx])


@dataclass(frozen=True)
class MarketData:
    pools: tuple[PoolData, ...]
    prices: dict[str, TokenPrices]
    gas: GasCosts
    eth_usd: TokenPrices
    data_end_ts: int
    position_usd: float  # size of the modelled position; caps its hourly fees at what the pool paid

    def token_prices(self, token: str) -> TokenPrices:
        if token not in self.prices:
            raise KeyError(f"no USD prices for token {token}; re-run the 'prices' download step")
        return self.prices[token]


# ---------------------------------------------------------------- hourly grids

def _scaled(values: pd.Series, scale: int) -> np.ndarray:
    """256-bit integers stored as strings -> floats divided by `scale` (exact integer division first)."""
    return np.array([int(v) / scale for v in values], dtype=float)


def build_grid(csv_path: Path) -> HourlyGrid:
    df = pd.read_csv(csv_path, usecols=["periodStartUnix", *BIG_INT_COLUMNS, "tvlUSD"],
                     dtype={c: str for c in BIG_INT_COLUMNS})
    df = (df.dropna(subset=list(BIG_INT_COLUMNS))
            .sort_values("periodStartUnix")
            .drop_duplicates("periodStartUnix", keep="last"))
    sqrt_price = _scaled(df["sqrtPrice"], Q96)
    liquidity = _scaled(df["liquidity"], 1)
    started = np.flatnonzero((sqrt_price > 0) & (liquidity > 0))
    if len(started) == 0:
        raise ValueError(f"{csv_path.name}: no hour with both a price and liquidity")
    keep = np.zeros(len(df), dtype=bool)
    keep[started[0]:] = True
    keep &= sqrt_price > 0
    df = df[keep]

    starts = df["periodStartUnix"].to_numpy(np.int64)
    n_hours = int((starts[-1] - starts[0]) // HOUR) + 1
    position = (starts - starts[0]) // HOUR

    def on_grid(values: np.ndarray) -> np.ndarray:
        out = np.full(n_hours, np.nan)
        out[position] = values
        return pd.Series(out).ffill().to_numpy()

    observed = np.zeros(n_hours, dtype=bool)
    observed[position] = True
    return HourlyGrid(
        t_end=starts[0] + HOUR * (np.arange(n_hours, dtype=np.int64) + 1),
        sqrt_price=on_grid(sqrt_price[keep]),
        fee_growth0=on_grid(_scaled(df["feeGrowthGlobal0X128"], Q128)),
        fee_growth1=on_grid(_scaled(df["feeGrowthGlobal1X128"], Q128)),
        liquidity=on_grid(liquidity[keep]),
        tvl_usd=on_grid(df["tvlUSD"].to_numpy(float)),
        observed=observed,
    )


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_grid(address: str, hourly_dir: Path, cache_dir: Path, force_refresh: bool) -> HourlyGrid:
    csv_path = hourly_dir / f"{address}.csv"
    cache_path = cache_dir / f"{address}.npz"
    fresh = cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime
    if fresh and not force_refresh:
        with np.load(cache_path) as saved:
            if all(f.name in saved.files for f in fields(HourlyGrid)):  # caches from older versions are rebuilt
                return HourlyGrid(**{f.name: saved[f.name] for f in fields(HourlyGrid)})
    grid = build_grid(csv_path)
    _save_npz_atomic(cache_path, {f.name: getattr(grid, f.name) for f in fields(HourlyGrid)})
    return grid


# ---------------------------------------------------------------- metadata, prices, gas

def load_pool_meta(data_dir: Path, universe: tuple[Pool, ...] | None,
                   label_categories: dict[str, str]) -> list[PoolMeta]:
    """Pools in `universe` (config order), or every pool in meta/pools.csv when universe is None."""
    meta = pd.read_csv(data_dir / "meta" / "pools.csv", dtype=str).set_index("id")
    meta.index = meta.index.str.lower()
    if universe is not None:
        missing = [p.label for p in universe if p.address not in meta.index]
        if missing:
            raise ValueError(f"pools missing from {data_dir / 'meta' / 'pools.csv'}: {missing}")
        rows = [(p.address, p.label, p.category) for p in universe]
    else:
        rows = [(addr, r["label"], label_categories.get(r["label"], "other")) for addr, r in meta.iterrows()]
    return [
        PoolMeta(address=addr, label=label, category=category, fee_tier=int(meta.at[addr, "feeTier"]),
                 token0=meta.at[addr, "token0_id"].lower(), token1=meta.at[addr, "token1_id"].lower(),
                 symbol0=meta.at[addr, "token0_symbol"], symbol1=meta.at[addr, "token1_symbol"],
                 decimals0=int(meta.at[addr, "token0_decimals"]), decimals1=int(meta.at[addr, "token1_decimals"]))
        for addr, label, category in rows
    ]


def load_token_prices(path: Path) -> TokenPrices:
    df = pd.read_csv(path, usecols=["date", "priceUSD"]).sort_values("date")
    usd = df["priceUSD"].astype(float).where(lambda s: s > 0)
    return TokenPrices(day_end=df["date"].to_numpy(np.int64) + DAY, usd=usd.ffill().to_numpy())


def load_gas_costs(path: Path | None) -> GasCosts:
    if path is None:
        return GasCosts(np.array([], dtype=np.int64), np.array([]), np.array([]))
    df = pd.read_csv(path, usecols=["day", "op", "median_cost_eth"])
    wide = df.pivot_table(index="day", columns="op", values="median_cost_eth").sort_index().ffill().bfill()
    days = pd.to_datetime(wide.index, utc=True).as_unit("s").astype("int64")  # unit-safe (pandas 3 parses to us)
    return GasCosts(days.to_numpy(np.int64), wide["entry"].to_numpy(float), wide["exit"].to_numpy(float))


def usd_per_raw_unit(prices: TokenPrices | None, decimals: int, t_end: np.ndarray) -> np.ndarray:
    if prices is None:
        return np.full(len(t_end), np.nan)
    return prices.at(t_end) / 10.0 ** decimals


def day_fees_per_hour(path: Path, fee_rate: float, t_end: np.ndarray) -> np.ndarray:
    """Pool fees in USD of the UTC day of every grid hour; days without a row paid nothing."""
    if not path.exists():
        return np.full(len(t_end), np.nan)
    df = pd.read_csv(path, usecols=["date", "volumeUSD", "feesUSD"])
    fees = df["feesUSD"].where(df["feesUSD"] > 0, df["volumeUSD"] * fee_rate)
    by_day = dict(zip(df["date"].to_numpy(np.int64), fees.to_numpy(float)))
    day = (t_end - 1) // DAY * DAY  # the hour ending at t_end started on this day
    return np.array([by_day.get(int(d), 0.0) for d in day])


def load_market(data_dir: Path, gas_file: Path | None, eth_token: str, universe: tuple[Pool, ...] | None,
                label_categories: dict[str, str], cache_dir: Path, max_workers: int,
                force_refresh: bool, position_usd: float) -> MarketData:
    metas = load_pool_meta(data_dir, universe, label_categories)
    task = partial(load_grid, hourly_dir=data_dir / "hourly", cache_dir=cache_dir, force_refresh=force_refresh)
    batch = run_each([m.address for m in metas], task, name=str, max_workers=max_workers, use_processes=True)
    if batch.failures:
        raise RuntimeError(f"could not load hourly data for {sorted(batch.failures)}")

    tokens = sorted({m.token0 for m in metas} | {m.token1 for m in metas} | {eth_token})
    price_dir = data_dir / "token_prices"
    prices = {t: load_token_prices(price_dir / f"{t}.csv") for t in tokens if (price_dir / f"{t}.csv").exists()}
    if eth_token not in prices:
        raise FileNotFoundError(f"no USD prices for the gas token {eth_token} in {price_dir}")
    pools = []
    for m in metas:
        grid = batch.results[m.address]
        pools.append(PoolData(m, grid, usd_per_raw_unit(prices.get(m.token0), m.decimals0, grid.t_end),
                              usd_per_raw_unit(prices.get(m.token1), m.decimals1, grid.t_end),
                              day_fees_per_hour(data_dir / "daily" / f"{m.address}.csv", m.fee_rate, grid.t_end)))
    pools = tuple(pools)
    no_daily = [p.meta.label for p in pools if np.isnan(p.day_fees_usd).all()]
    if no_daily:
        log.warning("no daily fees (run the 'daily' download step), fees are NOT capped for: %s", ", ".join(no_daily))
    data_end = max(int(p.grid.t_end[-1]) for p in pools)
    log.info("loaded %d pools, %d token price series; data ends %s", len(pools), len(prices),
             pd.Timestamp(data_end, unit="s"))
    return MarketData(pools, prices, load_gas_costs(gas_file), prices[eth_token], data_end, position_usd)
