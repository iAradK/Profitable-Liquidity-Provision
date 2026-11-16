"""Outcome of a passive Uniswap v3 position over a holding period, and panels of outcomes (pools x periods).

All returns are measured against holding the deposited tokens (HODL), valued at the closing price:
    IL        = V_LP(T) / V_HODL(T) - 1
    fee yield = value of the collected fee tokens at T / V_HODL(T)
    net       = IL + fee yield
These ratios do not depend on the numeraire. Fees are computed from feeGrowthGlobal, i.e. exactly what a
position of the given range earns per unit of liquidity while the price is inside its range.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from experiments.config import FeeCrediting, Horizon, SampleConfig
from experiments.data import DAY, HourlyGrid, MarketData, PoolData, PoolMeta, SwapReplay


def il_from_ratio(ratio: np.ndarray | float) -> np.ndarray | float:
    """Full-range (constant-product) IL for a price ratio R = P_T / P_0."""
    return 2.0 * np.sqrt(ratio) / (1.0 + ratio) - 1.0


def position_amounts(sqrt_price: np.ndarray | float, lower: float, upper: float) -> tuple:
    """Raw token amounts per unit of liquidity for the sqrt-price range [lower, upper] (0, inf = full range)."""
    clipped = np.clip(sqrt_price, lower, upper)
    return 1.0 / clipped - 1.0 / upper, clipped - lower


@dataclass(frozen=True)
class Outcome:
    il: float
    fee_yield: float
    ratio: float      # P_T / P_0
    variance: float   # realized variance of the hourly log price over the period

    @property
    def net(self) -> float:
        return self.il + self.fee_yield


def position_liquidity(pool: PoolData, i0: int, x0: float, y0: float, crediting: FeeCrediting | None) -> float:
    """Liquidity L that crediting.position_usd buys at grid state i0 (x0, y0: token amounts per unit of liquidity).

    0 means the position is treated as too small to matter: no crediting, or no USD prices at entry (then the
    position size is unknown).
    """
    if crediting is None:
        return 0.0
    value_per_liquidity = x0 * pool.usd0[i0] + y0 * pool.usd1[i0]
    if not np.isfinite(value_per_liquidity) or value_per_liquidity <= 0:
        return 0.0
    return crediting.position_usd / value_per_liquidity


def sharing_liquidity(pool: PoolData, i0: int, i1: int) -> np.ndarray:
    """Per hour from state i0 to i1: the in-range liquidity L_act the hour's fees were shared by.

    The fee-weighted liquidity of the hour's UTC day (fees paid / fee growth, see fee_weighted_liquidity), or the
    mean in-range liquidity at the two ends of the hour where the daily fees are unknown.
    """
    liquidity = pool.grid.liquidity[i0:i1 + 1]
    fee_weighted = pool.fee_liquidity[i0 + 1:i1 + 1]
    return np.where(np.isfinite(fee_weighted), fee_weighted, 0.5 * (liquidity[:-1] + liquidity[1:]))


def crediting_factor(pool: PoolData, i0: int, i1: int, x0: float, y0: float,
                     crediting: FeeCrediting | None) -> np.ndarray | float:
    """Per-hour share of the credit L * dG that the position keeps under the crediting rule (Section III-A).

    fixed_volume: L_act / (L_act + L). Swaps of a fixed size pay the same fees, now shared with the position.
        With L_act the fee-weighted harmonic mean of the liquidity at the swaps, this is an upper bound on the
        swap-by-swap credit sum_s fee_s * L / (L_s + L) (Jensen), and it never exceeds the fees paid.
    proportional: min(1, L_act / L). Arbitrage volume, and hence its fees, grows with the liquidity, up to the
        fees traders actually paid (a swap through nearly empty ticks makes dG jump without more volume).
    The two coincide for L << L_act.
    """
    position = position_liquidity(pool, i0, x0, y0, crediting)
    if position == 0.0:
        return 1.0
    ratio = sharing_liquidity(pool, i0, i1) / position
    if crediting.rule == "fixed_volume":
        return ratio / (1.0 + ratio)
    return np.minimum(1.0, ratio)


def swap_growth(after: np.ndarray, liquidity: np.ndarray, amount0: np.ndarray, amount1: np.ndarray,
                s_start: float, l_start: float, sa: float, sb: float, fee_rate: float, position: float,
                rule: str | None) -> tuple[float, float]:
    """Fee growth per unit of liquidity credited to a position of range [sa, sb] (sqrt prices) over a run of swaps.

    Each swap pays fee_rate on its input amount, shared by the liquidity in range L_s. A Swap event records only
    the liquidity after the swap; a swap that ends in a nearly empty region paid most of its fee before, where the
    liquidity was higher. Hence L_s is the larger of the liquidity before the swap (after the previous one, or
    l_start for the first) and after it. The credit is half when the price enters or leaves the range during the
    swap. With a position of liquidity L > 0 the rule scales each swap like crediting_factor does each hour:
    fixed_volume L_s / (L_s + L), proportional min(1, L_s / L).
    """
    active = liquidity > 0
    after, liquidity, amount0, amount1 = after[active], liquidity[active], amount0[active], amount1[active]
    before = np.concatenate([[s_start], after[:-1]])
    liquidity = np.maximum(liquidity, np.concatenate([[l_start], liquidity[:-1]]))
    credit = 0.5 * (((before >= sa) & (before <= sb)).astype(float) + ((after >= sa) & (after <= sb)))
    if position > 0 and rule == "fixed_volume":
        credit = credit * liquidity / (liquidity + position)
    elif position > 0 and rule == "proportional":
        credit = credit * np.minimum(1.0, liquidity / position)
    growth0 = np.where(amount0 > 0, amount0 * fee_rate, 0.0) / liquidity
    growth1 = np.where(amount1 > 0, amount1 * fee_rate, 0.0) / liquidity
    return float(credit @ growth0), float(credit @ growth1)


def replayed_fees(replay: SwapReplay, t: np.ndarray, s: np.ndarray, liquidity: np.ndarray, credit: np.ndarray,
                  sa: float, sb: float, fee_rate: float, position: float, rule: str | None) -> tuple[float, float]:
    """Swap-by-swap fee growth for the hours of t (grid state times) inside the replay windows. Their hourly credit
    is set to 0 in place, so the hourly and the swap-based parts never overlap."""
    fee0 = fee1 = 0.0
    for w0, w1 in replay.windows:
        hours = (t[:-1] >= w0) & (t[1:] <= w1)
        if not hours.any():
            continue
        credit[hours] = 0.0
        first, last = int(np.argmax(hours)), len(hours) - 1 - int(np.argmax(hours[::-1]))
        rows = slice(*np.searchsorted(replay.ts, [t[first], t[last + 1]]))
        g0, g1 = swap_growth(replay.sqrt_price[rows], replay.liquidity[rows], replay.amount0[rows],
                             replay.amount1[rows], s[first], liquidity[first], sa, sb, fee_rate, position, rule)
        fee0, fee1 = fee0 + g0, fee1 + g1
    return fee0, fee1


def position_outcome(pool: PoolData, i0: int, i1: int, lower: float = 0.0, upper: float = np.inf,
                     crediting: FeeCrediting | None = None) -> Outcome:
    """Position opened at grid state i0 and closed at state i1 with range [lower * s0, upper * s0] in sqrt price.

    Fees accrue in hours when the price is in range: full credit when the price is inside at both ends of the
    hour, half credit when it enters or leaves during the hour. Without `crediting`, fees are L * dG per unit of
    liquidity (Eq. fee_growth); with it, each hour's credit is scaled by crediting_factor. Hours inside the pool's
    replay windows are credited swap by swap instead (replayed_fees).
    """
    grid = pool.grid
    s = grid.sqrt_price[i0:i1 + 1]
    s0, s1 = s[0], s[-1]
    sa, sb = lower * s0, upper * s0
    x0, y0 = position_amounts(s0, sa, sb)
    x1, y1 = position_amounts(s1, sa, sb)
    hodl = x0 * s1**2 + y0
    inside = ((s >= sa) & (s <= sb)).astype(float)
    credit = 0.5 * (inside[:-1] + inside[1:])
    d0, d1 = np.diff(grid.fee_growth0[i0:i1 + 1]), np.diff(grid.fee_growth1[i0:i1 + 1])
    credit = credit * crediting_factor(pool, i0, i1, x0, y0, crediting)
    swap0 = swap1 = 0.0
    if pool.replay is not None:
        swap0, swap1 = replayed_fees(pool.replay, grid.t_end[i0:i1 + 1], s, grid.liquidity[i0:i1 + 1], credit,
                                     sa, sb, pool.meta.fee_rate,
                                     position_liquidity(pool, i0, x0, y0, crediting),
                                     crediting.rule if crediting else None)
    fee0 = credit @ d0 + swap0
    fee1 = credit @ d1 + swap1
    log_price = 2.0 * np.log(s)
    return Outcome(il=float((x1 * s1**2 + y1) / hodl - 1.0),
                   fee_yield=float((fee0 * s1**2 + fee1) / hodl),
                   ratio=float((s1 / s0) ** 2),
                   variance=float(np.sum(np.diff(log_price) ** 2)))


# ---------------------------------------------------------------- periods

def add_months(ts: int, months: int) -> int:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    total = d.year * 12 + d.month - 1 + months
    return int(d.replace(year=total // 12, month=total % 12 + 1).timestamp())


def advance(ts: int, horizon: Horizon) -> int:
    return add_months(ts, horizon.months) if horizon.months else ts + horizon.days * DAY


def make_periods(start_ts: int, end_ts: int, length: Horizon, step: Horizon | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Periods [start, start + length) every `step` (default: back to back) that end by end_ts."""
    step = step or length
    starts, ends = [], []
    t = start_ts
    while (end := advance(t, length)) <= end_ts:
        starts.append(t)
        ends.append(end)
        t = advance(t, step)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def sample_periods(sample: SampleConfig, data_end_ts: int, length: Horizon, step: Horizon | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    return make_periods(sample.start_ts, min(sample.end_ts, data_end_ts), length, step)


# ---------------------------------------------------------------- panels

@dataclass(frozen=True)
class Panel:
    """Outcomes of one position type for every pool and period; NaN where the pool had no data yet."""
    pools: tuple[PoolMeta, ...]
    horizon: Horizon
    starts: np.ndarray
    ends: np.ndarray
    il: np.ndarray        # (n_pools, n_periods)
    fee: np.ndarray
    ratio: np.ndarray
    variance: np.ndarray
    tvl: np.ndarray       # TVL (USD) at the period start
    hodl_usd: np.ndarray  # USD return of holding the deposited 50/50 tokens

    @property
    def net(self) -> np.ndarray:
        return self.il + self.fee

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.net)

    @property
    def n_pools(self) -> int:
        return len(self.pools)

    @property
    def n_periods(self) -> int:
        return len(self.starts)


def build_panel(market: MarketData, starts: np.ndarray, ends: np.ndarray, horizon: Horizon,
                lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> Panel:
    """Full-range positions by default; `lower`/`upper` (n_pools, n_periods) give per-cell sqrt-price range factors."""
    shape = (len(market.pools), len(starts))
    il, fee, ratio, variance, tvl, hodl_usd = (np.full(shape, np.nan) for _ in range(6))
    for p, pool in enumerate(market.pools):
        grid = pool.grid
        i0, i1 = grid.index_at(starts), grid.index_at(ends)
        for t in np.flatnonzero(i0 >= 0):
            lo = 0.0 if lower is None else lower[p, t]
            hi = np.inf if upper is None else upper[p, t]
            if np.isnan(lo) or np.isnan(hi):
                continue
            outcome = position_outcome(pool, int(i0[t]), int(i1[t]), lo, hi, market.crediting)
            il[p, t], fee[p, t] = outcome.il, outcome.fee_yield
            ratio[p, t], variance[p, t] = outcome.ratio, outcome.variance
        started = i0 >= 0
        tvl[p, started] = grid.tvl_usd[i0[started]]
        price0, price1 = market.token_prices(pool.meta.token0), market.token_prices(pool.meta.token1)
        growth0 = price0.at(ends) / price0.at(starts)
        growth1 = price1.at(ends) / price1.at(starts)
        hodl_usd[p] = np.where(started, 0.5 * (growth0 + growth1) - 1.0, np.nan)
    return Panel(tuple(p.meta for p in market.pools), horizon, starts, ends, il, fee, ratio, variance, tvl, hodl_usd)


def daily_log_prices(grid: HourlyGrid, start_ts: int, end_ts: int) -> np.ndarray:
    """Log raw price at every 00:00 UTC in [start_ts, end_ts] where the pool already existed."""
    idx = grid.index_at(np.arange(start_ts, end_ts + 1, DAY))
    return 2.0 * np.log(grid.sqrt_price[idx[idx >= 0]])
