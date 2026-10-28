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

from experiments.config import Horizon, SampleConfig
from experiments.data import DAY, HourlyGrid, MarketData, PoolData, PoolMeta


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


def fee_cap_scale(pool: PoolData, i0: int, i1: int, x0: float, y0: float, credit: np.ndarray,
                  d0: np.ndarray, d1: np.ndarray, position_usd: float) -> np.ndarray:
    """Per-hour factor <= 1 so a position of position_usd never earns more in a UTC day than all swappers paid.

    feeGrowth is fees per unit of in-range liquidity. When a swap pushes the price through nearly empty ticks it
    jumps, and a price-taker position (much larger than the liquidity actually in range) would be credited far
    more than the pool's total fees. The cap is daily because the subgraph's hourly feesUSD is zero before 2024.
    Hours with unknown USD prices, and pools without daily fees, are left uncapped.
    """
    usd0, usd1 = pool.usd0[i0:i1 + 1], pool.usd1[i0:i1 + 1]
    liquidity = position_usd / (x0 * usd0[0] + y0 * usd1[0])
    earned = np.nan_to_num(liquidity * credit * (d0 * usd0[1:] + d1 * usd1[1:]))
    days, day_of_hour = np.unique((pool.grid.t_end[i0 + 1:i1 + 1] - 1) // DAY, return_inverse=True)
    earned_per_day = np.bincount(day_of_hour, weights=earned, minlength=len(days))
    paid_per_day = np.zeros(len(days))
    paid_per_day[day_of_hour] = pool.day_fees_usd[i0 + 1:i1 + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(earned_per_day > paid_per_day, paid_per_day / earned_per_day, 1.0)
    return np.where(np.isfinite(scale), scale, 1.0)[day_of_hour]


def position_outcome(pool: PoolData, i0: int, i1: int, lower: float = 0.0, upper: float = np.inf,
                     position_usd: float | None = None) -> Outcome:
    """Position opened at grid state i0 and closed at state i1 with range [lower * s0, upper * s0] in sqrt price.

    Fees accrue in hours when the price is in range: full credit when the price is inside at both ends of the
    hour, half credit when it enters or leaves during the hour. With position_usd, each hour's fees are capped
    at the pool's feesUSD for that hour (see fee_cap_scale).
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
    if position_usd is not None:
        credit = credit * fee_cap_scale(pool, i0, i1, x0, y0, credit, d0, d1, position_usd)
    fee0 = credit @ d0
    fee1 = credit @ d1
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
            outcome = position_outcome(pool, int(i0[t]), int(i1[t]), lo, hi, market.position_usd)
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
