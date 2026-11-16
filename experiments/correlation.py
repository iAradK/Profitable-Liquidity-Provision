"""Token-level correlation model: daily USD log returns of every token, each pool's price-ratio return as the
difference of its two tokens' returns, and their trailing correlations over a calibration window.

The pool's own price history gives its pair variance; the token returns give what a single pool's history cannot:
the correlation between the two tokens of a pair (rho in sigma_pair^2 = sigma_A^2 + sigma_B^2 - 2 rho sigma_A
sigma_B) and the correlation between the price ratios of different pools, which sets the joint IL of a portfolio.
Tokens priced at a constant USD value (the subgraph prices stablecoins at exactly 1 USD) have zero return variance,
so their correlations are undefined (NaN) and such pools are treated as uncorrelated with the others.
"""

from dataclasses import dataclass

import numpy as np

from experiments.data import DAY, MarketData, PoolMeta
from experiments.outcomes import Panel
from experiments.selection import past_periods

MIN_DAYS = 20  # fewer joint daily returns than this in a window: correlation not estimated


@dataclass(frozen=True)
class TokenReturns:
    day_end: np.ndarray   # (n_days,) end of each daily return interval
    returns: np.ndarray   # (n_days, n_tokens) daily USD log returns; NaN where a price is missing
    tokens: tuple[str, ...]

    def column(self, token: str) -> int:
        return self.tokens.index(token)

    def window(self, start_ts: int, end_ts: int) -> np.ndarray:
        """Rows of the daily returns that lie inside [start_ts, end_ts]."""
        return self.returns[(self.day_end - DAY >= start_ts) & (self.day_end <= end_ts)]


def token_returns(market: MarketData, panel: Panel) -> TokenReturns:
    tokens = tuple(sorted({t for p in panel.pools for t in (p.token0, p.token1)}))
    grid = np.arange(int(panel.starts[0]), int(panel.ends[-1]) + 1, DAY)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.column_stack([np.diff(np.log(market.token_prices(t).at(grid))) for t in tokens])
    return TokenReturns(grid[1:], returns, tokens)


def ratio_weights(returns: TokenReturns, pools: tuple[PoolMeta, ...]) -> np.ndarray:
    """(n_tokens, n_pools): +1 for token0, -1 for token1, so returns @ weights are the pools' log price-ratio
    returns (the orientation does not matter: IL depends on |log ratio| only)."""
    weights = np.zeros((len(returns.tokens), len(pools)))
    for p, meta in enumerate(pools):
        weights[returns.column(meta.token0), p] += 1.0
        weights[returns.column(meta.token1), p] -= 1.0
    return weights


def calibration_window(panel: Panel, t: int, calibration_periods: int) -> tuple[int, int] | None:
    """(start, end) timestamps of the periods a decision at period t may use; None before any period ended."""
    past = past_periods(panel, t, calibration_periods)
    return (int(panel.starts[past[0]]), int(panel.starts[t])) if len(past) else None


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    both = np.isfinite(x) & np.isfinite(y)
    x, y = x[both], y[both]
    if len(x) < MIN_DAYS or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pair_correlations(returns: TokenReturns, pools: tuple[PoolMeta, ...], start_ts: int,
                      end_ts: int) -> np.ndarray:
    """Per pool: correlation of its two tokens' daily USD returns over the window (NaN for constant prices)."""
    rows = returns.window(start_ts, end_ts)
    return np.array([correlation(rows[:, returns.column(p.token0)], rows[:, returns.column(p.token1)])
                     for p in pools])


def trailing_pair_correlation(returns: TokenReturns, panel: Panel, decisions: np.ndarray,
                              calibration_periods: int) -> np.ndarray:
    """(n_pools, n_periods): token correlation of every pool from the calibration window before each decision."""
    rho = np.full((panel.n_pools, panel.n_periods), np.nan)
    for t in decisions:
        window = calibration_window(panel, t, calibration_periods)
        if window is not None:
            rho[:, t] = pair_correlations(returns, panel.pools, *window)
    return rho


def ratio_correlation_matrix(returns: TokenReturns, weights: np.ndarray, start_ts: int, end_ts: int) -> np.ndarray:
    """Correlation matrix of the pools' daily price-ratio returns over the window. Days with a missing price are
    dropped pairwise by zero-filling (a missing day contributes nothing); a pool without return variation gets an
    identity row and column (uncorrelated). The result is projected to the nearest valid correlation matrix."""
    rows = np.nan_to_num(returns.window(start_ts, end_ts)) @ weights
    n = weights.shape[1]
    if len(rows) < MIN_DAYS:
        return np.eye(n)
    std = rows.std(axis=0)
    live = std > 0
    matrix = np.eye(n)
    if live.sum() > 1:
        matrix[np.ix_(live, live)] = np.corrcoef(rows[:, live], rowvar=False)
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.clip(values, 1e-8, None)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)
