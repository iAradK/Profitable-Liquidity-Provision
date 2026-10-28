"""Pool scores computed from past outcomes only. Higher score = more attractive pool.

Every score takes a History (pools x past periods) and returns one value per pool, so a new
ranking rule is one new function added to `score_functions`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.special import ndtr

from experiments.stats import quiet


@dataclass(frozen=True)
class History:
    """Past outcomes, shape (n_pools, n_past_periods); NaN where a pool had no data."""
    il: np.ndarray
    fee: np.ndarray
    variance: np.ndarray

    @property
    def net(self) -> np.ndarray:
        return self.il + self.fee


ScoreFn = Callable[[History], np.ndarray]


def net_tail(history: History, tail_quantile: float) -> np.ndarray:
    """Net90 for tail_quantile = 0.10: median fee yield + the 10th percentile of IL."""
    return quiet(np.nanmedian, history.fee, axis=1) + quiet(np.nanquantile, history.il, tail_quantile, axis=1)


def median_fee(history: History) -> np.ndarray:
    return quiet(np.nanmedian, history.fee, axis=1)


def low_variance(history: History) -> np.ndarray:
    """The LVR ranking: lowest realized variance of the pair's log price first."""
    return -quiet(np.nanmean, history.variance, axis=1)


def fee_minus_lvr(history: History) -> np.ndarray:
    """LVR-adjusted yield: expected fees minus the expected full-range loss sigma^2 / 8."""
    return quiet(np.nanmean, history.fee, axis=1) - quiet(np.nanmean, history.variance, axis=1) / 8.0


def model_il_tail(variance: np.ndarray, quantile: float, iterations: int = 60) -> np.ndarray:
    """q-quantile of next-period full-range IL per pool, from the pools' past realized variances (pools x periods).

    The period's log price change x is modelled as N(0, s^2) with s^2 drawn uniformly from the pool's past
    realized variances (hourly data), a scale mixture of normals whose fat tails come from changing volatility.
    IL = sech(x / 2) - 1 depends only on |x|, so its q-quantile is IL at the z with P(|x| > z) = q.
    """
    sigma = np.sqrt(np.where(variance >= 0, variance, np.nan))
    n = np.sum(np.isfinite(sigma), axis=1)
    lo = np.zeros(len(sigma))
    hi = 10.0 * np.nan_to_num(quiet(np.nanmax, sigma, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iterations):
            z = 0.5 * (lo + hi)
            exceed = np.nansum(2.0 * ndtr(-z[:, None] / sigma), axis=1) / n
            lo, hi = np.where(exceed > quantile, z, lo), np.where(exceed > quantile, hi, z)
    return np.where(n > 0, 1.0 / np.cosh(0.25 * (lo + hi)) - 1.0, np.nan)


def lvr_tail(history: History, risk_aversion: float, tail_quantile: float) -> np.ndarray:
    """Fee - sigma^2/8 - lambda * tail loss: the LVR expected net yield, penalised by the model-based q-quantile
    IL loss. lambda = 0 is fee_minus_lvr; a large lambda ranks by the tail alone (Net90 is the empirical,
    tail-dominated extreme)."""
    return fee_minus_lvr(history) + risk_aversion * model_il_tail(history.variance, tail_quantile)


def lvr_tail_name(risk_aversion: float) -> str:
    return f"lvr_tail_{risk_aversion:g}"


def mean_net(history: History) -> np.ndarray:
    return quiet(np.nanmean, history.net, axis=1)


def relative_value(history: History) -> np.ndarray:
    """RV: mean of V_LP / V_HODL (= 1 + IL). Check the definition against ref. [22] (Roman et al.)."""
    return 1.0 + quiet(np.nanmean, history.il, axis=1)


def fee_adjusted_relative_value(history: History) -> np.ndarray:
    """FARV: mean of (V_LP + fees) / V_HODL (= 1 + net). Check the definition against ref. [22]."""
    return 1.0 + quiet(np.nanmean, history.net, axis=1)


def score_functions(tail_quantile: float, risk_aversions: tuple[float, ...] = ()) -> dict[str, ScoreFn]:
    lvr_tails = {lvr_tail_name(lam): partial(lvr_tail, risk_aversion=lam, tail_quantile=tail_quantile)
                 for lam in risk_aversions}
    return {
        "net_tail": partial(net_tail, tail_quantile=tail_quantile),
        "fee_minus_lvr": fee_minus_lvr,
        "median_fee": median_fee,
        "low_variance": low_variance,
        "mean_net": mean_net,
        "rv": relative_value,
        "farv": fee_adjusted_relative_value,
        **lvr_tails,
    }
