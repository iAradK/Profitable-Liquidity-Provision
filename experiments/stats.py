"""Statistics: moving-block bootstrap, coverage tests (Wilson, Kupiec, Christoffersen), Newey-West,
rank correlation, AR fits, the Lo-MacKinlay variance-ratio test and strategy performance metrics."""

import math
import warnings
from collections.abc import Callable

import numpy as np
from scipy import stats as sps
from scipy.special import xlogy


def finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def quiet(fn: Callable[..., np.ndarray], *args, **kwargs) -> np.ndarray:
    """Run a nan-aware NumPy reduction without 'All-NaN slice' / 'Mean of empty slice' warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- bootstrap

def block_bootstrap_indices(n: int, block: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """(n_samples, n) indices built from random moving blocks of length `block`."""
    block = int(min(max(block, 1), n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_samples, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_samples, -1)[:, :n]


def percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lo, hi = quiet(np.nanquantile, samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_mean(x: np.ndarray, block: int, n_samples: int, rng: np.random.Generator,
                   confidence: float) -> dict[str, float]:
    """Mean with a moving-block bootstrap interval and a two-sided p-value for mean = 0."""
    x = finite(x)
    if len(x) < 3:
        return {"mean": float(np.mean(x)) if len(x) else math.nan, "lo": math.nan, "hi": math.nan,
                "p_value": math.nan, "n": len(x)}
    means = x[block_bootstrap_indices(len(x), block, n_samples, rng)].mean(axis=1)
    mean = float(x.mean())
    lo, hi = percentile_interval(means, confidence)
    p_value = float(np.mean(np.abs(means - mean) >= abs(mean)))
    return {"mean": mean, "lo": lo, "hi": hi, "p_value": p_value, "n": len(x)}


def paired_difference(a: np.ndarray, b: np.ndarray, block: int, n_samples: int, rng: np.random.Generator,
                      confidence: float) -> dict[str, float]:
    """Block-bootstrap test of mean(a - b) = 0 over the periods where both are defined."""
    both = np.isfinite(a) & np.isfinite(b)
    return bootstrap_mean(np.asarray(a)[both] - np.asarray(b)[both], block, n_samples, rng, confidence)


# ---------------------------------------------------------------- coverage tests

def wilson_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = sps.norm.ppf(0.5 + confidence / 2.0)
    phat = k / n
    center = (phat + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def kupiec_test(k: int, n: int, target: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio: is the exceedance rate equal to `target`?"""
    if n == 0:
        return math.nan, math.nan
    phat = k / n
    ll_null = xlogy(n - k, 1 - target) + xlogy(k, target)
    ll_alt = xlogy(n - k, 1 - phat) + xlogy(k, phat)
    lr = float(-2.0 * (ll_null - ll_alt))
    return lr, float(sps.chi2.sf(lr, 1))


def transition_counts(hits: np.ndarray) -> np.ndarray:
    """2x2 counts of (previous, next) exceedance states; sum them across series to pool."""
    h = np.asarray(hits, dtype=int)
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, (h[:-1], h[1:]), 1)
    return counts


def christoffersen_test(counts: np.ndarray) -> tuple[float, float]:
    """Independence likelihood ratio: does an exceedance make the next one more likely?"""
    (n00, n01), (n10, n11) = counts
    total = n00 + n01 + n10 + n11
    if total == 0:
        return math.nan, math.nan
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / total
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = float(-2.0 * (ll_null - ll_alt))
    return lr, float(sps.chi2.sf(lr, 1))


# ---------------------------------------------------------------- correlation and t-statistics

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 3 or np.ptp(np.asarray(a)[both]) == 0 or np.ptp(np.asarray(b)[both]) == 0:
        return math.nan
    return float(sps.spearmanr(np.asarray(a)[both], np.asarray(b)[both]).statistic)


def newey_west(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Mean and its HAC (Bartlett kernel) t-statistic."""
    x = finite(x)
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else math.nan), math.nan
    e = x - x.mean()
    long_run = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        long_run += 2.0 * (1.0 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    se = math.sqrt(long_run / n) if long_run > 0 else math.nan
    return float(x.mean()), float(x.mean() / se) if se else math.nan


# ---------------------------------------------------------------- time-series models

def fit_ar(x: np.ndarray, order: int) -> dict[str, float]:
    """OLS AR(order) with intercept: coefficients phi1..phiN, AIC and BIC."""
    y = x[order:]
    columns = [np.ones(len(y))] + [x[order - lag:len(x) - lag] for lag in range(1, order + 1)]
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss = float(np.sum((y - design @ coef) ** 2))
    n, k = len(y), order + 1
    log_term = n * math.log(rss / n) if rss > 0 else -math.inf
    out = {f"phi{i}": float(coef[i]) for i in range(1, order + 1)}
    out.update(aic=log_term + 2 * k, bic=log_term + k * math.log(n), n=n)
    return out


def half_life(phi: float, phi_max: float = 1.0) -> float:
    """Days for a deviation to halve under AR(1) persistence phi; NaN outside (0, phi_max)."""
    return -math.log(2.0) / math.log(phi) if 0.0 < phi < phi_max else math.nan


def variance_ratio_test(log_prices: np.ndarray, q: int) -> tuple[float, float, float]:
    """Lo-MacKinlay variance ratio VR(q) with the heteroskedasticity-robust z* statistic and p-value."""
    r = np.diff(log_prices)
    n = len(r)
    if n <= 2 * q:
        return math.nan, math.nan, math.nan
    dev = r - r.mean()
    var1 = dev @ dev / (n - 1)
    multi = log_prices[q:] - log_prices[:-q] - q * r.mean()
    m = q * (n - q + 1) * (1 - q / n)
    vr = (multi @ multi / m) / var1
    denom = (dev @ dev) ** 2
    theta = sum((2.0 * (q - j) / q) ** 2 * n * np.sum(dev[j:] ** 2 * dev[:-j] ** 2) / denom for j in range(1, q))
    z = (vr - 1.0) / math.sqrt(theta)
    return float(vr), float(z), float(2.0 * sps.norm.sf(abs(z)))


# ---------------------------------------------------------------- performance

def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + finite(returns))
    if len(wealth) == 0:
        return math.nan
    return float(np.min(wealth / np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:] - 1.0))


def performance(returns: np.ndarray, periods_per_year: float, tail: float = 0.10) -> dict[str, float]:
    """Return and risk metrics; `tail` sets the quantile of the per-period VaR / CVaR (expected shortfall)."""
    r = finite(returns)
    if len(r) < 2:
        return {"n": len(r)}
    std = r.std(ddof=1)
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    cumulative = float(np.prod(1.0 + r) - 1.0)
    var = float(np.quantile(r, tail))
    return {
        "n": len(r),
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "cumulative": cumulative,
        "annualized": float((1.0 + cumulative) ** (periods_per_year / len(r)) - 1.0) if cumulative > -1 else -1.0,
        "sharpe": float(r.mean() / std * math.sqrt(periods_per_year)) if std > 0 else math.nan,
        "sortino": float(r.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else math.nan,
        "max_drawdown": max_drawdown(r),
        "share_positive": float(np.mean(r > 0)),
        "var": var,
        "cvar": float(r[r <= var].mean()),
        "worst": float(r.min()),
    }
