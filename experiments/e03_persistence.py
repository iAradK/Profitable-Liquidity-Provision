"""E3. Robust persistence analysis of the daily log price ratio.

  - rolling AR(1) (as in the paper): mean phi and half-life with block-bootstrap intervals, and the share of
    windows excluded because phi is outside (0, phi_max);
  - rolling ADF (H0: unit root) and KPSS (H0: stationary) rejection rates;
  - AR(1) versus AR(2) by AIC/BIC on the full sample and on both sides of the regime split;
  - Lo-MacKinlay variance-ratio tests on daily log returns (robust near a unit root).
Requires statsmodels (pip install statsmodels).
"""

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from experiments.context import Context
from experiments.data import DAY
from experiments.output import cached_map, save_table
from experiments.stats import block_bootstrap_indices, fit_ar, half_life, percentile_interval, variance_ratio_test

MIN_REGIME_DAYS = 90


@dataclass(frozen=True)
class PersistenceInput:
    label: str
    category: str
    log_prices: np.ndarray
    split_index: int
    window_days: int
    step_days: int
    phi_max: float
    variance_ratio_lags: tuple[int, ...]
    alpha: float
    n_samples: int
    confidence: float
    seed: int


def unit_root_pvalues(x: np.ndarray) -> tuple[float, float]:
    """(ADF p-value, KPSS p-value). KPSS p-values are interpolated within [0.01, 0.10]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x, autolag="AIC")[1]), float(kpss(x, regression="c", nlags="auto")[1])


def rolling_windows(inp: PersistenceInput) -> pd.DataFrame:
    x = inp.log_prices
    rows = []
    for start in range(0, len(x) - inp.window_days + 1, inp.step_days):
        window = x[start:start + inp.window_days]
        if np.ptp(window) == 0:  # no trades in the whole window (forward-filled price): nothing to fit
            rows.append({"phi": math.nan, "half_life": math.nan, "adf_p": math.nan, "kpss_p": math.nan,
                         "constant": True})
            continue
        phi = fit_ar(window, 1)["phi1"]
        adf_p, kpss_p = unit_root_pvalues(window)
        rows.append({"phi": phi, "half_life": half_life(phi, inp.phi_max), "adf_p": adf_p, "kpss_p": kpss_p,
                     "constant": False})
    return pd.DataFrame(rows)


def bootstrap_window_means(values: np.ndarray, block: int, inp: PersistenceInput) -> tuple[float, float]:
    if len(values) < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(inp.seed)
    means = values[block_bootstrap_indices(len(values), block, inp.n_samples, rng)].mean(axis=1)
    return percentile_interval(means, inp.confidence)


def regime_row(name: str, x: np.ndarray, phi_max: float) -> dict:
    ar1, ar2 = fit_ar(x, 1), fit_ar(x, 2)
    adf_p, kpss_p = unit_root_pvalues(x)
    persistence2 = ar2["phi1"] + ar2["phi2"]
    return {
        "regime": name, "n_days": len(x),
        "ar1_phi": ar1["phi1"], "ar1_half_life": half_life(ar1["phi1"], phi_max),
        "ar2_phi1": ar2["phi1"], "ar2_phi2": ar2["phi2"], "ar2_persistence": persistence2,
        "ar2_half_life": half_life(persistence2, phi_max),
        "aic_prefers": "AR(2)" if ar2["aic"] < ar1["aic"] else "AR(1)",
        "bic_prefers": "AR(2)" if ar2["bic"] < ar1["bic"] else "AR(1)",
        "adf_p": adf_p, "kpss_p": kpss_p,
    }


def analyse_pool(inp: PersistenceInput) -> dict:
    windows = rolling_windows(inp)
    n_constant = int(windows["constant"].sum())
    windows = windows.loc[~windows["constant"]]
    valid = windows["half_life"].notna()
    block = math.ceil(inp.window_days / inp.step_days)
    hl_lo, hl_hi = bootstrap_window_means(windows.loc[valid, "half_life"].to_numpy(), block, inp)
    phi_lo, phi_hi = bootstrap_window_means(windows.loc[valid, "phi"].to_numpy(), block, inp)
    summary = {
        "label": inp.label, "category": inp.category, "n_days": len(inp.log_prices), "n_windows": len(windows),
        "n_constant_windows": n_constant,
        "share_excluded": float(1.0 - valid.mean()) if len(windows) else math.nan,
        "mean_phi": windows.loc[valid, "phi"].mean(), "mean_phi_lo": phi_lo, "mean_phi_hi": phi_hi,
        "mean_half_life": windows.loc[valid, "half_life"].mean(), "mean_half_life_lo": hl_lo,
        "mean_half_life_hi": hl_hi, "median_half_life": windows.loc[valid, "half_life"].median(),
        "share_adf_rejects_unit_root": float((windows["adf_p"] < inp.alpha).mean()),
        "share_kpss_rejects_stationarity": float((windows["kpss_p"] < inp.alpha).mean()),
    }
    x = inp.log_prices
    regimes = [regime_row("full", x, inp.phi_max)]
    for name, part in (("pre_split", x[:inp.split_index]), ("post_split", x[inp.split_index:])):
        if len(part) >= MIN_REGIME_DAYS:
            regimes.append(regime_row(name, part, inp.phi_max))
    variance_ratios = []
    for q in inp.variance_ratio_lags:
        vr, z, p = variance_ratio_test(x, q)
        variance_ratios.append({"q": q, "variance_ratio": vr, "z_robust": z, "p_value": p})
    return {"summary": summary,
            "regimes": [{"label": inp.label, **r} for r in regimes],
            "variance_ratio": [{"label": inp.label, **r} for r in variance_ratios]}


def run(ctx: Context, out_dir: Path) -> bool:
    cfg, pers = ctx.config, ctx.config.persistence
    end = min(cfg.sample.end_ts, ctx.market.data_end_ts)
    days = np.arange(cfg.sample.start_ts, end + 1, DAY)
    items = []
    for p, pool in enumerate(ctx.market.pools):
        idx = pool.grid.index_at(days)
        exists = idx >= 0
        if exists.sum() < pers.window_days + 10:
            continue
        log_prices = 2.0 * np.log(pool.grid.sqrt_price[idx[exists]])
        split_index = int(np.sum(days[exists] < cfg.sample.regime_split_ts))
        items.append((pool.meta.address, PersistenceInput(
            pool.meta.label, pool.meta.category, log_prices, split_index, pers.window_days, pers.step_days,
            pers.phi_max, pers.variance_ratio_lags, pers.test_alpha, cfg.stats.bootstrap_samples,
            cfg.stats.confidence, cfg.seed + p)))
    results = cached_map(items, analyse_pool, out_dir / "pools", cfg.max_workers, cfg.force_refresh)
    ordered = [results[key] for key, _ in items]
    save_table(pd.DataFrame([r["summary"] for r in ordered]), out_dir / "rolling_ar1_summary.csv")
    save_table(pd.DataFrame([row for r in ordered for row in r["regimes"]]), out_dir / "regimes_ar1_vs_ar2.csv")
    save_table(pd.DataFrame([row for r in ordered for row in r["variance_ratio"]]), out_dir / "variance_ratio.csv")
    return True
