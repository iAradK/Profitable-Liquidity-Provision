"""E4. Forecast calibration: do the IL bounds have the coverage they claim?

Recursive backtest per pool on calendar months: each month the bound is forecast from past months only
(expanding window, and a rolling window), and an exceedance is a realized value below the forecast.
Reported: raw exceedance rate with a Wilson interval, Kupiec's proportion-of-failures test (is the rate equal
to the target?), Christoffersen's independence test (do exceedances cluster?) and the joint
conditional-coverage test. 'pooled' rows add up all pools (counts and transitions). 'il_tail_model' is the
model-based tail of the LVR mean-tail score (scale mixture of normals over past realized variances).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from experiments.context import Context
from experiments.output import save_table
from experiments.scores import model_il_tail
from experiments.stats import christoffersen_test, kupiec_test, transition_counts, wilson_interval

SIGMA_TARGET = float(sps.norm.cdf(-1.0))

# name -> (forecast from past IL, fees and realized variances, which realized series it bounds)
Forecast = Callable[[np.ndarray, np.ndarray, np.ndarray, float], float]
FORECASTS: dict[str, tuple[Forecast, str]] = {
    "il_tail": (lambda il, fee, var, q: float(np.quantile(il, q)), "il"),
    "il_tail_model": (lambda il, fee, var, q: float(model_il_tail(var[None, :], q)[0]), "il"),
    "il_sigma_bound": (lambda il, fee, var, q: float(il.mean() - il.std(ddof=1)), "il"),
    "il_median": (lambda il, fee, var, q: float(np.median(il)), "il"),
    "il_mean": (lambda il, fee, var, q: float(il.mean()), "il"),
    "net_tail": (lambda il, fee, var, q: float(np.median(fee) + np.quantile(il, q)), "net"),
}


def targets(tail_quantile: float) -> dict[str, float]:
    return {"il_tail": tail_quantile, "il_tail_model": tail_quantile, "il_sigma_bound": SIGMA_TARGET, "il_median": 0.5, "il_mean": 0.5,
            "net_tail": tail_quantile}


def exceedances(il: np.ndarray, fee: np.ndarray, var: np.ndarray, forecast: Forecast, realized: np.ndarray,
                min_train: int, window: int, tail_quantile: float) -> np.ndarray:
    """Boolean exceedance per out-of-sample month; window = 0 means expanding."""
    hits = []
    for t in range(min_train, len(il)):
        start = max(0, t - window) if window else 0
        hits.append(realized[t] < forecast(il[start:t], fee[start:t], var[start:t], tail_quantile))
    return np.array(hits, dtype=bool)


def coverage_row(hits: np.ndarray, counts: np.ndarray, target: float, confidence: float) -> dict:
    n, k = len(hits), int(hits.sum())
    lo, hi = wilson_interval(k, n, confidence)
    kupiec_lr, kupiec_p = kupiec_test(k, n, target)
    ind_lr, ind_p = christoffersen_test(counts)
    return {"n": n, "exceedances": k, "rate": k / n if n else np.nan, "target": target,
            "wilson_lo": lo, "wilson_hi": hi, "kupiec_lr": kupiec_lr, "kupiec_p": kupiec_p,
            "christoffersen_lr": ind_lr, "christoffersen_p": ind_p,
            "conditional_coverage_p": float(sps.chi2.sf(kupiec_lr + ind_lr, 2))}


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    panel = ctx.monthly_panel
    q, confidence = cfg.selection.tail_quantile, cfg.stats.confidence
    schemes = {"expanding": 0, f"rolling_{cfg.calibration.rolling_months}m": cfg.calibration.rolling_months}
    target_by_metric = targets(q)
    rows, pooled = [], {}
    for p, meta in enumerate(panel.pools):
        valid = panel.valid[p]
        il, fee, var = panel.il[p, valid], panel.fee[p, valid], panel.variance[p, valid]
        realized_by_name = {"il": il, "net": il + fee}
        for scheme, window in schemes.items():
            for metric, (forecast, realized_name) in FORECASTS.items():
                hits = exceedances(il, fee, var, forecast, realized_by_name[realized_name],
                                   cfg.calibration.min_train_months, window, q)
                if len(hits) == 0:
                    continue
                counts = transition_counts(hits)
                rows.append({"label": meta.label, "category": meta.category, "scheme": scheme, "metric": metric,
                             **coverage_row(hits, counts, target_by_metric[metric], confidence)})
                all_hits, all_counts = pooled.get((scheme, metric), (np.array([], dtype=bool), np.zeros((2, 2), int)))
                pooled[(scheme, metric)] = (np.concatenate([all_hits, hits]), all_counts + counts)
    for (scheme, metric), (hits, counts) in pooled.items():
        rows.append({"label": "pooled", "category": "", "scheme": scheme, "metric": metric,
                     **coverage_row(hits, counts, target_by_metric[metric], confidence)})
    save_table(pd.DataFrame(rows), out_dir / "exceedance_tests.csv")
    return True
