"""E14. Out-of-sample value of the token correlation: does adding rho to a pool score improve the pools an LP picks?

E8 (rho_beyond_variance) finds, averaging monthly cross-sectional regressions over the whole sample, that the
trailing token correlation rho ranks pools beyond the trailing pair variance and fee yield. This test was added after
that result. It asks whether an LP could have used it in real time: at each decision the score is the fitted value of
the monthly regressions

    rank(realized net return) ~ rank(-trailing variance) + rank(trailing fee) [+ rank(trailing rho)]

with the coefficients averaged over past months only (an expanding window, at least min_training_months), fitted
separately for every position range (full range and the E8 ranges). Both models use the same pools: eligible ones
with a defined rho (the subgraph prices stablecoins at exactly 1 USD, so pools of two stablecoins are left out).
Per range and k: the top-k portfolios after costs (mean with its interval, CVaR), paired block-bootstrap tests of
with_rho minus without_rho (mean and CVaR), and the out-of-sample information coefficient of each model with a
Newey-West test of their difference.

Outputs: performance.csv, tests.csv, ic.csv, coefficients.csv (mean coefficients of the last fit, per range).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, Horizon
from experiments.context import Context
from experiments.correlation import token_returns, trailing_pair_correlation
from experiments.e08_concentrated import range_factors, range_specs, standardized_ranks, trailing_means
from experiments.outcomes import Panel, build_panel
from experiments.output import save_table
from experiments.selection import cost_schedule, hold, prepare_backtest, selection_rules, top_k
from experiments.stats import (bootstrap_mean, finite, newey_west, paired_cvar_difference, paired_difference,
                               spearman, tail_mean)

log = logging.getLogger(__name__)

MODELS = {"with_rho": ("low_variance", "fee", "rho"), "without_rho": ("low_variance", "fee")}


def universe(eligible: np.ndarray, regressors: dict[str, np.ndarray]) -> np.ndarray:
    """Eligible pools with every regressor defined (the same pools for both models)."""
    return eligible & np.all([np.isfinite(r) for r in regressors.values()], axis=0)


def design(regressors: dict[str, np.ndarray], names: tuple[str, ...], mask: np.ndarray, t: int) -> np.ndarray:
    return np.column_stack([np.ones(mask.sum())] + [standardized_ranks(regressors[n][mask, t]) for n in names])


def out_of_sample_scores(panel: Panel, pools: np.ndarray, decisions: np.ndarray, regressors: dict[str, np.ndarray],
                         names: tuple[str, ...], min_pools: int, min_training: int
                         ) -> tuple[np.ndarray, np.ndarray | None]:
    """(scores, last mean coefficients). The score at decision t is the fitted value with the coefficients averaged
    over the monthly regressions of the decisions whose outcome was known by the start of t."""
    scores = np.full((panel.n_pools, panel.n_periods), np.nan)
    fitted: dict[int, np.ndarray] = {}
    mean_coefficients = None
    for t in decisions:
        mask = pools[:, t] & np.isfinite(panel.net[:, t])
        past = [s for s in fitted if panel.ends[s] <= panel.starts[t]]
        if len(past) >= min_training and pools[:, t].sum() >= min_pools:
            mean_coefficients = np.mean([fitted[s] for s in past], axis=0)
            scores[pools[:, t], t] = design(regressors, names, pools[:, t], t) @ mean_coefficients
        if mask.sum() >= min_pools:
            y = standardized_ranks(panel.net[mask, t])
            fitted[t] = np.linalg.lstsq(design(regressors, names, mask, t), y, rcond=None)[0]
    return scores, mean_coefficients


def range_rows(range_name: str, panel: Panel, pools: np.ndarray, decisions: np.ndarray,
               regressors: dict[str, np.ndarray], ctx: Context) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    cfg: ExperimentConfig = ctx.config
    stats, tail = cfg.stats, cfg.selection.tail_quantile
    schedule = cost_schedule(ctx.market, panel.starts, cfg.costs)
    rng = ctx.rng(140)
    scores, coefficients = {}, []
    for model, names in MODELS.items():
        scores[model], last = out_of_sample_scores(panel, pools, decisions, regressors, names, stats.min_pools_for_ic,
                                                   cfg.rho_score.min_training_months)
        if last is not None:
            coefficients.append({"range": range_name, "model": model,
                                 **{f"coef_{n}": c for n, c in zip(names, last[1:])}})
    tested = decisions[np.isfinite(scores["with_rho"][:, decisions]).any(axis=0)]
    ic = {model: np.array([spearman(s[np.isfinite(s[:, t]) & np.isfinite(panel.net[:, t]), t],
                                    panel.net[np.isfinite(s[:, t]) & np.isfinite(panel.net[:, t]), t])
                           for t in tested]) for model, s in scores.items()}
    ic_rows = []
    for model, values in ic.items():
        mean, t_stat = newey_west(values, stats.newey_west_lags)
        ic_rows.append({"range": range_name, "model": model, "ic_mean": mean, "ic_nw_t": t_stat,
                        "n_months": len(finite(values))})
    mean, t_stat = newey_west(ic["with_rho"] - ic["without_rho"], stats.newey_west_lags)
    ic_rows.append({"range": range_name, "model": "with_minus_without", "ic_mean": mean, "ic_nw_t": t_stat,
                    "n_months": len(finite(ic["with_rho"] - ic["without_rho"]))})
    perf, tests = [], []
    for k in cfg.selection.top_k:
        results = {model: hold(model, panel, tested, tuple(top_k(s[:, t], k) for t in tested), schedule)
                   for model, s in scores.items()}
        for model, result in results.items():
            interval = bootstrap_mean(result.excess, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
            values = finite(result.excess)
            perf.append({"range": range_name, "k": k, "model": model, "mean": interval["mean"],
                         "mean_lo": interval["lo"], "mean_hi": interval["hi"],
                         "cvar": tail_mean(values, tail) if len(values) else np.nan, "n_months": len(values)})
        a, b = results["with_rho"].excess, results["without_rho"].excess
        mean = paired_difference(a, b, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
        cvar = paired_cvar_difference(a, b, tail, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
        tests.append({"range": range_name, "k": k, "mean_difference": mean["mean"], "mean_lo": mean["lo"],
                      "mean_hi": mean["hi"], "mean_p_value": mean["p_value"], "cvar_difference": cvar["difference"],
                      "cvar_lo": cvar["lo"], "cvar_hi": cvar["hi"], "cvar_p_value": cvar["p_value"],
                      "n": mean["n"]})
    return perf, tests, ic_rows, coefficients


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    full = ctx.monthly_panel
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    backtest = prepare_backtest(full, rules, {})
    decisions = backtest.decisions
    rho = trailing_pair_correlation(token_returns(ctx.market, full), full, decisions, rules.calibration_periods)
    variance, fee = trailing_means(full, decisions, rules.calibration_periods)
    regressors = {"low_variance": -variance, "fee": fee, "rho": rho}
    pools = universe(backtest.eligible, regressors)
    panels = [("full", full)]
    for spec in range_specs(cfg.concentrated.half_widths, cfg.concentrated.sigma_multiples):
        lower, upper = range_factors(spec, ctx)
        panels.append((spec.name, build_panel(ctx.market, full.starts, full.ends, full.horizon, lower, upper)))
    tables: dict[str, list[dict]] = {"performance": [], "tests": [], "ic": [], "coefficients": []}
    for name, panel in panels:
        for table, rows in zip(tables.values(), range_rows(name, panel, pools, decisions, regressors, ctx)):
            table += rows
        log.info("E14: %s done", name)
    for name, rows in tables.items():
        save_table(pd.DataFrame(rows), out_dir / f"{name}.csv")
    return True
