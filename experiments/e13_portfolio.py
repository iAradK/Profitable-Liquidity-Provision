"""E13. Correlation-aware portfolios: does the token correlation model improve the choice of k pools together?

The E5 top-k strategies rank every pool alone by S_lambda = fee - sigma^2/8 + lambda * IL_q and hold the k best,
so the portfolio's tail is taken as the mean of the pools' own tails. The joint IL of k positions depends on how
their price ratios move together, which a single pool's history cannot give: two fee tiers of one pair have the
same IL, and pairs that share a token or whose tokens are correlated lose together. Here the k pools are chosen
greedily to maximise

    mean_{i in P} (fee_i - sigma_i^2 / 8) + lambda * q-quantile of the portfolio IL  mean_{i in P} IL_i,

with the portfolio IL simulated from the past data only. Each scenario draws one past month of the calibration
window for all pools at once (volatility regimes are shared across pools; every pool's marginal is the same scale
mixture of normals as in S_lambda) and the pools' log price ratio changes x ~ N(0, D C D), D = the pools' realized
volatilities of that month; IL_i = sech(x_i / 2) - 1. Variants, all on the [experiments.selection] lambda grid
(lambda = 0 is the same for all of them and is left out):
  topk        hold the k best by S_lambda (E5)
  one_per_pair  the same with at most one fee tier of each token pair (a rule that needs no correlation model)
  joint_indep   greedy with C = I: the shared volatility regime only, no correlation model
  joint_corr    greedy with C from the token model: correlations of the pools' daily price-ratio returns over the
                calibration window (experiments.correlation)
joint_corr against joint_indep isolates the correlation model; against one_per_pair it tests whether the model
does more than remove duplicate fee tiers.

Outputs: performance.csv (per k, lambda and variant: mean excess return after costs with its interval, CVaR, the
share of months holding two tiers of one pair, the mean model correlation between held pools), tests.csv (paired
block-bootstrap tests of the mean and the CVaR), returns.csv (monthly returns and picks).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, Horizon
from experiments.context import Context
from experiments.correlation import calibration_window, ratio_correlation_matrix, ratio_weights, token_returns
from experiments.output import save_table
from experiments.scores import fee_minus_lvr, lvr_tail
from experiments.selection import (Backtest, StrategyResult, cost_schedule, hold, history_at, prepare_backtest,
                                   selection_rules, top_k)
from experiments.stats import bootstrap_mean, finite, paired_cvar_difference, paired_difference, quiet, tail_mean

log = logging.getLogger(__name__)

VARIANTS = ("topk", "one_per_pair", "joint_indep", "joint_corr")
COMPARISONS = (("joint_corr", "topk"), ("joint_corr", "joint_indep"), ("joint_corr", "one_per_pair"),
               ("one_per_pair", "topk"))


def scenario_il(variance: np.ndarray, correlation: np.ndarray, n_scenarios: int,
                rng: np.random.Generator) -> np.ndarray:
    """(n_scenarios, n_pools) simulated full-range IL. variance: (n_pools, n_past) realized monthly variances."""
    fallback = quiet(np.nanmean, variance, axis=1)
    variance = np.where(np.isfinite(variance), variance, fallback[:, None])
    months = rng.integers(0, variance.shape[1], n_scenarios)
    values, vectors = np.linalg.eigh(correlation)
    z = rng.standard_normal((n_scenarios, len(correlation))) @ (vectors * np.sqrt(np.clip(values, 0.0, None))).T
    x = z * np.sqrt(np.clip(variance[:, months].T, 0.0, None))
    return 1.0 / np.cosh(x / 2.0) - 1.0


def greedy_portfolio(base: np.ndarray, il: np.ndarray, candidates: np.ndarray, k: int, risk_aversion: float,
                     tail_quantile: float) -> np.ndarray:
    """Add one pool at a time, the one that maximises mean(base) + lambda * quantile(mean IL) of the set."""
    chosen: list[int] = []
    total = np.zeros(il.shape[0])
    remaining = list(candidates)
    while remaining and len(chosen) < k:
        n = len(chosen) + 1
        values = [(base[chosen].sum() + base[c]) / n
                  + risk_aversion * np.quantile((total + il[:, c]) / n, tail_quantile) for c in remaining]
        best = remaining.pop(int(np.argmax(values)))
        chosen.append(best)
        total += il[:, best]
    return np.array(chosen, dtype=int)


def one_per_pair(scores: np.ndarray, families: np.ndarray, k: int) -> np.ndarray:
    held, seen = [], set()
    for p in top_k(scores, len(scores)):
        if families[p] not in seen:
            held.append(p)
            seen.add(families[p])
        if len(held) == k:
            break
    return np.array(held, dtype=int)


Key = tuple[str, float]  # (variant, lambda)


def portfolio_holdings(backtest: Backtest, ctx: Context, k: int
                       ) -> tuple[dict[Key, list[np.ndarray]], dict[Key, list[float]]]:
    """Holdings per (variant, lambda) and decision, plus the mean model correlation of each held set."""
    cfg = ctx.config
    sel, panel = cfg.selection, backtest.panel
    returns = token_returns(ctx.market, panel)
    weights = ratio_weights(returns, panel.pools)
    families = np.array([p.family for p in panel.pools])
    rng = ctx.rng(130 + k)
    holdings: dict[Key, list[np.ndarray]] = {(v, lam): [] for v in VARIANTS for lam in sel.risk_aversion}
    held_correlation: dict[Key, list[float]] = {key: [] for key in holdings}
    for t in backtest.decisions:
        eligible = np.flatnonzero(backtest.eligible[:, t])
        window = calibration_window(panel, t, backtest.rules.calibration_periods)
        history = history_at(panel, t, backtest.rules.calibration_periods)
        base = fee_minus_lvr(history)
        candidates = eligible[np.isfinite(base[eligible])]
        if window is None or len(candidates) == 0:
            for key in holdings:
                holdings[key].append(np.array([], dtype=int))
            continue
        correlation = ratio_correlation_matrix(returns, weights, *window)
        sub = np.ix_(candidates, candidates)
        il = {"joint_indep": scenario_il(history.variance[candidates], np.eye(len(candidates)),
                                         cfg.portfolio.scenarios, rng),
              "joint_corr": scenario_il(history.variance[candidates], correlation[sub], cfg.portfolio.scenarios, rng)}
        for lam in sel.risk_aversion:
            scores = np.where(backtest.eligible[:, t], lvr_tail(history, lam, sel.tail_quantile), np.nan)
            picks = {"topk": top_k(scores, k), "one_per_pair": one_per_pair(scores, families, k)}
            for variant in ("joint_indep", "joint_corr"):
                local = greedy_portfolio(base[candidates], il[variant], np.arange(len(candidates)), k, lam,
                                         sel.tail_quantile)
                picks[variant] = candidates[local]
            for variant, held in picks.items():
                holdings[(variant, lam)].append(held)
                pairs = correlation[np.ix_(held, held)][np.triu_indices(len(held), 1)]
                held_correlation[(variant, lam)].append(float(np.mean(np.abs(pairs))) if len(pairs) else np.nan)
    return holdings, held_correlation


def run_portfolios(ctx: Context, backtest: Backtest) -> tuple[list[dict], list[dict], list[dict]]:
    cfg = ctx.config
    stats, tail = cfg.stats, cfg.selection.tail_quantile
    panel, decisions = backtest.panel, backtest.decisions
    schedule = cost_schedule(ctx.market, panel.starts, cfg.costs)
    families = np.array([p.family for p in panel.pools])
    rng = ctx.rng(13)
    perf, tests, monthly = [], [], []
    for k in (k for k in cfg.selection.top_k if k > 1):
        holdings, held_correlation = portfolio_holdings(backtest, ctx, k)
        results: dict[Key, StrategyResult] = {
            key: hold(f"{key[0]}_{key[1]:g}", panel, decisions, tuple(held), schedule) for key, held in holdings.items()}
        for (variant, lam), result in results.items():
            interval = bootstrap_mean(result.excess, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
            values = finite(result.excess)
            perf.append({"k": k, "risk_aversion": lam, "variant": variant, "mean": interval["mean"],
                         "mean_lo": interval["lo"], "mean_hi": interval["hi"],
                         "cvar": tail_mean(values, tail) if len(values) else np.nan,
                         "share_same_pair": float(np.mean([len(set(families[h])) < len(h)
                                                           for h in holdings[(variant, lam)] if len(h)])),
                         "mean_abs_correlation_held": quiet(np.nanmean, np.array(held_correlation[(variant, lam)]))})
            for j, t in enumerate(decisions):
                monthly.append({"period": pd.Timestamp(int(panel.starts[t]), unit="s").strftime("%Y-%m"), "k": k,
                                "risk_aversion": lam, "variant": variant, "excess": result.excess[j],
                                "picks": "; ".join(panel.pools[p].label for p in holdings[(variant, lam)][j])})
        for lam in cfg.selection.risk_aversion:
            for a, b in COMPARISONS:
                x, y = results[(a, lam)].excess, results[(b, lam)].excess
                mean = paired_difference(x, y, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
                cvar = paired_cvar_difference(x, y, tail, stats.block_months, stats.bootstrap_samples, rng,
                                              stats.confidence)
                tests.append({"k": k, "risk_aversion": lam, "variant": a, "baseline": b,
                              "mean_difference": mean["mean"], "mean_lo": mean["lo"], "mean_hi": mean["hi"],
                              "mean_p_value": mean["p_value"], "cvar_difference": cvar["difference"],
                              "cvar_lo": cvar["lo"], "cvar_hi": cvar["hi"], "cvar_p_value": cvar["p_value"],
                              "n": mean["n"]})
        log.info("E13: k = %d done", k)
    return perf, tests, monthly


def run(ctx: Context, out_dir: Path) -> bool:
    cfg: ExperimentConfig = ctx.config
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    backtest = prepare_backtest(ctx.monthly_panel, rules, {})  # eligibility only; scores are computed per variant
    perf, tests, monthly = run_portfolios(ctx, backtest)
    save_table(pd.DataFrame(perf), out_dir / "performance.csv")
    save_table(pd.DataFrame(tests), out_dir / "tests.csv")
    save_table(pd.DataFrame(monthly), out_dir / "returns.csv")
    return True
