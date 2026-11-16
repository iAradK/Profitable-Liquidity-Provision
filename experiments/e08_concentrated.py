"""E8. Concentrated ranges: do full-range pool scores still pick good pools for concentrated positions?

Passive positions opened at each month start with range +-w around the entry price, or +-k sigma (sigma: the
pool's trailing monthly log-price volatility from the calibration window), never narrower than one tick spacing.
Fees accrue only in hours when the price is in range, from feeGrowthGlobal (exact fees per unit of liquidity),
so a position's liquidity per dollar follows from its range: no V0/TVL share and no fudge factor.

Scores: the mean-tail family S_lambda over [experiments.selection].risk_aversion (lambda = 0 is fee - sigma^2/8)
and Net90 for comparison. Per range and score: information coefficient of (a) the full-range score and (b) the
same score computed on that range's own history, against the realized concentrated net return. Per range: the
monthly rank correlation between full-range IL and concentrated IL, and the mean fee, IL and net return.
Top-k: the mean return after costs of the top k pools by each full-range score, held as positions of the range,
against all eligible pools and against two fee baselines (highest median fee; highest median fee among the pools with
below-median trailing variance), with paired block-bootstrap tests of the mean and of the CVaR. The picks do not
depend on the range; picks_by_category / picks_summary describe them (categories, two fee tiers of one pair, the
trailing token correlation rho of the held pools). rho_beyond_variance: per range, monthly cross-sectional regressions
(Fama-MacBeth) of the rank of the realized net return on the ranks of the trailing pair variance, fee yield and token
correlation rho, i.e. whether rho carries information beyond the pair variance it enters; pools with constant-price
tokens (rho undefined) are left out. Validation: the hourly fee approximation is compared with exact swap-by-swap fees from the
BigQuery Swap logs for the validation pools and months.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, Horizon
from experiments.context import Context
from experiments.data import TICK_BASE, TICK_SPACING, MarketData
from experiments.correlation import token_returns, trailing_pair_correlation
from experiments.outcomes import Panel, build_panel, position_amounts, position_liquidity, position_outcome
from experiments.output import save_table
from experiments.plotting import SERIES_COLORS, STRATEGY_LABELS, apply_style, plt, save_figure
from experiments.scores import ScoreFn, lvr_tail_name, score_functions
from experiments.selection import (FEE_BASELINES, Backtest, StrategyResult, cost_schedule, information_coefficients,
                                   past_periods, prepare_backtest, run_strategies, score_matrix, selection_rules,
                                   summarize_ic, top_k, with_screened_scores)
from experiments.stats import (bootstrap_mean, finite, newey_west, paired_cvar_difference, paired_difference, quiet,
                               spearman, tail_mean)
from experiments.swaps import decode_swaps, swap_fee_yield

@dataclass(frozen=True)
class RangeSpec:
    name: str
    half_width: float = 0.0       # +-w around the entry price
    sigma_multiple: float = 0.0   # +-k sigma of the log price


def range_specs(half_widths: tuple[float, ...], sigma_multiples: tuple[float, ...]) -> list[RangeSpec]:
    return ([RangeSpec(f"±{w:.0%}", half_width=w) for w in half_widths]
            + [RangeSpec(f"±{k:g}σ", sigma_multiple=k) for k in sigma_multiples])


def min_log_half_width(fee_tier: int) -> float:
    return TICK_SPACING[fee_tier] * math.log(TICK_BASE)


def range_factors(spec: RangeSpec, ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) sqrt-price factors per (pool, period); NaN where sigma is not yet known."""
    panel = ctx.monthly_panel
    lower = np.full((panel.n_pools, panel.n_periods), np.nan)
    upper = np.full_like(lower, np.nan)
    calibration = ctx.config.selection.calibration_months
    for p, meta in enumerate(panel.pools):
        floor = min_log_half_width(meta.fee_tier)
        for t in range(panel.n_periods):
            if spec.half_width:
                down, up = -math.log(1.0 - spec.half_width), math.log(1.0 + spec.half_width)
            else:
                past = past_periods(panel, t, calibration)
                sigma = math.sqrt(quiet(np.nanmean, panel.variance[p, past])) if len(past) else math.nan
                down = up = spec.sigma_multiple * sigma
            if math.isfinite(down):
                lower[p, t] = math.exp(-max(down, floor) / 2.0)
                upper[p, t] = math.exp(max(up, floor) / 2.0)
    return lower, upper


def range_scores(tail_quantile: float, risk_aversions: tuple[float, ...]) -> dict[str, tuple[float, ScoreFn]]:
    """name -> (lambda, score): S_lambda over the grid (lambda = 0 is fee - sigma^2/8), and Net90 (lambda NaN)."""
    scores = score_functions(tail_quantile, risk_aversions)
    family = {"fee_minus_lvr": 0.0} | {lvr_tail_name(lam): lam for lam in risk_aversions}
    return {name: (lam, scores[name]) for name, lam in family.items()} | {"net_tail": (math.nan, scores["net_tail"])}


def period_labels(panel_starts: np.ndarray, periods: np.ndarray) -> np.ndarray:
    return np.array([pd.Timestamp(int(panel_starts[t]), unit="s").strftime("%Y-%m") for t in periods])


def baseline_tests(results: dict[str, StrategyResult], range_name: str, k: int, cfg: ExperimentConfig,
                   rng: np.random.Generator) -> list[dict]:
    """Every strategy against each fee baseline: paired block-bootstrap tests of the mean excess return and of its
    CVaR (a positive CVaR difference is a smaller loss in the bad months)."""
    stats, tail = cfg.stats, cfg.selection.tail_quantile
    rows = []
    for name, result in results.items():
        for baseline in (b for b in FEE_BASELINES if b in results and b != name):
            a, b = result.excess, results[baseline].excess
            mean = paired_difference(a, b, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
            cvar = paired_cvar_difference(a, b, tail, stats.block_months, stats.bootstrap_samples, rng,
                                          stats.confidence)
            rows.append({"range": range_name, "k": k, "strategy": name, "baseline": baseline,
                         "mean_difference": mean["mean"], "mean_lo": mean["lo"], "mean_hi": mean["hi"],
                         "mean_p_value": mean["p_value"], "cvar_difference": cvar["difference"],
                         "cvar_lo": cvar["lo"], "cvar_hi": cvar["hi"], "cvar_p_value": cvar["p_value"],
                         "n": mean["n"]})
    return rows


def top_k_rows(spec: RangeSpec, backtest: Backtest, ctx: Context) -> tuple[list[dict], list[dict]]:
    """Mean monthly excess return after costs (with its CVaR) of the top k pools by each full-range score, held as
    positions of this range; 'equal_weight' (every eligible pool) is the reference. Also without the excluded
    months. Second: the paired tests against the fee baselines."""
    cfg = ctx.config
    schedule = cost_schedule(ctx.market, backtest.panel.starts, cfg.costs)
    no_eth = np.full(backtest.panel.n_periods, np.nan)
    excluded = np.isin(period_labels(backtest.panel.starts, backtest.decisions), cfg.fees.exclude_periods)
    block, rng = cfg.stats.block_months, ctx.rng(80)
    rows, tests = [], []
    for k in cfg.selection.top_k:
        results = run_strategies(backtest, (*backtest.scores, "equal_weight"), k, schedule, no_eth)
        for name, result in results.items():
            test = bootstrap_mean(result.excess, block, cfg.stats.bootstrap_samples, rng, cfg.stats.confidence)
            values = finite(result.excess)
            rows.append({"range": spec.name, "k": k, "strategy": name, "mean_excess": test["mean"],
                         "mean_lo": test["lo"], "mean_hi": test["hi"],
                         "mean_excess_excluding": quiet(np.nanmean, result.excess[~excluded]),
                         "share_positive": quiet(np.nanmean, values > 0),
                         "cvar": tail_mean(values, cfg.selection.tail_quantile) if len(values) else np.nan})
        tests += baseline_tests(results, spec.name, k, cfg, rng)
    return rows, tests


def pick_tables(backtest: Backtest, rho: np.ndarray, ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Who each score picks (the picks are the same for every range): the share of held positions per pool
    category, and per strategy the share of months holding two fee tiers of one pair and the mean trailing token
    correlation of the held pools against that of all eligible pools (pools with an undefined rho left out)."""
    panel, decisions = backtest.panel, backtest.decisions
    categories = np.array([p.category for p in panel.pools])
    families = np.array([p.family for p in panel.pools])
    by_category, summary = [], []
    for k in ks:
        for name, scores in backtest.scores.items():
            held = [top_k(scores[:, t], k) for t in decisions]
            flat = np.concatenate(held)
            for category in sorted(set(categories)):
                by_category.append({"k": k, "strategy": name, "category": category,
                                    "share": float(np.mean(categories[flat] == category)) if len(flat) else np.nan})
            held_rho = np.concatenate([rho[h, t] for h, t in zip(held, decisions)])
            eligible_rho = np.concatenate([rho[backtest.eligible[:, t], t] for t in decisions])
            summary.append({"k": k, "strategy": name,
                            "share_same_pair": float(np.mean([len(set(families[h])) < len(h) for h in held])),
                            "mean_rho_held": quiet(np.nanmean, held_rho),
                            "mean_rho_eligible": quiet(np.nanmean, eligible_rho)})
    return pd.DataFrame(by_category), pd.DataFrame(summary)


def standardized_ranks(x: np.ndarray) -> np.ndarray:
    ranks = pd.Series(x).rank().to_numpy()
    return (ranks - ranks.mean()) / ranks.std()


def trailing_means(full: Panel, decisions: np.ndarray, calibration_periods: int) -> tuple[np.ndarray, np.ndarray]:
    """(variance, fee) of every pool averaged over the calibration window before each decision."""
    variance = np.full((full.n_pools, full.n_periods), np.nan)
    fee = np.full_like(variance, np.nan)
    for t in decisions:
        past = past_periods(full, t, calibration_periods)
        if len(past):
            variance[:, t] = quiet(np.nanmean, full.variance[:, past], axis=1)
            fee[:, t] = quiet(np.nanmean, full.fee[:, past], axis=1)
    return variance, fee


def rho_beyond_variance(range_name: str, panel: Panel, eligible: np.ndarray, decisions: np.ndarray,
                        regressors: dict[str, np.ndarray], cfg: ExperimentConfig) -> dict[str, float]:
    """Fama-MacBeth: each month, OLS of the standardized rank of the realized net return on the standardized ranks
    of the regressors (low variance = minus the trailing variance, fee, rho); mean coefficient and its Newey-West
    t-statistic over the months. Also the plain rank IC of rho."""
    names = list(regressors)
    coefficients, ic_rho = [], []
    for t in decisions:
        columns = [regressors[n][:, t] for n in names]
        mask = eligible[:, t] & np.isfinite(panel.net[:, t]) & np.all([np.isfinite(c) for c in columns], axis=0)
        if mask.sum() < cfg.stats.min_pools_for_ic:
            continue
        x = np.column_stack([np.ones(mask.sum())] + [standardized_ranks(c[mask]) for c in columns])
        y = standardized_ranks(panel.net[mask, t])
        coefficients.append(np.linalg.lstsq(x, y, rcond=None)[0][1:])
        ic_rho.append(spearman(regressors["rho"][mask, t], panel.net[mask, t]))
    row = {"range": range_name, "n_months": len(coefficients)}
    if not coefficients:
        return row
    coefficients = np.array(coefficients)
    for i, name in enumerate(names):
        mean, t_stat = newey_west(coefficients[:, i], cfg.stats.newey_west_lags)
        row |= {f"coef_{name}": mean, f"t_{name}": t_stat}
    mean, t_stat = newey_west(np.array(ic_rho), cfg.stats.newey_west_lags)
    return row | {"ic_rho": mean, "t_ic_rho": t_stat}


def range_results(ctx: Context, specs: list[RangeSpec]) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    full = ctx.monthly_panel
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    scores = range_scores(cfg.selection.tail_quantile, cfg.selection.risk_aversion)
    backtest = prepare_backtest(full, rules, {name: fn for name, (_, fn) in scores.items()})
    decisions, eligible = backtest.decisions, backtest.eligible
    baseline_fns = score_functions(cfg.selection.tail_quantile)
    baselines = {name: score_matrix(full, rules, eligible, baseline_fns[name]) for name in ("median_fee", "low_variance")}
    picking_scores = with_screened_scores(backtest.scores | baselines, eligible)
    rho = trailing_pair_correlation(token_returns(ctx.market, full), full, decisions, rules.calibration_periods)
    variance, fee = trailing_means(full, decisions, rules.calibration_periods)
    regressors = {"low_variance": -variance, "fee": fee, "rho": rho}
    by_category, pick_summary = pick_tables(Backtest(full, rules, eligible, picking_scores), rho, cfg.selection.top_k)
    stats = cfg.stats
    rng = ctx.rng(8)
    ic_args = (stats.newey_west_lags, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
    summary, cells, top_k, tests = [], [], [], []
    rho_rows = [rho_beyond_variance("full", full, eligible, decisions, regressors, cfg)]
    for spec in specs:
        lower, upper = range_factors(spec, ctx)
        panel = build_panel(ctx.market, full.starts, full.ends, full.horizon, lower, upper)
        il_rank = np.array([spearman(full.il[eligible[:, t], t], panel.il[eligible[:, t], t]) for t in decisions])
        rows, range_tests = top_k_rows(spec, Backtest(panel, rules, eligible, picking_scores), ctx)
        top_k += rows
        tests += range_tests
        rho_rows.append(rho_beyond_variance(spec.name, panel, eligible, decisions, regressors, cfg))
        mask = eligible[:, decisions]
        range_stats = {
            "il_rank_corr_mean": quiet(np.nanmean, il_rank),
            "mean_net": quiet(np.nanmean, panel.net[:, decisions][mask]),
            "mean_fee": quiet(np.nanmean, panel.fee[:, decisions][mask]),
            "mean_il": quiet(np.nanmean, panel.il[:, decisions][mask]),
        }
        for name, (lam, score_fn) in scores.items():
            own_scores = score_matrix(panel, rules, eligible, score_fn)
            ic_full = information_coefficients(backtest.scores[name], panel.net, eligible, stats.min_pools_for_ic)
            ic_own = information_coefficients(own_scores, panel.net, eligible, stats.min_pools_for_ic)
            summary.append({
                "range": spec.name, "half_width": spec.half_width, "sigma_multiple": spec.sigma_multiple,
                "score": name, "risk_aversion": lam,
                **{f"full_score_{k}": v for k, v in summarize_ic(ic_full[decisions], *ic_args).items()},
                **{f"own_score_{k}": v for k, v in summarize_ic(ic_own[decisions], *ic_args).items()},
                **range_stats,
            })
        for p, meta in enumerate(panel.pools):
            for t in decisions[eligible[p, decisions]]:
                cells.append({"range": spec.name, "label": meta.label, "category": meta.category,
                              "period": pd.Timestamp(int(panel.starts[t]), unit="s").strftime("%Y-%m"),
                              "il": panel.il[p, t], "fee": panel.fee[p, t], "net": panel.net[p, t],
                              "full_range_net": full.net[p, t]})
    return {"ic_by_range": pd.DataFrame(summary), "concentrated_outcomes": pd.DataFrame(cells),
            "top_k_by_range": pd.DataFrame(top_k), "tests_by_range": pd.DataFrame(tests),
            "picks_by_category": by_category, "picks_summary": pick_summary,
            "rho_beyond_variance": pd.DataFrame(rho_rows)}


def plot_ic(summary: pd.DataFrame, risk_aversions: tuple[float, ...], path_stem: Path) -> None:
    """IC of the full-range scores against concentrated outcomes, per range (mean and 95% CI), for S_lambda at
    `risk_aversions` and Net90."""
    apply_style()
    summary = summary[summary["risk_aversion"].isin(risk_aversions) | summary["risk_aversion"].isna()]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ranges = list(dict.fromkeys(summary["range"]))
    names = list(dict.fromkeys(summary["score"]))
    x = np.arange(len(ranges))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        part = summary[summary["score"] == name].set_index("range").loc[ranges]
        mean = part["full_score_ic_mean"].to_numpy()
        err = np.vstack([mean - part["full_score_ic_lo"], part["full_score_ic_hi"] - mean])
        lam = part["risk_aversion"].iloc[0]
        label = STRATEGY_LABELS.get(name, name) if math.isnan(lam) else f"S, λ={lam:g}"
        ax.errorbar(x - 0.4 + width * (i + 0.5), mean, yerr=err, fmt="o", color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    markersize=4, capsize=1.5, linewidth=1.0, label=label)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x, ranges)
    ax.set(xlabel="Position range", ylabel="IC of the full-range score (mean, 95% CI)")
    ax.legend(loc="lower left", ncols=len(names), fontsize=7)
    save_figure(fig, path_stem)


# ---------------------------------------------------------------- swap-level validation

def validation_table(market: MarketData, swaps: pd.DataFrame, ctx: Context, specs: list[RangeSpec]) -> pd.DataFrame:
    conc = ctx.config.concentrated
    panel = ctx.monthly_panel
    months = np.flatnonzero((panel.starts >= conc.validation_start_ts) & (panel.ends <= conc.validation_end_ts))
    fixed = [RangeSpec("full")] + [s for s in specs if s.half_width]
    rows = []
    for pool in market.pools:
        if pool.meta.address not in conc.validation_pools:
            continue
        pool_swaps = swaps[swaps["pool"] == pool.meta.address]
        grid = pool.grid
        for t in months:
            start, end = int(panel.starts[t]), int(panel.ends[t])
            i0, i1 = int(grid.index_at(start)), int(grid.index_at(end))
            month_swaps = pool_swaps[(pool_swaps["ts"] >= start) & (pool_swaps["ts"] < end)]
            s0, s1 = grid.sqrt_price[i0], grid.sqrt_price[i1]
            for spec in fixed:
                lower, upper = ((0.0, math.inf) if not spec.half_width
                                else (math.sqrt(1 - spec.half_width), math.sqrt(1 + spec.half_width)))
                hourly = position_outcome(pool, i0, i1, lower, upper, market.crediting).fee_yield
                liquidity = (position_liquidity(pool, i0, *position_amounts(s0, lower * s0, upper * s0),
                                                market.crediting) if market.crediting.rule == "fixed_volume" else 0.0)
                exact = swap_fee_yield(month_swaps, s0, s1, grid.liquidity[i0], pool.meta.fee_rate, lower * s0, upper * s0,
                                       liquidity)
                rows.append({"label": pool.meta.label, "period": pd.Timestamp(start, unit="s").strftime("%Y-%m"),
                             "range": spec.name, "n_swaps": len(month_swaps), "fee_yield_hourly": hourly,
                             "fee_yield_swaps": exact, "rel_error": hourly / exact - 1.0 if exact else math.nan})
    return pd.DataFrame(rows)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    specs = range_specs(cfg.concentrated.half_widths, cfg.concentrated.sigma_multiples)
    tables = range_results(ctx, specs)
    for name, table in tables.items():
        save_table(table, out_dir / f"{name}.csv")
    plot_ic(tables["ic_by_range"], cfg.concentrated.plot_risk_aversion, out_dir / "ic_vs_range")
    if cfg.data.swaps_file is None:
        return True
    swaps = decode_swaps((cfg.data.swaps_file,), cfg.cache_dir / "swaps.parquet")
    validation = validation_table(ctx.market, swaps, ctx, specs)
    save_table(validation, out_dir / "swap_validation.csv")
    if not validation.empty:
        by_range = validation.groupby("range")["rel_error"].agg(
            median_abs_rel_error=lambda e: e.abs().median(), max_abs_rel_error=lambda e: e.abs().max(), n="count")
        save_table(by_range, out_dir / "swap_validation_summary.csv", index=True)
    return True
