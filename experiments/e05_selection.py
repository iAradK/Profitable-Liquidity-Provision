"""E5. Rolling out-of-sample pool selection (the main experiment).

Each period t, every eligible pool is scored with data that ended by the start of t; the top k pools are held
for one period with entry/exit gas and the entry swap fee deducted. Strategies: Net90 ('net_tail') and the
baselines in [experiments.selection].strategies (fee - sigma^2/8 is the key one), plus random top-k portfolios.

The LVR mean-tail family fee - sigma^2/8 - lambda * tail ('lvr_tail_<lambda>', model-based tail, lambda grid in
[experiments.selection].risk_aversion) runs alongside; lambda = 0 is fee - sigma^2/8, Net90 the empirical extreme.

Outputs per k: monthly returns, performance table with risk metrics (VaR/CVaR at the tail quantile, worst month,
drawdown; on the return relative to HODL and on the USD return), paired block-bootstrap tests of Net90 against
every baseline, the random-portfolio distribution, subperiods, a cumulative-return figure, and the mean-tail
risk-return frontier (table and figure). The information coefficient (score at t vs realized net at t, before costs)
is computed for every score, per pool and per pair family.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig
from experiments.context import Context
from experiments.data import MarketData
from experiments.outcomes import Panel, build_panel, sample_periods
from experiments.output import save_table
from experiments.plotting import STRATEGY_LABELS, apply_style, plt, save_figure, strategy_color
from experiments.scores import lvr_tail_name, score_functions
from experiments.selection import (Backtest, StrategyResult, cost_schedule, eth_returns,
                                   family_information_coefficients, information_coefficients, prepare_backtest,
                                   random_portfolios, run_strategies, selection_rules, summarize_ic)
from experiments.stats import paired_difference, performance

MAIN_STRATEGY = "net_tail"


def month_label(ts: np.ndarray) -> list[str]:
    return [pd.Timestamp(int(t), unit="s").strftime("%Y-%m-%d") for t in ts]


def ic_tables(backtest: Backtest, config: ExperimentConfig, block: int, rng: np.random.Generator
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, stats = backtest.panel, config.stats
    families = [p.family for p in panel.pools]
    by_period, summary = {}, []
    for name, scores in backtest.scores.items():
        per_pool = information_coefficients(scores, panel.net, backtest.eligible, stats.min_pools_for_ic)
        per_family = family_information_coefficients(scores, panel.net, backtest.eligible, families,
                                                     stats.min_pools_for_ic)
        by_period[name] = per_pool[backtest.decisions]
        for level, ic in (("pool", per_pool), ("family", per_family)):
            summary.append({"score": name, "level": level,
                            **summarize_ic(ic[backtest.decisions], stats.newey_west_lags, block,
                                           stats.bootstrap_samples, rng, stats.confidence)})
    by_period_df = pd.DataFrame(by_period).assign(period=month_label(panel.starts[backtest.decisions]))
    return by_period_df, pd.DataFrame(summary)


def returns_table(results: dict[str, StrategyResult], panel: Panel, k: int) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        for j, t in enumerate(result.decisions):
            picks = result.holdings[j] if result.holdings else []
            rows.append({"period": month_label([panel.starts[t]])[0], "k": k, "strategy": name,
                         "excess": result.excess[j], "usd": result.usd[j], "cost": result.cost[j],
                         "n_held": len(picks), "picks": "; ".join(panel.pools[i].label for i in picks)})
    return pd.DataFrame(rows)


def performance_table(results: dict[str, StrategyResult], k: int, periods_per_year: float,
                      tail: float) -> pd.DataFrame:
    rows = []
    reference = results.get("equal_weight")
    for name, result in results.items():
        for measure in ("excess", "usd"):
            values = getattr(result, measure)
            row = {"k": k, "strategy": name, "measure": measure, **performance(values, periods_per_year, tail)}
            if reference is not None and name != "equal_weight":
                both = np.isfinite(values) & np.isfinite(getattr(reference, measure))
                if both.any():
                    row["hit_rate_vs_equal_weight"] = float(np.mean(values[both] > getattr(reference, measure)[both]))
            row["mean_cost"] = float(np.nanmean(result.cost))
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(results: dict[str, StrategyResult], k: int, block: int, config: ExperimentConfig,
                     rng: np.random.Generator) -> pd.DataFrame:
    main = results[MAIN_STRATEGY]
    rows = []
    for name, result in results.items():
        if name == MAIN_STRATEGY:
            continue
        for measure in ("excess", "usd"):
            test = paired_difference(getattr(main, measure), getattr(result, measure), block,
                                     config.stats.bootstrap_samples, rng, config.stats.confidence)
            rows.append({"k": k, "baseline": name, "measure": measure, "mean_difference": test["mean"],
                         "lo": test["lo"], "hi": test["hi"], "p_value": test["p_value"], "n": test["n"]})
    return pd.DataFrame(rows)


def subperiod_table(results: dict[str, StrategyResult], panel: Panel, k: int, config: ExperimentConfig
                    ) -> pd.DataFrame:
    decisions = next(iter(results.values())).decisions
    starts = panel.starts[decisions]
    years = pd.to_datetime(starts, unit="s").year
    windows = {str(y): years == y for y in sorted(set(years))}
    for w in config.sample.event_windows:
        windows[w.name] = (starts >= w.start_ts) & (starts < w.end_ts)
    rows = []
    for window, mask in windows.items():
        if not mask.any():
            continue
        row = {"k": k, "window": window, "n_periods": int(mask.sum())}
        row.update({name: float(np.nanmean(result.excess[mask])) for name, result in results.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cumulative(results: dict[str, StrategyResult], panel: Panel, k: int, path_stem: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    dates = pd.to_datetime(panel.ends[next(iter(results.values())).decisions], unit="s")
    for name, result in results.items():
        if np.all(np.isnan(result.excess)) or name == "hodl_5050" or name.startswith("lvr_tail_"):
            continue
        wealth = np.cumprod(1.0 + np.nan_to_num(result.excess))
        ax.plot(dates, 100 * (wealth - 1.0), color=strategy_color(name), label=STRATEGY_LABELS.get(name, name),
                linewidth=2.0 if name == MAIN_STRATEGY else 1.2)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set(ylabel="Cumulative return vs HODL, after costs (%)", title=f"Top-{k} pools, monthly rebalancing")
    ax.legend(ncols=2, loc="upper left")
    save_figure(fig, path_stem)


def frontier_table(perf: pd.DataFrame, risk_aversions: tuple[float, ...]) -> pd.DataFrame:
    """Return vs downside of the mean-tail family (lambda = 0 is fee - sigma^2/8) and of the other strategies."""
    lam = {lvr_tail_name(x): x for x in risk_aversions} | {"fee_minus_lvr": 0.0}
    rows = perf[perf["measure"] == "excess"].copy()
    rows.insert(2, "risk_aversion", rows["strategy"].map(lam))
    return rows[["k", "strategy", "risk_aversion", "mean", "median", "var", "cvar", "worst", "max_drawdown",
                 "sortino", "share_positive"]]


def plot_frontier(frontier: pd.DataFrame, path_stem: Path) -> None:
    apply_style()
    ks = sorted(frontier["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.0), squeeze=False)
    for ax, k in zip(axes[0], ks):
        part = frontier[frontier["k"] == k]
        family = part[part["risk_aversion"].notna()].sort_values("risk_aversion")
        ax.plot(100 * family["cvar"], 100 * family["mean"], color=strategy_color("fee_minus_lvr"), marker="o",
                label="Fee − σ²/8 − λ·tail")
        for _, row in family.iterrows():
            ax.annotate(f"λ={row['risk_aversion']:g}", (100 * row["cvar"], 100 * row["mean"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        others = part[part["risk_aversion"].isna() & part["mean"].notna()
                      & ~part["strategy"].isin(["hodl_5050", "hodl_eth"])]
        for _, row in others.iterrows():
            ax.scatter(100 * row["cvar"], 100 * row["mean"], color=strategy_color(row["strategy"]), marker="s",
                       zorder=3, label=STRATEGY_LABELS.get(row["strategy"], row["strategy"]))
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set(xlabel="CVaR of monthly excess return (%)", title=f"Top-{k}")
    axes[0][0].set_ylabel("Mean monthly excess return (%)")
    axes[0][-1].legend(loc="lower right", fontsize=6)
    save_figure(fig, path_stem)


def run_selection_experiment(market: MarketData, config: ExperimentConfig, out_dir: Path,
                             panel: Panel | None = None) -> None:
    sel = config.selection
    if panel is None:
        panel = build_panel(market, *sample_periods(config.sample, market.data_end_ts, sel.horizon), sel.horizon)
    rules = selection_rules(config, sel.horizon, sel.calibration_months)
    backtest = prepare_backtest(panel, rules, score_functions(sel.tail_quantile, sel.risk_aversion))
    strategies = (*sel.strategies, *(lvr_tail_name(lam) for lam in sel.risk_aversion))
    schedule = cost_schedule(market, panel.starts, config.costs)
    eth_return = eth_returns(market, panel)
    block = sel.horizon.periods_for_months(config.stats.block_months)
    rng = np.random.default_rng(config.seed)

    by_period, ic_summary = ic_tables(backtest, config, block, rng)
    save_table(by_period, out_dir / "ic_by_period.csv")
    save_table(ic_summary, out_dir / "ic_summary.csv")
    universe = pd.DataFrame({"period": month_label(panel.starts[backtest.decisions]),
                             "n_eligible": backtest.eligible[:, backtest.decisions].sum(axis=0)})
    save_table(universe, out_dir / "eligible_pools_per_period.csv")

    returns, perf, comparisons, subperiods, random_rows = [], [], [], [], []
    for k in sel.top_k:
        results = run_strategies(backtest, strategies, k, schedule, eth_return)
        returns.append(returns_table(results, panel, k))
        perf.append(performance_table(results, k, sel.horizon.periods_per_year, sel.tail_quantile))
        comparisons.append(comparison_table(results, k, block, config, rng))
        subperiods.append(subperiod_table(results, panel, k, config))
        draws = random_portfolios(backtest, k, schedule, sel.random_draws, rng)
        main_mean = float(np.nanmean(results[MAIN_STRATEGY].excess))
        random_rows.append({"k": k, "net_tail_mean_excess": main_mean, "random_mean": float(np.mean(draws)),
                            "random_p05": float(np.quantile(draws, 0.05)), "random_p95": float(np.quantile(draws, 0.95)),
                            "share_random_at_least_net_tail": float(np.mean(draws >= main_mean))})
        plot_cumulative(results, panel, k, out_dir / f"cumulative_top{k}")

    save_table(pd.concat(returns), out_dir / "returns.csv")
    perf_df = pd.concat(perf)
    save_table(perf_df, out_dir / "performance.csv")
    frontier = frontier_table(perf_df, sel.risk_aversion)
    save_table(frontier, out_dir / "risk_return_frontier.csv")
    plot_frontier(frontier, out_dir / "risk_return_frontier")
    save_table(pd.concat(comparisons), out_dir / "net_tail_vs_baselines.csv")
    save_table(pd.concat(subperiods), out_dir / "subperiods.csv")
    save_table(pd.DataFrame(random_rows), out_dir / "random_portfolios.csv")


def run(ctx: Context, out_dir: Path) -> bool:
    horizon = ctx.config.selection.horizon
    panel = ctx.monthly_panel if horizon.months == 1 else ctx.panel(horizon)
    run_selection_experiment(ctx.market, ctx.config, out_dir, panel)
    return True
