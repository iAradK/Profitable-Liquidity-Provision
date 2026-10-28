"""E6. Sensitivity grid: does the E5 result depend on the tail level, horizon, calibration window or k?

Every combination of [experiments.sensitivity] is one E5-style backtest of Net-q (median fee + q-quantile IL)
against fee - sigma^2/8 and equal weight. Per cell: mean excess return of each, the paired block-bootstrap
difference Net-q minus fee - sigma^2/8, and the information coefficient of Net-q. Differences are also given
per month so cells with different horizons are comparable.
"""

import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import MONTH_DAYS, Horizon
from experiments.context import Context
from experiments.outcomes import Panel
from experiments.output import cached_map, save_table
from experiments.plotting import DIVERGING, INK, apply_style, plt, save_figure
from experiments.scores import fee_minus_lvr, net_tail
from experiments.selection import (CostSchedule, SelectionRules, cost_schedule, information_coefficients,
                                   prepare_backtest, run_strategies, selection_rules)
from experiments.stats import newey_west, paired_difference

STRATEGIES = ("net_tail", "fee_minus_lvr", "equal_weight")


@dataclass(frozen=True)
class Cell:
    horizon: Horizon
    tail_quantile: float
    calibration_months: int
    k: int
    panel: Panel
    rules: SelectionRules
    schedule: CostSchedule
    block: int
    n_samples: int
    confidence: float
    nw_lags: int
    min_pools_for_ic: int
    seed: int


def evaluate_cell(cell: Cell) -> dict:
    scores = {"net_tail": partial(net_tail, tail_quantile=cell.tail_quantile), "fee_minus_lvr": fee_minus_lvr}
    backtest = prepare_backtest(cell.panel, cell.rules, scores)
    no_eth = np.full(cell.panel.n_periods, np.nan)
    results = run_strategies(backtest, STRATEGIES, cell.k, cell.schedule, no_eth)
    rng = np.random.default_rng(cell.seed)
    diff = paired_difference(results["net_tail"].excess, results["fee_minus_lvr"].excess, cell.block,
                             cell.n_samples, rng, cell.confidence)
    ic = information_coefficients(backtest.scores["net_tail"], cell.panel.net, backtest.eligible,
                                  cell.min_pools_for_ic)[backtest.decisions]
    ic_mean, ic_t = newey_west(ic, cell.nw_lags)
    per_month = MONTH_DAYS / cell.horizon.approx_days
    return {
        "horizon": str(cell.horizon), "tail_quantile": cell.tail_quantile,
        "calibration_months": cell.calibration_months, "k": cell.k, "n_decisions": len(backtest.decisions),
        **{f"mean_excess_{name}": float(np.nanmean(results[name].excess)) for name in STRATEGIES},
        "diff_vs_fee_minus_lvr": diff["mean"], "diff_lo": diff["lo"], "diff_hi": diff["hi"], "diff_p": diff["p_value"],
        "diff_per_month": diff["mean"] * per_month,
        "ic_mean": ic_mean, "ic_nw_t": ic_t,
    }


def plot_heatmaps(grid: pd.DataFrame, k: int, limit: float, path_stem: Path) -> None:
    """Net-q minus fee - sigma^2/8 (% per month) per tail level x calibration window, one panel per horizon."""
    apply_style()
    part_k = grid[grid["k"] == k]
    horizons = list(dict.fromkeys(part_k["horizon"]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.0 * len(horizons), 2.0), squeeze=False,
                             constrained_layout=True)
    for ax, horizon in zip(axes[0], horizons):
        part = part_k[part_k["horizon"] == horizon]
        table = part.pivot(index="tail_quantile", columns="calibration_months", values="diff_per_month") * 100
        image = ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=INK, fontsize=7)
        ax.set_xticks(range(table.shape[1]), [("exp." if m == 0 else f"{m}m") for m in table.columns])
        ax.set_yticks(range(table.shape[0]), [f"P{round(q * 100)}" for q in table.index])
        ax.grid(False)
        ax.set_title(f"Horizon {horizon}", fontsize=8)
        ax.set_xlabel("Calibration window", fontsize=7)
    axes[0][0].set_ylabel("Tail level")
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="% per month")
    save_figure(fig, path_stem)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg, sens = ctx.config, ctx.config.sensitivity
    items = []
    for horizon in sens.horizons:
        panel = ctx.monthly_panel if horizon == Horizon(months=1) else ctx.panel(horizon)
        schedule = cost_schedule(ctx.market, panel.starts, cfg.costs)
        for q, months, k in itertools.product(sens.tail_quantiles, sens.calibration_months, sens.top_k):
            key = f"{horizon}_q{q:g}_cal{months}_k{k}"
            items.append((key, Cell(horizon, q, months, k, panel, selection_rules(cfg, horizon, months), schedule,
                                    ctx.block_periods(horizon), cfg.stats.bootstrap_samples, cfg.stats.confidence,
                                    cfg.stats.newey_west_lags, cfg.stats.min_pools_for_ic, cfg.seed)))
    results = cached_map(items, evaluate_cell, out_dir / "cells", cfg.max_workers, cfg.force_refresh)
    grid = pd.DataFrame([results[key] for key, _ in items])
    save_table(grid, out_dir / "sensitivity_grid.csv")
    limit = float(np.nanmax(np.abs(grid["diff_per_month"]))) * 100  # one colour scale for every k
    for k in sorted(grid["k"].unique()):
        plot_heatmaps(grid, int(k), limit if np.isfinite(limit) and limit > 0 else 1.0,
                      out_dir / f"sensitivity_heatmap_k{k}")
    return True
