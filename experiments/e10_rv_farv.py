"""E10. Comparison with the fee-adjusted metrics RV and FARV (paper refs. [22], [27]).

Interpretation used here (verify against Roman et al., ref. [22], before reporting):
  RV   = V_LP / V_HODL            = 1 + IL         (relative value of the LP position)
  FARV = (V_LP + fees) / V_HODL   = 1 + IL + fees  (fee-adjusted relative value)
Each is ranked by its historical mean over the calibration window, so FARV ranks pools like the mean net
return. If ref. [22] defines them differently, change `relative_value` / `fee_adjusted_relative_value` in
experiments/scores.py; this experiment needs no other change.

Outputs: monthly rank agreement between Net90 and RV/FARV, the information coefficient of each, their top-k
backtests, and the months where the top pick differs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Horizon
from experiments.context import Context
from experiments.output import save_table
from experiments.scores import score_functions
from experiments.selection import (cost_schedule, eth_returns, information_coefficients, prepare_backtest,
                                   run_strategies, selection_rules, summarize_ic, top_k)
from experiments.stats import paired_difference, performance, spearman

COMPARED = ("rv", "farv")
SCORES = ("net_tail", *COMPARED, "fee_minus_lvr")


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    panel = ctx.monthly_panel
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    all_scores = score_functions(cfg.selection.tail_quantile)
    backtest = prepare_backtest(panel, rules, {name: all_scores[name] for name in SCORES})
    decisions, scores, eligible = backtest.decisions, backtest.scores, backtest.eligible
    stats = cfg.stats
    rng = ctx.rng(10)

    agreement, disagreements = [], []
    for t in decisions:
        mask = eligible[:, t]
        if mask.sum() < 2:
            continue
        period = pd.Timestamp(int(panel.starts[t]), unit="s").strftime("%Y-%m")
        top3 = {name: set(top_k(scores[name][:, t], 3).tolist()) for name in SCORES}
        top1 = {name: int(top_k(scores[name][:, t], 1)[0]) for name in SCORES}
        row = {"period": period, "n_eligible": int(mask.sum())}
        for name in COMPARED:
            row[f"spearman_net_tail_{name}"] = spearman(scores["net_tail"][mask, t], scores[name][mask, t])
            row[f"same_top1_{name}"] = top1["net_tail"] == top1[name]
            row[f"top3_overlap_{name}"] = len(top3["net_tail"] & top3[name]) / 3
        agreement.append(row)
        if any(top1["net_tail"] != top1[name] for name in COMPARED):
            disagreements.append({"period": period, **{
                f"{name}_pick": panel.pools[top1[name]].label for name in ("net_tail", *COMPARED)}, **{
                f"{name}_pick_net": panel.net[top1[name], t] for name in ("net_tail", *COMPARED)}})
    agreement_df = pd.DataFrame(agreement)
    save_table(agreement_df, out_dir / "rank_agreement_by_period.csv")
    save_table(agreement_df.drop(columns="period").mean().rename("mean").to_frame(), out_dir / "rank_agreement_summary.csv",
               index=True)
    save_table(pd.DataFrame(disagreements), out_dir / "top_pick_disagreements.csv")

    ic_rows = []
    for name in SCORES:
        ic = information_coefficients(scores[name], panel.net, eligible, stats.min_pools_for_ic)[decisions]
        ic_rows.append({"score": name, **summarize_ic(ic, stats.newey_west_lags, stats.block_months,
                                                      stats.bootstrap_samples, rng, stats.confidence)})
    save_table(pd.DataFrame(ic_rows), out_dir / "ic_summary.csv")

    schedule = cost_schedule(ctx.market, panel.starts, cfg.costs)
    perf_rows, test_rows = [], []
    for k in cfg.selection.top_k:
        results = run_strategies(backtest, SCORES, k, schedule, eth_returns(ctx.market, panel))
        for name, result in results.items():
            perf_rows.append({"k": k, "strategy": name, **performance(result.excess, panel.horizon.periods_per_year)})
        for name in COMPARED:
            test = paired_difference(results["net_tail"].excess, results[name].excess, stats.block_months,
                                     stats.bootstrap_samples, rng, stats.confidence)
            test_rows.append({"k": k, "baseline": name, **test})
    save_table(pd.DataFrame(perf_rows), out_dir / "performance.csv")
    save_table(pd.DataFrame(test_rows), out_dir / "net_tail_vs_rv_farv.csv")
    return True
