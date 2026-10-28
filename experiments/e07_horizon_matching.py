"""E7. Horizon matching: does setting the holding period from the half-life improve outcomes (paper Rule 4)?

At every monthly decision, each eligible pool's half-life is estimated from the trailing window of daily log
prices only (AR(1), as in E3), and its matched horizon is ceil(half-life in months), clipped to
[1, max_months]. The position is then held for the matched horizon, or for each fixed horizon, with entry and
exit costs paid once per holding. Outcomes are compared as monthly-equivalent net returns, both for the pools
Net90 selects and for all eligible pools. Only decisions where the longest horizon fits in the data are used,
so every policy is evaluated on the same decisions.
"""

import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import MONTH_DAYS, Horizon
from experiments.context import Context
from experiments.data import DAY
from experiments.outcomes import add_months, daily_log_prices, position_outcome
from experiments.output import save_table
from experiments.scores import net_tail
from experiments.selection import cost_schedule, prepare_backtest, selection_rules, top_k
from experiments.stats import fit_ar, half_life, paired_difference

MIN_DAYS_FOR_HALF_LIFE = 60


def matched_months(half_life_days: float, max_months: int) -> int:
    if not math.isfinite(half_life_days):
        return max_months
    return int(np.clip(math.ceil(half_life_days / MONTH_DAYS), 1, max_months))


def monthly_equivalent(total_return: float, months: int) -> float:
    return (1.0 + total_return) ** (1.0 / months) - 1.0 if total_return > -1.0 else -1.0


def decision_rows(ctx: Context) -> pd.DataFrame:
    cfg, hm = ctx.config, ctx.config.horizon_matching
    panel, market = ctx.monthly_panel, ctx.market
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    backtest = prepare_backtest(panel, rules, {"net_tail": partial(net_tail, tail_quantile=cfg.selection.tail_quantile)})
    schedule = cost_schedule(market, panel.starts, cfg.costs)
    longest = max(hm.max_months, *hm.fixed_months)
    end_limit = min(cfg.sample.end_ts, market.data_end_ts)
    capital_per_position = cfg.costs.capital_usd / hm.top_k

    rows = []
    for t in backtest.decisions:
        start = int(panel.starts[t])
        if add_months(start, longest) > end_limit:
            continue
        selected = set(top_k(backtest.scores["net_tail"][:, t], hm.top_k).tolist())
        for p in np.flatnonzero(backtest.eligible[:, t]):
            pool = market.pools[p]
            log_prices = daily_log_prices(pool.grid, start - hm.window_days * DAY, start)
            phi = fit_ar(log_prices, 1)["phi1"] if len(log_prices) >= MIN_DAYS_FOR_HALF_LIFE else math.nan
            hl_days = half_life(phi, cfg.persistence.phi_max)
            matched = matched_months(hl_days, hm.max_months)
            row = {"period": pd.Timestamp(start, unit="s").strftime("%Y-%m"), "label": pool.meta.label,
                   "selected_by_net_tail": p in selected, "half_life_days": hl_days, "matched_months": matched}
            i0 = int(pool.grid.index_at(start))
            for months in sorted(set(hm.fixed_months) | {matched}):
                i1 = int(pool.grid.index_at(add_months(start, months)))
                exit_t = min(t + months, len(schedule.exit_gas_usd) - 1)
                cost = ((schedule.entry_gas_usd[t] + schedule.exit_gas_usd[exit_t]) / capital_per_position
                        + cfg.costs.entry_swap_fraction * pool.meta.fee_rate)
                outcome = position_outcome(pool, i0, i1, position_usd=market.position_usd)
                value = monthly_equivalent(outcome.net - cost, months)
                if months in hm.fixed_months:
                    row[f"fixed_{months}m"] = value
                if months == matched:
                    row["matched"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def policy_summary(rows: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    cfg, hm = ctx.config, ctx.config.horizon_matching
    policies = ["matched"] + [f"fixed_{m}m" for m in hm.fixed_months]
    block = max(cfg.stats.block_months, hm.max_months)
    rng = ctx.rng(7)
    out = []
    for subset, part in (("selected_by_net_tail", rows[rows["selected_by_net_tail"]]), ("all_eligible", rows)):
        by_period = part.groupby("period")[policies].mean()
        for policy in policies:
            row = {"subset": subset, "policy": policy, "n_decisions": len(by_period),
                   "mean_monthly_net": by_period[policy].mean()}
            if policy != "matched":
                test = paired_difference(by_period["matched"].to_numpy(), by_period[policy].to_numpy(), block,
                                         cfg.stats.bootstrap_samples, rng, cfg.stats.confidence)
                row.update(matched_minus_policy=test["mean"], lo=test["lo"], hi=test["hi"], p_value=test["p_value"])
            out.append(row)
    return pd.DataFrame(out)


def run(ctx: Context, out_dir: Path) -> bool:
    rows = decision_rows(ctx)
    if rows.empty:
        raise RuntimeError("no decision leaves room for the longest horizon; check the sample dates")
    save_table(rows, out_dir / "decisions.csv")
    save_table(policy_summary(rows, ctx), out_dir / "horizon_policies.csv")
    save_table(rows["matched_months"].value_counts().sort_index().rename_axis("matched_months")
               .reset_index(name="n"), out_dir / "matched_horizon_distribution.csv")
    return True
