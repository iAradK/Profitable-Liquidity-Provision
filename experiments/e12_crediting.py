"""E12. Fee crediting: how much do the results depend on how the added liquidity changes trading (Section III-A)?

A hypothetical position adds liquidity L to the pool. Two crediting rules bound the effect:
  fixed_volume  L * dG * L_act / (L_act + L)  swaps of a fixed size pay the same fee, now shared with L (main)
  proportional  L * dG * min(1, L_act / L)    arbitrage volume, and hence its fees, grows with the liquidity, up to
                                              the fees traders paid
L_act is the fee-weighted in-range liquidity of each day (experiments.outcomes.sharing_liquidity). The rules
coincide for L << L_act. Each cell of [experiments.fees] sensitivity_rules x sensitivity_position_usd
rebuilds the monthly full-range panel, with the position (and the capital that gas is charged on) of that size,
and reruns the E5 backtest.

Outputs:
  crediting_performance.csv  per cell, k and strategy: mean monthly excess return after costs, its block-bootstrap
                             interval, the mean without [experiments.fees].exclude_periods, the mean with the
                             thin-liquidity pool-months ineligible for every strategy, CVaR
  crediting_ic.csv           per cell and score: information coefficient against the realized net return
  credited_share.csv         per cell: the pool-months with the largest share of the pool's fees credited to the
                             position (proportional crediting can exceed 100%)
  replay.csv                 months covered by [experiments.data].replay_swaps_files, full-range position: fee
                             yield from the hourly accumulators vs swap by swap, without a position effect (raw)
                             and under each rule, and the credited shares. The thin-liquidity months among them
                             are computed swap by swap in every experiment (experiments.swaps)
"""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import CREDITING_RULES, ExperimentConfig, FeeCrediting, Horizon
from experiments.context import Context
from experiments.data import MarketData, PoolData
from experiments.outcomes import (Panel, build_panel, crediting_factor, position_amounts, position_liquidity,
                                  position_outcome, sample_periods, sharing_liquidity)
from experiments.output import save_table
from experiments.scores import lvr_tail_name, score_functions
from experiments.selection import (cost_schedule, information_coefficients, prepare_backtest, run_strategies,
                                   selection_rules, summarize_ic, without_thin_months)
from experiments.stats import bootstrap_mean, performance, quiet
from experiments.swaps import covers, decode_swaps, implied_over_paid, swap_fee_yield

log = logging.getLogger(__name__)

TOP_SHARES = 10  # pool-months with the largest credited share kept per cell


def month(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s").strftime("%Y-%m")


def cell_strategies(config: ExperimentConfig) -> tuple[str, ...]:
    return ("fee_minus_lvr", *(lvr_tail_name(lam) for lam in config.selection.risk_aversion),
            "median_fee", "fee_low_variance", "net_tail", "equal_weight")


def cell_results(market: MarketData, panel: Panel, config: ExperimentConfig, crediting: FeeCrediting,
                 rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    sel, stats = config.selection, config.stats
    rules = selection_rules(config, Horizon(months=1), sel.calibration_months)
    backtest = prepare_backtest(panel, rules, score_functions(sel.tail_quantile, sel.risk_aversion))
    costs = replace(config.costs, capital_usd=crediting.position_usd)
    schedule = cost_schedule(market, panel.starts, costs)
    no_eth = np.full(panel.n_periods, np.nan)
    excluded = np.isin([month(panel.starts[t]) for t in backtest.decisions], config.fees.exclude_periods)
    cell = {"rule": crediting.rule, "position_usd": crediting.position_usd}
    robust = without_thin_months(backtest, market.thin_months)
    perf_rows, ic_rows = [], []
    for k in sel.top_k:
        no_thin = run_strategies(robust, cell_strategies(config), k, schedule, no_eth)
        for name, result in run_strategies(backtest, cell_strategies(config), k, schedule, no_eth).items():
            interval = bootstrap_mean(result.excess, stats.block_months, stats.bootstrap_samples, rng,
                                      stats.confidence)
            perf_rows.append({**cell, "k": k, "strategy": name, "mean": interval["mean"], "mean_lo": interval["lo"],
                              "mean_hi": interval["hi"],
                              "mean_excluding": quiet(np.nanmean, result.excess[~excluded]),
                              "mean_no_thin": quiet(np.nanmean, no_thin[name].excess),
                              "cvar": performance(result.excess, sel.horizon.periods_per_year,
                                                  sel.tail_quantile).get("cvar", np.nan)})
    for name, scores in backtest.scores.items():
        ic = information_coefficients(scores, panel.net, backtest.eligible, stats.min_pools_for_ic)
        ic_rows.append({**cell, "score": name,
                        **summarize_ic(ic[backtest.decisions], stats.newey_west_lags, stats.block_months,
                                       stats.bootstrap_samples, rng, stats.confidence)})
    return perf_rows, ic_rows


def credited_share(pool: PoolData, i0: int, i1: int, crediting: FeeCrediting) -> float:
    """Share of all fees paid in the pool over grid states i0..i1 that a full-range position is credited with.

    The fees paid in an hour are its fee growth times L_act (sharing_liquidity), valued at the hourly USD prices.
    """
    x0, y0 = position_amounts(pool.grid.sqrt_price[i0], 0.0, np.inf)
    position = position_liquidity(pool, i0, x0, y0, crediting)
    if position == 0.0:
        return np.nan
    grid = pool.grid
    d0, d1 = np.diff(grid.fee_growth0[i0:i1 + 1]), np.diff(grid.fee_growth1[i0:i1 + 1])
    usd = np.nan_to_num(d0 * pool.usd0[i0 + 1:i1 + 1] + d1 * pool.usd1[i0 + 1:i1 + 1])
    paid = float(np.sum(usd * sharing_liquidity(pool, i0, i1)))
    credited = float(np.sum(usd * position * crediting_factor(pool, i0, i1, x0, y0, crediting)))
    return credited / paid if paid > 0 else np.nan


def share_rows(market: MarketData, panel: Panel, crediting: FeeCrediting) -> list[dict]:
    rows = []
    for pool in market.pools:
        i0s, i1s = pool.grid.index_at(panel.starts), pool.grid.index_at(panel.ends)
        for t in np.flatnonzero(i0s >= 0):
            share = credited_share(pool, int(i0s[t]), int(i1s[t]), crediting)
            if np.isfinite(share):
                rows.append({"rule": crediting.rule, "position_usd": crediting.position_usd,
                             "label": pool.meta.label, "period": month(panel.starts[t]), "credited_share": share})
    top = sorted(rows, key=lambda r: -r["credited_share"])[:TOP_SHARES]
    return top


def replay_table(market: MarketData, swaps: pd.DataFrame, panel: Panel, position_usd: float) -> pd.DataFrame:
    """Full-range fee yield of each month covered by the swap exports: from the hourly accumulators (without the
    replay) vs swap by swap, with no position effect (raw) and under each rule for a position of position_usd."""
    rows = []
    for pool in market.pools:
        pool_swaps = swaps[swaps["pool"] == pool.meta.address]
        if pool_swaps.empty:
            continue
        hourly_pool = replace(pool, replay=None)
        grid = pool.grid
        for t in range(panel.n_periods):
            start, end = int(panel.starts[t]), int(panel.ends[t])
            i0, i1 = int(grid.index_at(start)), int(grid.index_at(end))
            if i0 < 0 or not covers(pool_swaps, start, end):
                continue
            month_swaps = pool_swaps[(pool_swaps["ts"] >= start) & (pool_swaps["ts"] < end)]
            s0, s1 = grid.sqrt_price[i0], grid.sqrt_price[i1]
            x0, y0 = position_amounts(s0, 0.0, np.inf)
            row = {"label": pool.meta.label, "period": month(start), "n_swaps": len(month_swaps),
                   "implied_over_paid": implied_over_paid(pool, i0, i1),
                   "raw_hourly": position_outcome(hourly_pool, i0, i1).fee_yield,
                   "raw_swaps": swap_fee_yield(month_swaps, s0, s1, grid.liquidity[i0], pool.meta.fee_rate, 0.0, np.inf)}
            for rule in CREDITING_RULES:
                crediting = FeeCrediting(rule, position_usd)
                hourly = position_outcome(hourly_pool, i0, i1, crediting=crediting).fee_yield
                exact = swap_fee_yield(month_swaps, s0, s1, grid.liquidity[i0], pool.meta.fee_rate, 0.0, np.inf,
                                       position_liquidity(pool, i0, x0, y0, crediting), rule)
                row |= {f"{rule}_hourly": hourly, f"{rule}_swaps": exact,
                        f"{rule}_rel_error": hourly / exact - 1.0 if exact else np.nan,
                        f"{rule}_credited_share": credited_share(hourly_pool, i0, i1, crediting)}
            rows.append(row)
    return pd.DataFrame(rows)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    starts, ends = sample_periods(cfg.sample, ctx.market.data_end_ts, Horizon(months=1))
    rng = ctx.rng(12)
    perf, ic, shares = [], [], []
    for rule in cfg.fees.sensitivity_rules:
        for size in cfg.fees.sensitivity_position_usd:
            crediting = FeeCrediting(rule, size)
            log.info("E12: %s crediting, position %.0f USD", rule, size)
            market = replace(ctx.market, crediting=crediting)
            panel = build_panel(market, starts, ends, Horizon(months=1))
            cell_perf, cell_ic = cell_results(market, panel, cfg, crediting, rng)
            perf += cell_perf
            ic += cell_ic
            shares += share_rows(market, panel, crediting)
    save_table(pd.DataFrame(perf), out_dir / "crediting_performance.csv")
    save_table(pd.DataFrame(ic), out_dir / "crediting_ic.csv")
    save_table(pd.DataFrame(shares), out_dir / "credited_share.csv")
    if cfg.data.replay_swaps_files:
        swaps = decode_swaps(cfg.data.replay_swaps_files, cfg.cache_dir / "replay_swaps.parquet")
        replay = replay_table(ctx.market, swaps, ctx.monthly_panel, cfg.fees.main.position_usd)
        save_table(replay, out_dir / "replay.csv")
    return True
