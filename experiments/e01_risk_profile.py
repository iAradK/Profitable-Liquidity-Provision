"""E1. IL and net-yield profile per pool, with moving-block bootstrap confidence intervals.

Main view: overlapping 30-day windows started every day (the block bootstrap accounts for the overlap).
Robustness: non-overlapping calendar months.
The four IL metrics follow the paper (computed on the symmetric ratio R* = max(R, 1/R)); added are the P10 of
IL, the fee and net-yield distribution, and Net90 = median fee + P10(IL).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Horizon
from experiments.context import Context
from experiments.outcomes import Panel, build_panel, il_from_ratio, sample_periods
from experiments.output import cached_map, save_table
from experiments.stats import block_bootstrap_indices, percentile_interval

WINDOW_DAYS = 30


@dataclass(frozen=True)
class PoolSeries:
    window: str
    label: str
    category: str
    il: np.ndarray
    fee: np.ndarray
    ratio: np.ndarray
    tail_quantile: float
    block: int
    n_samples: int
    confidence: float
    seed: int


def profile_stats(il: np.ndarray, fee: np.ndarray, ratio: np.ndarray, tail_quantile: float) -> dict[str, float]:
    symmetric = np.maximum(ratio, 1.0 / ratio)
    mu, sigma = symmetric.mean(), symmetric.std()
    p05, p95 = np.quantile(symmetric, [0.05, 0.95])
    sigma_edges = [il_from_ratio(edge) for edge in (mu - sigma, mu + sigma) if edge > 0]
    net = il + fee
    il_tail = np.quantile(il, tail_quantile)
    return {
        "il_mean_metric": float(il_from_ratio(mu)),
        "il_median_metric": float(il_from_ratio(np.median(symmetric))),
        "il_bound_5_95": float(min(il_from_ratio(p05), il_from_ratio(p95))),
        "il_sigma_bound": float(min(sigma_edges)) if sigma_edges else np.nan,
        "il_tail": float(il_tail),
        "fee_median": float(np.median(fee)),
        "net_mean": float(net.mean()),
        "net_median": float(np.median(net)),
        "net_tail": float(np.quantile(net, tail_quantile)),
        "net90": float(np.median(fee) + il_tail),
    }


def profile_with_intervals(series: PoolSeries) -> dict:
    point = profile_stats(series.il, series.fee, series.ratio, series.tail_quantile)
    rng = np.random.default_rng(series.seed)
    indices = block_bootstrap_indices(len(series.il), series.block, series.n_samples, rng)
    samples = pd.DataFrame([profile_stats(series.il[i], series.fee[i], series.ratio[i], series.tail_quantile)
                            for i in indices])
    row = {"window": series.window, "label": series.label, "category": series.category, "n": len(series.il)}
    for name, value in point.items():
        lo, hi = percentile_interval(samples[name].to_numpy(), series.confidence)
        row.update({name: value, f"{name}_lo": lo, f"{name}_hi": hi})
    return row


def pool_series(panel: Panel, window: str, block: int, ctx: Context) -> list[tuple[str, PoolSeries]]:
    cfg = ctx.config
    items = []
    for p, meta in enumerate(panel.pools):
        valid = panel.valid[p]
        if valid.sum() < 2 * block:
            continue
        items.append((f"{window}_{meta.address}", PoolSeries(
            window, meta.label, meta.category, panel.il[p, valid], panel.fee[p, valid], panel.ratio[p, valid],
            cfg.selection.tail_quantile, block, cfg.stats.bootstrap_samples, cfg.stats.confidence, cfg.seed + p)))
    return items


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    starts, ends = sample_periods(cfg.sample, ctx.market.data_end_ts, Horizon(days=WINDOW_DAYS), Horizon(days=1))
    overlapping = build_panel(ctx.market, starts, ends, Horizon(days=WINDOW_DAYS))
    items = (pool_series(overlapping, f"rolling_{WINDOW_DAYS}d", cfg.stats.block_days, ctx)
             + pool_series(ctx.monthly_panel, "calendar_month", cfg.stats.block_months, ctx))
    results = cached_map(items, profile_with_intervals, out_dir / "pools", cfg.max_workers, cfg.force_refresh)
    save_table(pd.DataFrame([results[key] for key, _ in items]), out_dir / "risk_profile.csv")
    return True
