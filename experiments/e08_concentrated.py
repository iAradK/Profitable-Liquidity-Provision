"""E8. Concentrated ranges: do full-range pool scores still pick good pools for concentrated positions?

Passive positions opened at each month start with range +-w around the entry price, or +-k sigma (sigma: the
pool's trailing monthly log-price volatility from the calibration window), never narrower than one tick spacing.
Fees accrue only in hours when the price is in range, from feeGrowthGlobal (exact fees per unit of liquidity),
so a position's liquidity per dollar follows from its range: no V0/TVL share and no fudge factor.

Scores: the mean-tail family S_lambda over [experiments.selection].risk_aversion (lambda = 0 is fee - sigma^2/8)
and Net90 for comparison. Per range and score: information coefficient of (a) the full-range score and (b) the
same score computed on that range's own history, against the realized concentrated net return. Per range: the
monthly rank correlation between full-range IL and concentrated IL, and the mean fee, IL and net return. Validation: the hourly fee approximation is compared with exact swap-by-swap fees from the
BigQuery Swap logs for the validation pools and months.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Horizon
from experiments.context import Context
from experiments.data import TICK_BASE, TICK_SPACING, MarketData
from experiments.outcomes import build_panel, position_amounts, position_outcome
from experiments.output import save_table
from experiments.plotting import SERIES_COLORS, STRATEGY_LABELS, apply_style, plt, save_figure
from experiments.scores import ScoreFn, lvr_tail_name, score_functions
from experiments.selection import (information_coefficients, past_periods, prepare_backtest, score_matrix,
                                   selection_rules, summarize_ic)
from experiments.stats import quiet, spearman

WORD = 64  # hex characters per 32-byte ABI word


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


def range_results(ctx: Context, specs: list[RangeSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = ctx.config
    full = ctx.monthly_panel
    rules = selection_rules(cfg, Horizon(months=1), cfg.selection.calibration_months)
    scores = range_scores(cfg.selection.tail_quantile, cfg.selection.risk_aversion)
    backtest = prepare_backtest(full, rules, {name: fn for name, (_, fn) in scores.items()})
    decisions, eligible = backtest.decisions, backtest.eligible
    stats = cfg.stats
    rng = ctx.rng(8)
    ic_args = (stats.newey_west_lags, stats.block_months, stats.bootstrap_samples, rng, stats.confidence)
    summary, cells = [], []
    for spec in specs:
        lower, upper = range_factors(spec, ctx)
        panel = build_panel(ctx.market, full.starts, full.ends, full.horizon, lower, upper)
        il_rank = np.array([spearman(full.il[eligible[:, t], t], panel.il[eligible[:, t], t]) for t in decisions])
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
    return pd.DataFrame(summary), pd.DataFrame(cells)


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

def _signed(word: str) -> int:
    value = int(word, 16)
    return value - (1 << 256) if value >= 1 << 255 else value


def decode_swaps(csv_path: Path, cache_path: Path) -> pd.DataFrame:
    """Swap logs -> amount0, amount1 (raw, signed, pool perspective), sqrt_price and liquidity after the swap."""
    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(cache_path)
    raw = pd.read_csv(csv_path, usecols=["block_timestamp", "block_number", "log_index", "pool", "data"])
    words = [[d[2 + i * WORD: 2 + (i + 1) * WORD] for i in range(4)] for d in raw["data"]]
    swaps = pd.DataFrame({
        "pool": raw["pool"].str.lower(),
        "ts": pd.to_datetime(raw["block_timestamp"], utc=True).dt.as_unit("s").astype("int64"),
        "block": raw["block_number"].astype(np.int64), "log_index": raw["log_index"].astype(np.int64),
        "amount0": [float(_signed(w[0])) for w in words], "amount1": [float(_signed(w[1])) for w in words],
        "sqrt_price": [int(w[2], 16) / 2**96 for w in words], "liquidity": [float(int(w[3], 16)) for w in words],
    }).sort_values(["block", "log_index"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    swaps.to_parquet(tmp, index=False)
    os.replace(tmp, cache_path)
    return swaps


def swap_fee_yield(swaps: pd.DataFrame, s0: float, s1: float, fee_rate: float, sa: float, sb: float) -> float:
    """Exact fees per unit of liquidity earned in range, valued at the closing price, relative to HODL."""
    active = swaps[swaps["liquidity"] > 0]
    after = active["sqrt_price"].to_numpy()
    before = np.concatenate([[s0], after[:-1]])
    credit = 0.5 * (((before >= sa) & (before <= sb)).astype(float) + ((after >= sa) & (after <= sb)))
    amount0, amount1, liquidity = (active[c].to_numpy() for c in ("amount0", "amount1", "liquidity"))
    growth0 = np.where(amount0 > 0, amount0 * fee_rate, 0.0) / liquidity
    growth1 = np.where(amount1 > 0, amount1 * fee_rate, 0.0) / liquidity
    x0, y0 = position_amounts(s0, sa, sb)
    return float(((credit @ growth0) * s1**2 + credit @ growth1) / (x0 * s1**2 + y0))


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
                hourly = position_outcome(pool, i0, i1, lower, upper, market.position_usd).fee_yield
                exact = swap_fee_yield(month_swaps, s0, s1, pool.meta.fee_rate, lower * s0, upper * s0)
                rows.append({"label": pool.meta.label, "period": pd.Timestamp(start, unit="s").strftime("%Y-%m"),
                             "range": spec.name, "n_swaps": len(month_swaps), "fee_yield_hourly": hourly,
                             "fee_yield_swaps": exact, "rel_error": hourly / exact - 1.0 if exact else math.nan})
    return pd.DataFrame(rows)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    specs = range_specs(cfg.concentrated.half_widths, cfg.concentrated.sigma_multiples)
    summary, cells = range_results(ctx, specs)
    save_table(summary, out_dir / "ic_by_range.csv")
    save_table(cells, out_dir / "concentrated_outcomes.csv")
    plot_ic(summary, cfg.concentrated.plot_risk_aversion, out_dir / "ic_vs_range")
    if cfg.data.swaps_file is None:
        return True
    swaps = decode_swaps(cfg.data.swaps_file, cfg.cache_dir / "swaps.parquet")
    validation = validation_table(ctx.market, swaps, ctx, specs)
    save_table(validation, out_dir / "swap_validation.csv")
    if not validation.empty:
        by_range = validation.groupby("range")["rel_error"].agg(
            median_abs_rel_error=lambda e: e.abs().median(), max_abs_rel_error=lambda e: e.abs().max(), n="count")
        save_table(by_range, out_dir / "swap_validation_summary.csv", index=True)
    return True
