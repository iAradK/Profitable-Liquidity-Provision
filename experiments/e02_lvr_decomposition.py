"""E2. LVR decomposition: how much of each pool's IL does Gaussian (LVR) theory explain?

For a full-range position the expected loss over a period is about sigma_pair^2 / 8, where sigma_pair^2 is the
variance of the pair's log price; this approximates both LVR and the expected IL versus HODL.
  - predicted IL: -sigma_pair^2 / 8, with sigma_pair^2 the realized variance of the hourly pool log price
    (ex-post) or of the previous month (ex-ante);
  - decomposition: sigma_pair^2 = sigma_A^2 + sigma_B^2 - 2 rho sigma_A sigma_B, from daily USD token prices;
  - residual = realized IL - predicted IL is the tail component the theory misses.
Summaries use pool-months with TVL >= min_tvl_usd at the month start (all months are in the panel file).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.context import Context
from experiments.data import DAY, TokenPrices
from experiments.output import save_table
from experiments.plotting import (CATEGORY_ORDER, INK_SECONDARY, SERIES_COLORS, apply_style, category_color, plt,
                                  save_figure)


def daily_log_returns(prices: TokenPrices, start_ts: int, end_ts: int) -> np.ndarray:
    """Returns on the daily grid start_ts, start_ts + 1 day, ..., end_ts, so both tokens of a pool align even
    when one has no tokenDayData on some days (prices.at carries the last close forward)."""
    grid = np.arange(start_ts, end_ts + 1, DAY)
    return np.diff(np.log(prices.at(grid)))


def pool_month_table(ctx: Context) -> pd.DataFrame:
    panel = ctx.monthly_panel
    rows = []
    for p, meta in enumerate(panel.pools):
        price0, price1 = ctx.market.token_prices(meta.token0), ctx.market.token_prices(meta.token1)
        previous_variance = np.nan
        for t in np.flatnonzero(panel.valid[p]):
            r0 = daily_log_returns(price0, panel.starts[t], panel.ends[t])
            r1 = daily_log_returns(price1, panel.starts[t], panel.ends[t])
            var0, var1, cov = float(r0 @ r0), float(r1 @ r1), float(r0 @ r1)
            variance = panel.variance[p, t]
            rows.append({
                "label": meta.label, "category": meta.category,
                "month": pd.Timestamp(panel.starts[t], unit="s").strftime("%Y-%m"),
                "tvl_usd": panel.tvl[p, t], "il": panel.il[p, t], "fee": panel.fee[p, t], "net": panel.net[p, t],
                "variance_hourly": variance,
                "predicted_il": -variance / 8.0,
                "predicted_il_exante": -previous_variance / 8.0,
                "fee_minus_lvr": panel.fee[p, t] - variance / 8.0,
                "var_token0": var0, "var_token1": var1, "cov_tokens": cov,
                "rho": cov / np.sqrt(var0 * var1) if var0 > 0 and var1 > 0 else np.nan,
                "var_pair_daily": var0 + var1 - 2.0 * cov,
            })
            previous_variance = variance
    table = pd.DataFrame(rows)
    table["residual"] = table["il"] - table["predicted_il"]
    return table


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    resid = y[ok] - (slope * x[ok] + intercept)
    return {"slope": slope, "intercept": intercept, "r2": 1.0 - resid.var() / y[ok].var()}


def summarize(group: pd.DataFrame, tail_quantile: float) -> pd.Series:
    individual = group["var_token0"] + group["var_token1"]
    ex_post = fit_line(group["predicted_il"].to_numpy(), group["il"].to_numpy())
    ex_ante = fit_line(group["predicted_il_exante"].to_numpy(), group["il"].to_numpy())
    return pd.Series({
        "n_pool_months": len(group),
        "mean_il": group["il"].mean(),
        "mean_predicted_il": group["predicted_il"].mean(),
        "mean_residual": group["residual"].mean(),
        "residual_tail": group["residual"].quantile(tail_quantile),
        "il_tail": group["il"].quantile(tail_quantile),
        "predicted_tail": group["predicted_il"].quantile(tail_quantile),
        "slope_ex_post": ex_post["slope"], "r2_ex_post": ex_post["r2"],
        "slope_ex_ante": ex_ante["slope"], "r2_ex_ante": ex_ante["r2"],
        "mean_net": group["net"].mean(), "mean_fee_minus_lvr": group["fee_minus_lvr"].mean(),
        "mean_var_individual": individual.mean(),
        "mean_covariance_term": (-2.0 * group["cov_tokens"]).mean(),
        "covariance_offset_share": (2.0 * group["cov_tokens"]).mean() / individual.mean(),
        "mean_rho": group["rho"].mean(),
        "mean_var_pair_daily": group["var_pair_daily"].mean(),
        "mean_var_pair_hourly": group["variance_hourly"].mean(),
    })


def plot_predicted_vs_realized(table: pd.DataFrame, path_stem: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for category in CATEGORY_ORDER:
        part = table[table["category"] == category]
        ax.scatter(100 * part["predicted_il"], 100 * part["il"], s=9, color=category_color(category),
                   label=category, edgecolors="white", linewidths=0.3)
    low = 100 * min(table["predicted_il"].quantile(0.005), table["il"].quantile(0.005))
    ax.plot([low, 0], [low, 0], color=INK_SECONDARY, linewidth=1, linestyle="--", label="realized = predicted")
    ax.set(xlabel="Predicted IL, −σ²/8 (%)", ylabel="Realized monthly IL (%)",
           xlim=(low, 0.5), ylim=(low * 1.5, 0.5))
    ax.legend(loc="upper left")
    save_figure(fig, path_stem)


def plot_decomposition(summary: pd.DataFrame, path_stem: Path) -> None:
    apply_style()
    categories = [c for c in CATEGORY_ORDER if c in summary.index]
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x - width / 2 - 0.01, summary.loc[categories, "mean_var_individual"], width,
           color=SERIES_COLORS[0], label="σ²_A + σ²_B (no correlation)")
    ax.bar(x + width / 2 + 0.01, summary.loc[categories, "mean_var_pair_daily"], width,
           color=SERIES_COLORS[1], label="σ²_pair (with correlation)")
    ax.set(yscale="log", ylabel="Mean monthly variance of log price", xticks=x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend(loc="upper left")
    save_figure(fig, path_stem)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    table = pool_month_table(ctx)
    save_table(table, out_dir / "pool_months.csv")
    liquid = table[table["tvl_usd"] >= cfg.eligibility.min_tvl_usd]
    q = cfg.selection.tail_quantile
    by_category = liquid.groupby("category").apply(summarize, tail_quantile=q, include_groups=False)
    by_pool = liquid.groupby(["category", "label"]).apply(summarize, tail_quantile=q, include_groups=False)
    overall = summarize(liquid, q).rename("ALL").to_frame().T
    save_table(pd.concat([by_category, overall]), out_dir / "summary_by_category.csv", index=True)
    save_table(by_pool, out_dir / "summary_by_pool.csv", index=True)
    plot_predicted_vs_realized(liquid, out_dir / "predicted_vs_realized_il")
    plot_decomposition(by_category, out_dir / "variance_decomposition")
    return True
