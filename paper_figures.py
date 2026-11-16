"""
Figures of the ICBC paper that are drawn from saved experiment results (no experiment needs to rerun):

  fig_dollar.pdf          E5: cumulative gain over HODL after costs (USD, summed) of the top-k pools of each strategy
                          for the configured capital, month by month
  fig_calibration.pdf     E4: share of months worse than the 10th-percentile IL forecast (12-month window) with a
                          Wilson interval, for all pools and per category, model tail against the empirical percentile
  fig_fees_vs_il.pdf      E2: per pool, mean monthly fee yield against mean monthly IL (log scales), by category,
                          with the break-even line fee = IL
  fig_concentrated.pdf    E8: top-k mean and CVaR of the monthly excess return per position range, for the fee-only
                          ranking, the lowest variance and S_lambda
  fig_picks.pdf           E8: share of the held top-k positions per pool category, for each score

Called by make_paper_assets.py, which copies the PDFs into the paper's generated/ folder.
"""

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from experiments.plotting import (CATEGORY_ORDER, GRID, INK_SECONDARY, SERIES_COLORS, apply_style, category_color,
                                  plt, save_figure, strategy_color)
from experiments.stats import wilson_interval

CATEGORY_LABELS = {"stable-volatile": "Stable–volatile", "stable-stable": "Stablecoin pair",
                   "correlated": "Correlated", "governance": "Governance", "high-vol": "High-vol./declining"}
COLUMN_WIDTH = 3.45  # inches, one IEEE column
MIN_LOSS_PCT = 1e-6  # log-scale floor for pools whose IL rounds to zero (stablecoin pairs)

# S_lambda members shown in the concentrated-range figures: one hue, darker for a larger lambda (magnitude)
LAMBDA_SHADES = {0.0: "#f2a37e", 0.25: SERIES_COLORS[1], 1.0: "#9c3d14"}
LAMBDA_MARKERS = {0.0: "^", 0.25: "D", 1.0: "v"}


def lambda_name(lam: float) -> str:
    return "fee_minus_lvr" if lam == 0 else f"lvr_tail_{lam:g}"


def lambda_label(lam: float) -> str:
    return f"$S_{{{lam:g}}}$"


def range_label(name: str) -> str:
    return name.replace("σ", r"$\sigma$")


def fees_vs_il(results: Path, out: Path) -> None:
    pools = pd.read_csv(results / "E2" / "summary_by_pool.csv")
    loss = np.clip(-100 * pools["mean_il"], MIN_LOSS_PCT, None)
    fee = 100 * (pools["mean_net"] - pools["mean_il"])
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
    span = np.array([MIN_LOSS_PCT / 2, 4.0])
    ax.plot(span, span, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(2.5e-3, 6.5e-3, "fee = IL", color=INK_SECONDARY, fontsize=7, rotation=45, rotation_mode="anchor",
            transform_rotates_text=True, ha="center", va="bottom")
    ax.fill_between(span, span, 10.0, color=GRID, alpha=0.35, linewidth=0, zorder=0)
    ax.text(2e-5, 0.6, "fee > IL\n(beats HODL)", color=INK_SECONDARY, fontsize=7, va="center")
    for category in CATEGORY_ORDER:
        rows = pools["category"] == category
        ax.scatter(loss[rows], fee[rows], s=26, color=category_color(category), edgecolor="white", linewidth=0.8,
                   label=CATEGORY_LABELS[category], zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(MIN_LOSS_PCT / 2, 3.0), ylim=(1e-3, 3.0),
           xlabel="Mean monthly IL (%, log scale)", ylabel="Mean monthly fee yield (%)")
    ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
    save_figure(fig, out / "fig_fees_vs_il")


def concentrated(results: Path, k: int, out: Path) -> None:
    top = pd.read_csv(results / "E8" / "top_k_by_range.csv").query("k == @k")
    ranges = list(dict.fromkeys(top["range"]))
    x = np.arange(len(ranges))
    series = [("median_fee", "Highest median fee", strategy_color("median_fee"), "o"),
              ("low_variance", "Lowest $\\sigma^2$", strategy_color("low_variance"), "s")]
    series += [(lambda_name(lam), lambda_label(lam), color, LAMBDA_MARKERS[lam]) for lam, color in LAMBDA_SHADES.items()]
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.35), sharex=True)
    n_fixed = sum(not r.endswith("σ") for r in ranges)
    for ax, column, label in ((axes[0], "mean_excess", "Mean (%)"), (axes[1], "cvar", "CVaR (%)")):
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.axvline(n_fixed - 0.5, color=GRID, linewidth=1.2)
        for name, legend, color, marker in series:
            values = top.set_index("strategy").loc[name].set_index("range").loc[ranges, column]
            ax.plot(x, 100 * values, color=color, marker=marker, markersize=4.5, linewidth=1.4, label=legend)
        ax.set_ylabel(label)
    axes[1].set_xticks(x, [range_label(r) for r in ranges])
    axes[1].set_xlabel("Position range (fixed width | multiple of the trailing volatility)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=7, columnspacing=0.8,
                   handletextpad=0.3)
    fig.align_ylabels(axes)
    save_figure(fig, out / "fig_concentrated")


def picks(results: Path, k: int, risk_aversion: tuple[float, ...], out: Path) -> None:
    shares = pd.read_csv(results / "E8" / "picks_by_category.csv").query("k == @k")
    rows = [("median_fee", "Highest median fee")] + [(lambda_name(lam), lambda_label(lam))
                                                    for lam in (0.0, *risk_aversion) if lam <= 1]
    rows.append(("low_variance", "Lowest $\\sigma^2$"))
    table = shares.pivot(index="strategy", columns="category", values="share").loc[[r for r, _ in rows]]
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.4))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for category in CATEGORY_ORDER:
        values = 100 * table.get(category, pd.Series(0.0, index=table.index)).fillna(0).to_numpy()
        ax.barh(y, values, left=left, height=0.7, color=category_color(category), edgecolor="white", linewidth=1.0,
                label=CATEGORY_LABELS[category])
        left += values
    ax.set_yticks(y, [label for _, label in rows])
    ax.set(xlim=(0, 100), xlabel=f"Share of the top-{k} positions (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=3, fontsize=6.5, columnspacing=0.8,
              handletextpad=0.3, handlelength=1.0)
    save_figure(fig, out / "fig_picks")


# Tail forecasts in the calibration figure: E4 metric -> (legend, color, marker). Ours wears the S_lambda hue, the
# empirical percentile the Net90 hue (Net90 is built on it).
CALIBRATION_SERIES = {"il_tail_model": ("Model tail (ours)", SERIES_COLORS[1], "D"),
                      "il_tail": ("Empirical $P_{10}$", strategy_color("net_tail"), "o")}


def calibration_rates(e4: pd.DataFrame, scheme: str, confidence: float) -> pd.DataFrame:
    """Exceedance rate and Wilson interval per (metric, row), the rows being all pools and each category."""
    pools = e4[(e4["label"] != "pooled") & (e4["scheme"] == scheme)].drop_duplicates(["label", "metric"])
    rows = []
    for metric in CALIBRATION_SERIES:
        sub = pools[pools["metric"] == metric]
        groups = [("All pools", sub)] + [(CATEGORY_LABELS[c], sub[sub["category"] == c]) for c in CATEGORY_ORDER]
        for name, g in groups:
            n, k = int(g["n"].sum()), int(g["exceedances"].sum())
            lo, hi = wilson_interval(k, n, confidence)
            rows.append({"metric": metric, "row": name, "rate": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def calibration(results: Path, tail_quantile: float, confidence: float, out: Path) -> None:
    e4 = pd.read_csv(results / "E4" / "exceedance_tests.csv")
    rolling = next(s for s in e4["scheme"].unique() if s.startswith("rolling"))
    rates = calibration_rates(e4, rolling, confidence)
    names = list(dict.fromkeys(rates["row"]))
    y = np.arange(len(names))[::-1].astype(float)
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.5))
    ax.axvline(100 * tail_quantile, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
    for offset, (metric, (legend, color, marker)) in zip((0.16, -0.16), CALIBRATION_SERIES.items()):
        r = rates[rates["metric"] == metric].set_index("row").loc[names]
        x = 100 * r["rate"].to_numpy()
        err = 100 * np.vstack([r["rate"] - r["lo"], r["hi"] - r["rate"]])
        ax.errorbar(x, y + offset, xerr=err, fmt=marker, color=color, markersize=4.5, elinewidth=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.6, label=legend, zorder=3)
    ax.axhline(len(names) - 1.5, color=GRID, linewidth=1.0)
    ax.set_yticks(y, names)
    ax.set(xlim=(0, None), xlabel="Months worse than the forecast (%), 95% CI", ylim=(-0.6, len(names) - 0.2))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncols=2, fontsize=6.5, handletextpad=0.2,
              columnspacing=1.0)
    save_figure(fig, out / "fig_calibration")


# Strategies in the dollar figure: E5 name -> (legend, color, line width). S_lambda members wear the lambda shades.
DOLLAR_SERIES = {
    "fee_minus_lvr": ("$S_0$", LAMBDA_SHADES[0.0], 1.4),
    "lvr_tail_0.25": ("$S_{0.25}$", LAMBDA_SHADES[0.25], 2.0),
    "median_fee": ("Highest median fee", strategy_color("median_fee"), 1.4),
    "net_tail": ("Net90", strategy_color("net_tail"), 1.4),
    "low_variance": ("Lowest $\\sigma^2$", strategy_color("low_variance"), 1.4),
    "equal_weight": ("Equal weight", strategy_color("equal_weight"), 1.4),
}


def cumulative_usd(results: Path, k: int, capital_usd: float) -> pd.DataFrame:
    """Cumulative gain over HODL after costs (USD, summed, not compounded) of the top-k pools of each strategy."""
    returns = pd.read_csv(results / "E5" / "returns.csv", parse_dates=["period"]).query("k == @k")
    monthly = returns.pivot(index="period", columns="strategy", values="excess")
    return capital_usd * monthly.cumsum()


def dollar_path(results: Path, k: int, capital_usd: float, out: Path) -> None:
    paths = cumulative_usd(results, k, capital_usd)
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.8))
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    for name, (legend, color, width) in DOLLAR_SERIES.items():
        ax.plot(paths.index, paths[name] / 1000, color=color, linewidth=width, label=legend)
    ax.set_ylabel("Gain over HODL (K USD)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=6.5, columnspacing=0.8,
              handletextpad=0.3, handlelength=1.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, out / "fig_dollar")


def make_figures(results: Path, k: int, risk_aversion: tuple[float, ...], tail_quantile: float,
                 confidence: float, capital_usd: float) -> Path:
    """Draws every figure into <results>/paper and returns that folder."""
    out = results / "paper"
    dollar_path(results, k, capital_usd, out)
    calibration(results, tail_quantile, confidence, out)
    fees_vs_il(results, out)
    concentrated(results, k, out)
    picks(results, k, risk_aversion, out)
    return out
