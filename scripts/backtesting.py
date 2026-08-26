from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION (match your other scripts)
# ==========================================
BASE_PATH = Path("./pool_data/updated")  # not used here, but kept for consistency
OUT_DIR = Path("backtesting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input produced by your monthly IL script
IN_PATH = Path("monthly_il") / "monthly_il.dat"

# Keep the same pool ordering you use in figures/legends
DEFAULT_SHORT_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]

# Backtest settings
MIN_TRAIN_MONTHS = 12  # expanding window starts OOS at index 12 (month 13)

# Metric names + targets (your requirement)
METRICS = ["Bound (10%)", "Sigma Bound", "Median", "Mean"]
TARGETS: Dict[str, float] = {
    "Bound (10%)": 0.10,
    "Sigma Bound": 0.1587,  # ~Phi(-1) under Normal
    "Median": 0.50,
    "Mean": 0.50,
}

# ==========================================
# 2. LOAD IL DATA (wide .dat: month, il_45, il_cb, ...)
# ==========================================
if not IN_PATH.exists():
    raise FileNotFoundError(f"Could not find input file: {IN_PATH}")

df_wide = pd.read_csv(IN_PATH, sep=r"\s+", engine="python")
if "month" not in df_wide.columns:
    raise ValueError(f"Input file must contain a 'month' column. Got: {df_wide.columns.tolist()}")

df_wide = df_wide.sort_values("month").reset_index(drop=True)

# Extract only pools we want and that exist in file
pool_cols = {}
for short in DEFAULT_SHORT_ORDER:
    col = f"il_{short}"
    if col in df_wide.columns:
        pool_cols[short] = col

if not pool_cols:
    raise ValueError(
        f"No expected IL columns found in {IN_PATH}. "
        f"Expected something like: {', '.join('il_'+s for s in DEFAULT_SHORT_ORDER)}"
    )

# Build per-pool series (drop NaNs)
pool_series: Dict[str, pd.Series] = {}
for short, col in pool_cols.items():
    s = pd.to_numeric(df_wide[col], errors="coerce")
    s = s.dropna().reset_index(drop=True)
    pool_series[short] = s

# ==========================================
# 3. BACKTEST (expanding window exceedance)
# ==========================================
def backtest_exceedance(il: pd.Series, min_train: int = 12) -> Dict[str, float]:
    """
    Returns exceedance rates:
      alpha_hat = fraction of OOS months where realized IL < predicted metric
    """
    breaches = {m: 0 for m in METRICS}
    count_oos = 0

    values = il.values.astype(float)
    if len(values) <= min_train:
        return {m: np.nan for m in METRICS}

    for t in range(min_train, len(values)):
        hist = values[:t]
        il_real = values[t]

        # Metrics computed on history
        val_bound = np.nanpercentile(hist, 10)
        val_sigma = np.nanmean(hist) - np.nanstd(hist, ddof=1)
        val_median = np.nanmedian(hist)
        val_mean = np.nanmean(hist)

        metrics_vals = {
            "Bound (10%)": val_bound,
            "Sigma Bound": val_sigma,
            "Median": val_median,
            "Mean": val_mean,
        }

        # Exceedance event: realized worse (more negative) than metric
        for m in METRICS:
            if il_real < metrics_vals[m]:
                breaches[m] += 1

        count_oos += 1

    if count_oos == 0:
        return {m: np.nan for m in METRICS}

    return {m: breaches[m] / count_oos for m in METRICS}


rows = []
for short in DEFAULT_SHORT_ORDER:
    if short not in pool_series:
        continue
    rates = backtest_exceedance(pool_series[short], min_train=MIN_TRAIN_MONTHS)
    n_oos = max(len(pool_series[short]) - MIN_TRAIN_MONTHS, 0)
    row = {"pool_short": short, "n_oos": n_oos, **rates}
    rows.append(row)

df_rates = pd.DataFrame(rows).set_index("pool_short").loc[[s for s in DEFAULT_SHORT_ORDER if s in pool_cols]]

# Save raw exceedance
df_rates.to_csv(OUT_DIR / "exceedance_rates.csv")
print(f"[OK] Wrote raw exceedance rates: {OUT_DIR / 'exceedance_rates.csv'}")

# ==========================================
# 4. NORMALIZATION (make metrics comparable)
# ==========================================
def closeness_to_target(alpha_hat: float, target: float) -> float:
    """
    Normalized score in [0,1], where 1 is perfect calibration (alpha_hat == target).
    Scales by the max possible deviation given the target.
    """
    if alpha_hat is None or np.isnan(alpha_hat):
        return np.nan
    denom = max(target, 1.0 - target)  # max possible |alpha-target|
    score = 1.0 - abs(alpha_hat - target) / denom
    return float(np.clip(score, 0.0, 1.0))


def z_score(alpha_hat: float, target: float, n: int) -> float:
    """
    Standardized deviation from target using binomial standard error:
      z = (alpha_hat - target) / sqrt(target(1-target)/n)
    """
    if alpha_hat is None or np.isnan(alpha_hat) or n <= 0:
        return np.nan
    se = math.sqrt(target * (1.0 - target) / n)
    if se == 0:
        return np.nan
    return float((alpha_hat - target) / se)


# Compute normalized tables
df_scores = df_rates.copy()
df_z = df_rates.copy()

for m in METRICS:
    tgt = TARGETS[m]
    df_scores[m] = [
        closeness_to_target(a, tgt) for a in df_rates[m].values
    ]
    df_z[m] = [
        z_score(a, tgt, int(n)) for a, n in zip(df_rates[m].values, df_rates["n_oos"].values)
    ]

# Keep only metric columns for these exports (plus n_oos where helpful)
df_scores_out = df_scores[["n_oos"] + METRICS]
df_z_out = df_z[["n_oos"] + METRICS]

df_scores_out.to_csv(OUT_DIR / "normalized_closeness_scores.csv")
df_z_out.to_csv(OUT_DIR / "target_deviation_zscores.csv")

print(f"[OK] Wrote normalized scores: {OUT_DIR / 'normalized_closeness_scores.csv'}")
print(f"[OK] Wrote z-scores: {OUT_DIR / 'target_deviation_zscores.csv'}")

# ==========================================
# 5. PLOT (normalized comparison, no heatmap)
# ==========================================
# Grouped bar chart: y = closeness score in [0,1]
plot_df = df_scores_out.reset_index()

x = np.arange(len(plot_df["pool_short"]))
bar_w = 0.18

plt.figure(figsize=(10, 5))
for i, m in enumerate(METRICS):
    plt.bar(x + (i - 1.5) * bar_w, plot_df[m].values, width=bar_w, label=f"{m} (target={TARGETS[m]:.3g})")

plt.ylim(0, 1.05)
plt.xticks(x, plot_df["pool_short"].values)
plt.ylabel("Calibration closeness score (1 = best)")
plt.xlabel("Pool")
plt.title("Backtest calibration across metrics (normalized to each metric’s target)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

fig_path = OUT_DIR / "backtest_normalized_scores.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[OK] Saved plot: {fig_path}")

# ==========================================
# 6. OPTIONAL: LATEX TABLE (raw exceedance)
# ==========================================
latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Exceedance rates $\hat{\alpha}$ for backtested IL risk metrics (expanding-window).}")
latex_lines.append(r"\label{tab:backtest_exceedance}")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\textbf{Pool} & \textbf{Bound (10\%)} & \textbf{Sigma bound} & \textbf{Median} & \textbf{Mean} \\")
latex_lines.append(r"\midrule")

for short in DEFAULT_SHORT_ORDER:
    if short not in df_rates.index:
        continue
    r = df_rates.loc[short]
    latex_lines.append(
        f"{short} & "
        f"{r['Bound (10%)']:.4f} & {r['Sigma Bound']:.4f} & {r['Median']:.4f} & {r['Mean']:.4f} \\\\"
    )

latex_lines.append(r"\midrule")
latex_lines.append(
    r"\textit{Target} & "
    f"{TARGETS['Bound (10%)']:.2f} & {TARGETS['Sigma Bound']:.3f} & {TARGETS['Median']:.2f} & {TARGETS['Mean']:.2f} \\\\"
)
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\footnotesize{\emph{Note:} $\hat{\alpha}$ is the fraction of out-of-sample months where realized IL is more negative than the forecast metric.}")
latex_lines.append(r"\end{table}")

tex_path = OUT_DIR / "latex_table_exceedance.tex"
tex_path.write_text("\n".join(latex_lines), encoding="utf-8")
print(f"[OK] Wrote LaTeX table: {tex_path}")
