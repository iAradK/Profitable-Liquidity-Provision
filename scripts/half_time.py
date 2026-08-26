from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ==========================================
# 1. CONFIGURATION (aligned with your other scripts)
# ==========================================
BASE_PATH = Path("./pool_data/updated")

OUT_DIR = Path("half_life")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the same pool ordering you use in figures/legends
DEFAULT_SHORT_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]

GLOB_PATTERNS = ["0x*.json"]  # discover JSON exports under BASE_PATH

PRICE_FIELD = "close"
WINDOW_DAYS = 180


# ==========================================
# 2. HELPERS (robust file discovery + parsing)
# ==========================================
def _pool_short_from_filename(path: Path) -> str:
    """
    Use first byte after 0x, e.g. 0x4e... -> '4e'
    """
    m = re.match(r"0x([0-9a-fA-F]{2})", path.name)
    return (m.group(1) if m else path.stem[:2]).lower()


def _find_list_of_dicts_recursive(data: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively search for a list whose elements are dicts.

    Preference order:
      1) data['data']['poolDayDatas']
      2) any key named 'poolDayDatas' anywhere
      3) the first list found that is list[dict]
    """
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], dict):
            inner = data["data"]
            if "poolDayDatas" in inner and isinstance(inner["poolDayDatas"], list):
                lst = inner["poolDayDatas"]
                if all(isinstance(x, dict) for x in lst):
                    return lst

    if isinstance(data, dict):
        if "poolDayDatas" in data and isinstance(data["poolDayDatas"], list):
            lst = data["poolDayDatas"]
            if all(isinstance(x, dict) for x in lst):
                return lst

    if isinstance(data, dict):
        for v in data.values():
            found = _find_list_of_dicts_recursive(v)
            if found is not None:
                return found

    if isinstance(data, list):
        if len(data) == 0 or all(isinstance(x, dict) for x in data):
            return [x for x in data if isinstance(x, dict)]
        for v in data:
            found = _find_list_of_dicts_recursive(v)
            if found is not None:
                return found

    return None


# ==========================================
# 3. DATA LOADER
# ==========================================
def load_pool_data(file_path: Path, price_field: str = "close") -> pd.Series:
    """
    Parses Uniswap V3 Subgraph JSON structure and returns a daily close price series.

    Expected structure: { "data": { "poolDayDatas": [ ... ] } }
    Returns:
      pd.Series indexed by daily datetime (naive), containing close prices.
    """
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    recs = _find_list_of_dicts_recursive(raw)
    if recs is None or len(recs) == 0:
        raise ValueError(f"Invalid JSON structure in {file_path.name}: could not find poolDayDatas.")

    df = pd.DataFrame(recs)

    if "date" not in df.columns or price_field not in df.columns:
        raise ValueError(f"Missing 'date' or '{price_field}' columns in {file_path.name}")

    df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s", utc=True).dt.tz_convert(None)
    df[price_field] = pd.to_numeric(df[price_field], errors="coerce")

    df = df.dropna(subset=["date", price_field]).sort_values("date").set_index("date")
    df = df[df[price_field] > 0]

    # Daily close series, forward-fill gaps
    s = df[price_field].resample("D").last().ffill().dropna()

    return s


# ==========================================
# 4. ANALYSIS LOGIC (Rolling AR(1) on log-prices)
# ==========================================
def calculate_rolling_halflife(price_series: pd.Series, window: int = 180) -> Tuple[float, float]:
    """
    1) Invert price if mean < 1 (standardize orientation).
    2) Log transform: X_t = ln(Price).
    3) Rolling AR(1): X_{t} = alpha + phi * X_{t-1}.
    4) Half-life HL = -ln(2) / ln(phi), for 0 < phi < 0.999.
    Returns (mean_HL, mean_phi) across valid rolling windows.
    """
    if price_series.empty or len(price_series) < window + 2:
        return float("nan"), float("nan")

    # 1) Orientation adjustment (same logic you used elsewhere)
    if price_series.mean() < 1:
        price_series = 1.0 / price_series

    # 2) log
    X = np.log(price_series.values.astype(float))

    model = LinearRegression()
    phis: List[float] = []

    # 3) rolling AR(1)
    # window_slice has length=window; we regress x[t] on x[t-1] inside the window.
    for i in range(len(X) - window):
        window_slice = X[i : i + window]
        y = window_slice[1:]
        x = window_slice[:-1].reshape(-1, 1)

        model.fit(x, y)
        phi = float(model.coef_[0])

        if 0 < phi < 0.999:
            phis.append(phi)

    if not phis:
        return float("nan"), float("nan")

    phis_arr = np.array(phis, dtype=float)
    halflives = -np.log(2.0) / np.log(phis_arr)

    return float(np.mean(halflives)), float(np.mean(phis_arr))


# ==========================================
# 5. MAIN
# ==========================================
def main() -> None:
    # discover JSON files
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files.extend(BASE_PATH.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No JSON files matched patterns {GLOB_PATTERNS} under {BASE_PATH}")

    # keep only the pools in DEFAULT_SHORT_ORDER
    pool_map: Dict[str, Path] = {}
    for p in files:
        short = _pool_short_from_filename(p)
        if short in DEFAULT_SHORT_ORDER:
            pool_map[short] = p

    missing = [k for k in DEFAULT_SHORT_ORDER if k not in pool_map]
    if missing:
        print("[WARN] Missing pools (no matching JSON found):", ", ".join(missing))

    results: List[Dict[str, Any]] = []
    print(f"Analyzing {len(pool_map)} pools...")

    for short in DEFAULT_SHORT_ORDER:
        if short not in pool_map:
            continue

        path = pool_map[short]
        pool_label = f"0x{short}"

        try:
            prices = load_pool_data(path, price_field=PRICE_FIELD)
            hl, phi = calculate_rolling_halflife(prices, window=WINDOW_DAYS)

            if not np.isnan(hl):
                results.append({"pool_short": short, "pool": path.stem, "HalfLife": hl, "Phi": phi})
                print(f"  [OK] {pool_label}: HL={hl:.2f} days, phi={phi:.4f}")
            else:
                print(f"  [SKIP] {pool_label}: insufficient mean-reverting windows (phi in (0,1)).")

        except Exception as e:
            print(f"  [ERR] {pool_label}: {e}")

    if not results:
        print("No valid results found to plot.")
        return

    df_res = pd.DataFrame(results)

    # sort by HalfLife descending (as before)
    df_res = df_res.sort_values("HalfLife", ascending=False).reset_index(drop=True)

    # save stats
    df_res.to_csv(OUT_DIR / "halflife_stats.csv", index=False)

    # also save a simple .dat for pgfplots if you want later
    dat_path = OUT_DIR / "halflife_stats.dat"
    with dat_path.open("w", encoding="utf-8") as f:
        f.write("pool_short\tHalfLife\tPhi\n")
        for _, r in df_res.iterrows():
            f.write(f"{r['pool_short']}\t{r['HalfLife']:.10g}\t{r['Phi']:.10g}\n")

    # Plotting (same as your current style)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        ["0x" + s for s in df_res["pool_short"]],
        df_res["HalfLife"],
        color="#7b68ee",
        alpha=0.75,
        edgecolor="navy",
        height=0.6,
    )

    # annotate phi inside bars
    for bar, phi in zip(bars, df_res["Phi"]):
        width = bar.get_width()
        plt.text(
            width - (width * 0.02),
            bar.get_y() + bar.get_height() / 2,
            rf"$\phi = {phi:.3f}$",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

    plt.gca().invert_yaxis()
    plt.xlabel(f"Mean {WINDOW_DAYS}-day rolling half-life (days)", fontsize=12)
    plt.ylabel("Pool", fontsize=12)
    plt.title("Mean Reversion Predictability (Half-Life)", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_png = OUT_DIR / "mean_reversion_plot.png"
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"\nSaved: {OUT_DIR / 'halflife_stats.csv'}")
    print(f"Saved: {dat_path}")
    print(f"Graph saved to {out_png}")


if __name__ == "__main__":
    main()
