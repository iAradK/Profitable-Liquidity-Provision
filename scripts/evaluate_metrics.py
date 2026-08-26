from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------
# User paths
# -----------------------
BASE_PATH = Path("./pool_data/updated")
OUT_DIR = Path("metric_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Configuration
# -----------------------

# Pick pools by file path (recommended): put the JSON files you want in this list.
# If left empty, we will auto-load all JSON files under BASE_PATH matching FILE_GLOB.
POOL_FILES: List[Path] = []

# If POOL_FILES is empty, we use this glob to discover files:
FILE_GLOB = "0x*.json"

# Legend/x-axis order for your bar plot (use your preferred order here).
# You can replace this list with any subset you want (e.g., ["ab","cd","ef"]).
POOL_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]

# Which price field to use from poolDayDatas records:
# In The Graph, poolDayDatas typically has token0Price and token1Price (strings).
# Choose one; IL is symmetric in 1/R, but the monthly ratio R will invert if you swap.
PRICE_FIELD_CANDIDATES = ("token0Price", "token1Price")

# Use symmetric monthly ratios R* = max(R, 1/R) before computing moments on R.
# This often matches “magnitude of deviation” logic and avoids mean(R) being < 1 due to direction.
USE_SYMMETRIC_RATIO = True

# Time horizon (h): monthly ratios computed from end-of-month prices (h = 1 month).
# -----------------------


# -----------------------
# Helpers
# -----------------------
def _pool_short_from_filename(p: Path) -> str:
    m = re.match(r"0x([0-9a-fA-F]{2})", p.name)
    return (m.group(1) if m else p.stem[:2]).lower()


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _find_pool_day_datas(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively find a list under a key named 'poolDayDatas' (GraphQL export style).
    Accepts structures like: {"data":{"poolDayDatas":[...]}}.
    """
    if isinstance(obj, dict):
        if "poolDayDatas" in obj and isinstance(obj["poolDayDatas"], list):
            lst = obj["poolDayDatas"]
            if all(isinstance(x, dict) for x in lst):
                return lst
        for v in obj.values():
            found = _find_pool_day_datas(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_pool_day_datas(v)
            if found is not None:
                return found
    return None


def _extract_daily_df(json_path: Path) -> pd.DataFrame:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    recs = _find_pool_day_datas(raw)
    if recs is None:
        raise ValueError(f"Could not find 'poolDayDatas' records in: {json_path}")

    df = pd.DataFrame(recs)

    # Date handling: poolDayDatas uses 'date' as UNIX seconds (start of day).
    if "date" not in df.columns:
        raise ValueError(f"No 'date' field found in poolDayDatas for: {json_path}")

    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["dt"] = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True)
    df["ym"] = df["dt"].dt.strftime("%Y-%m")

    # Pick a price field that exists
    price_field = None
    for cand in PRICE_FIELD_CANDIDATES:
        if cand in df.columns:
            price_field = cand
            break
    if price_field is None:
        raise ValueError(
            f"None of {PRICE_FIELD_CANDIDATES} found in {json_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df["price"] = pd.to_numeric(df[price_field], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df.sort_values("dt").reset_index(drop=True)

    return df[["dt", "ym", "price"]]


def _end_of_month_prices(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    One price per month: take the last record within each YYYY-MM.
    """
    # Last record per ym
    df_eom = df_daily.groupby("ym", as_index=False).tail(1).copy()
    df_eom = df_eom.sort_values("dt").reset_index(drop=True)
    df_eom["month_idx"] = np.arange(1, len(df_eom) + 1)
    return df_eom[["ym", "month_idx", "price"]]


def _monthly_ratios(df_eom: pd.DataFrame) -> pd.Series:
    """
    R_t = p_t / p_{t-1} (one-month holding period)
    """
    p = df_eom["price"].to_numpy(dtype=float)
    if len(p) < 2:
        return pd.Series([], dtype=float)
    R = p[1:] / p[:-1]
    if USE_SYMMETRIC_RATIO:
        R = np.maximum(R, 1.0 / R)
    return pd.Series(R)


def il_from_R(R: np.ndarray | float) -> np.ndarray | float:
    """
    IL(R) = 2*sqrt(R)/(1+R) - 1   in [-1, 0].
    """
    return 2.0 * np.sqrt(R) / (1.0 + R) - 1.0


@dataclass
class PoolMetrics:
    pool: str
    il_median_pct: float
    il_bound_5_95_pct: float
    il_sigma_bound_pct: float
    il_mean_pct: float
    n_months: int


def compute_pool_metrics(pool_short: str, json_path: Path) -> PoolMetrics:
    df_daily = _extract_daily_df(json_path)
    df_eom = _end_of_month_prices(df_daily)
    R = _monthly_ratios(df_eom).to_numpy(dtype=float)
    n = int(R.size)

    if n == 0:
        return PoolMetrics(pool_short, np.nan, np.nan, np.nan, np.nan, 0)

    mu = float(np.mean(R))
    sigma = float(np.std(R, ddof=0))
    med = float(np.median(R))
    p05 = float(np.quantile(R, 0.05))
    p95 = float(np.quantile(R, 0.95))

    # IL_mean: IL(mu_R)
    il_mean = float(il_from_R(mu))

    # IL_median: IL(median_R)
    il_median = float(il_from_R(med))

    # IL_bound (5--95): min(IL(p05), IL(p95))
    il_bound = float(min(il_from_R(p05), il_from_R(p95)))

    # IL_sigma-bound: evaluate at mu±sigma if positive; otherwise skip invalid edge.
    candidates: List[float] = []
    for edge in (mu - sigma, mu + sigma):
        if edge > 0:
            candidates.append(float(il_from_R(edge)))
    il_sigma_bound = float(min(candidates)) if candidates else np.nan

    # Convert to percent
    return PoolMetrics(
        pool=pool_short,
        il_median_pct=100.0 * il_median,
        il_bound_5_95_pct=100.0 * il_bound,
        il_sigma_bound_pct=100.0 * il_sigma_bound,
        il_mean_pct=100.0 * il_mean,
        n_months=n,
    )


def metrics_to_tikz(metrics: List[PoolMetrics], x_order: List[str]) -> str:
    """
    Emit a pgfplots bar-chart snippet with coordinates for the four series.
    """
    m_by_pool = {m.pool: m for m in metrics}
    pools = [p for p in x_order if p in m_by_pool]

    def coord_line(field: str) -> str:
        parts = []
        for p in pools:
            v = getattr(m_by_pool[p], field)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
            parts.append(f"({p},{v:.6f})")
        return " ".join(parts)

    return f"""
% Auto-generated from Python
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.48\\textwidth,
    height=0.42\\textwidth,
    ybar,
    bar width=6pt,
    enlarge x limits=0.18,
    ymin=-2.2, ymax=0.1,
    ymajorgrids=true,
    grid style={{dotted}},
    ylabel={{Impermanent loss (\\%)}},
    symbolic x coords={{{','.join(pools)}}},
    xtick=data,
    xticklabel style={{rotate=45, anchor=east}},
    legend style={{
        at={{(0.63,0.02)}},
        anchor=south west,
        draw=black,
        fill=white,
        font=\\small,
    }},
    legend cell align={{left}},
]

% IL_median
\\addplot[draw=blue!70!black, fill=blue!25, bar shift=-12pt] coordinates {{ {coord_line('il_median_pct')} }};
\\addlegendentry{{$\\mathrm{{IL}}_{{\\text{{median}}}}$}}

% IL_bound (5--95)
\\addplot[draw=red!70!black, fill=red!25, bar shift=-4pt] coordinates {{ {coord_line('il_bound_5_95_pct')} }};
\\addlegendentry{{$\\mathrm{{IL}}_{{\\text{{bound}}}}~(5\\text{{--}}95)$}}

% IL_sigma-bound
\\addplot[draw=brown!70!black, fill=brown!20, bar shift=4pt] coordinates {{ {coord_line('il_sigma_bound_pct')} }};
\\addlegendentry{{$\\mathrm{{IL}}_{{\\sigma\\text{{--}}\\text{{bound}}}}$}}

% IL_mean
\\addplot[draw=black, fill=black!45, bar shift=12pt] coordinates {{ {coord_line('il_mean_pct')} }};
\\addlegendentry{{$\\mathrm{{IL}}_{{\\text{{mean}}}}$}}

\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Summary statistics computed over the monthly IL samples for each pool.}}
\\label{{fig:monthly_il_summary_metrics}}
\\end{{figure}}
""".strip()


def main() -> None:
    # Discover files
    files = POOL_FILES[:] if POOL_FILES else sorted(BASE_PATH.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No JSON files found under {BASE_PATH} with pattern {FILE_GLOB}")

    # Compute metrics for each file
    metrics: List[PoolMetrics] = []
    for fp in files:
        pool_short = _pool_short_from_filename(fp)
        # If you want ONLY the pools in POOL_ORDER, uncomment:
        # if pool_short not in set(POOL_ORDER): continue
        metrics.append(compute_pool_metrics(pool_short, fp))

    # Save a CSV for inspection
    df_out = pd.DataFrame([m.__dict__ for m in metrics]).sort_values("pool")
    csv_path = OUT_DIR / "monthly_il_metrics.csv"
    df_out.to_csv(csv_path, index=False)

    # Emit TikZ snippet
    tikz = metrics_to_tikz(metrics, POOL_ORDER)
    tex_path = OUT_DIR / "monthly_il_summary_metrics.tex"
    tex_path.write_text(tikz + "\n", encoding="utf-8")

    print(f"Wrote CSV : {csv_path}")
    print(f"Wrote TeX : {tex_path}")
    print("\n--- TikZ snippet ---\n")
    print(tikz)


if __name__ == "__main__":
    main()
