from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Paths / settings
# -----------------------------
BASE_PATH = Path("./pool_data/updated")

OUT_DIR = Path("monthly_R")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STATS_DAT = OUT_DIR / "monthly_R_stats_summary.dat"
OUT_STATS_CSV = OUT_DIR / "monthly_R_stats_summary.csv"

DEFAULT_SHORT_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]
GLOB_PATTERNS = ["0x*.json"]

PRICE_FIELD = "close"

# If True, removes the earliest and latest month (often incomplete endpoints)
DROP_FIRST_LAST_MONTHS = True


# -----------------------------
# Helpers (robust JSON parsing)
# -----------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _parse_ts(obj: Dict[str, Any]) -> Optional[int]:
    for k in ("date", "timestamp", "time", "blockTimestamp", "block_timestamp"):
        if k not in obj:
            continue
        v = obj[k]
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            try:
                s2 = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except Exception:
                pass
    return None


def _find_list_of_dicts_recursive(data: Any) -> Optional[List[Dict[str, Any]]]:
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


def _pool_short_from_filename(path: Path) -> str:
    m = re.match(r"0x([0-9a-fA-F]{2})", path.name)
    return (m.group(1) if m else path.stem[:2]).lower()


def first_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[0]) if len(s) else float("nan")


def last_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


# -----------------------------
# Core computation
# -----------------------------
def compute_monthly_R_from_json(json_path: Path, price_field: str = "close") -> pd.DataFrame:
    """
    Returns a DataFrame with:
      month (timestamp), p_start, p_end, R_month
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    recs = _find_list_of_dicts_recursive(raw)
    if recs is None or not recs:
        raise ValueError(f"Could not find poolDayDatas records in JSON: {json_path}")

    df = pd.DataFrame(recs)

    if price_field not in df.columns:
        raise ValueError(f"Missing '{price_field}' in {json_path.name}. Columns: {list(df.columns)[:30]}")

    # timestamps
    if "date" in df.columns:
        # common Graph export: unix seconds
        df["ts"] = pd.to_numeric(df["date"], errors="coerce")
    else:
        # fallback: try parse per row
        df["ts"] = df.apply(lambda r: _parse_ts(r.to_dict()), axis=1)

    df = df.dropna(subset=["ts"]).copy()
    df["date"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True).dt.tz_convert(None)

    # prices
    df["p"] = pd.to_numeric(df[price_field], errors="coerce")
    df.loc[df["p"] <= 0, "p"] = pd.NA

    df = df.dropna(subset=["p"]).sort_values("date").reset_index(drop=True)

    # month bucket
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month", as_index=False)
          .agg(p_start=("p", first_valid),
               p_end=("p", last_valid))
          .dropna(subset=["p_start", "p_end"])
          .reset_index(drop=True)
    )

    monthly["R_month"] = monthly["p_end"] / monthly["p_start"]
    monthly["month_str"] = monthly["month"].dt.strftime("%Y-%m")

    return monthly[["month", "month_str", "p_start", "p_end", "R_month"]]


def drop_first_last_months(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 2:
        return df.iloc[0:0].copy()
    return df.iloc[1:-1].reset_index(drop=True)


def main() -> None:
    # discover files
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files.extend(BASE_PATH.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No JSON files matched {GLOB_PATTERNS} under {BASE_PATH}")

    # compute per-pool, keep only desired shorts
    summary_rows: List[Dict[str, Any]] = []
    pool_series: Dict[str, pd.DataFrame] = {}

    for p in files:
        short = _pool_short_from_filename(p)
        if short not in DEFAULT_SHORT_ORDER:
            continue

        monthly = compute_monthly_R_from_json(p, price_field=PRICE_FIELD)
        if DROP_FIRST_LAST_MONTHS:
            monthly = drop_first_last_months(monthly)

        pool_series[short] = monthly

        # save per-pool monthly R file
        out_csv = OUT_DIR / f"monthly_R_{p.stem}.csv"
        monthly.to_csv(out_csv, index=False)

        Rvals = pd.to_numeric(monthly["R_month"], errors="coerce").dropna()
        summary_rows.append({
            "pool_short": short,
            "pool": p.stem,
            "n_months": int(len(Rvals)),
            "mean_R_month": float(Rvals.mean()) if len(Rvals) else float("nan"),
            "std_R_month": float(Rvals.std(ddof=1)) if len(Rvals) >= 2 else float("nan"),
        })

        print(f"[OK] {short}: wrote {out_csv.name} ({len(monthly)} months)")

    if not summary_rows:
        raise ValueError(f"Found JSONs, but none matched DEFAULT_SHORT_ORDER={DEFAULT_SHORT_ORDER}")

    # order the summary rows according to DEFAULT_SHORT_ORDER
    order_index = {k: i for i, k in enumerate(DEFAULT_SHORT_ORDER)}
    summary_rows.sort(key=lambda r: order_index.get(r["pool_short"], 999))

    # write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_STATS_CSV, index=False)

    # write summary DAT for pgfplots (tab-separated)
    # columns: pool_short mean_R_month std_R_month n_months
    with OUT_STATS_DAT.open("w", encoding="utf-8") as f:
        f.write("pool_short\tmean_R_month\tstd_R_month\tn_months\n")
        for _, r in summary_df.iterrows():
            f.write(f"{r['pool_short']}\t{r['mean_R_month']:.10g}\t{r['std_R_month']:.10g}\t{int(r['n_months'])}\n")

    print(f"\nSaved summary CSV : {OUT_STATS_CSV}")
    print(f"Saved summary DAT : {OUT_STATS_DAT}")
    print(f"DROP_FIRST_LAST_MONTHS={DROP_FIRST_LAST_MONTHS}")


if __name__ == "__main__":
    main()
