from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# -----------------------------
# User inputs / paths (as requested)
# -----------------------------
BASE_PATH = Path("./pool_data/updated")
OUT_DIR = Path("monthly_il")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "monthly_il.dat"

# Keep the same pool ordering you use in figures/legends
DEFAULT_SHORT_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]

GLOB_PATTERNS = ["0x*.json"]  # discover pool JSON exports


# -----------------------------
# Helpers
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
    # Graph poolDayDatas typically uses 'date' (unix seconds)
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


def _month_key_from_ts(ts: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m")


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
      2) any key named 'poolDayDatas'
      3) first list that is list[dict]
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


def il_from_R(R: float) -> Optional[float]:
    """IL(R) = 2*sqrt(R)/(1+R) - 1, defined for R>0."""
    if R is None or not (R > 0):
        return None
    return (2.0 * math.sqrt(R) / (1.0 + R)) - 1.0


# -----------------------------
# Core: monthly IL per pool
# -----------------------------
def load_monthly_il(path: Path, price_field: str = "close") -> Dict[str, float]:
    """
    For each month:
      p_start = first valid daily price in that month
      p_end   = last valid daily price in that month
      R       = p_end / p_start
      IL      = 2*sqrt(R)/(1+R) - 1
    Returns: { "YYYY-MM" -> IL_month }.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    recs = _find_list_of_dicts_recursive(raw)
    if recs is None:
        raise ValueError(f"Could not find poolDayDatas records in JSON: {path}")

    # month -> (first_ts, p_start, last_ts, p_end)
    per_month: Dict[str, Dict[str, Any]] = {}

    for r in recs:
        ts = _parse_ts(r)
        if ts is None:
            continue

        p = _to_float(r.get(price_field))
        if p is None or p <= 0:
            continue

        mk = _month_key_from_ts(ts)
        slot = per_month.get(mk)
        if slot is None:
            per_month[mk] = {"first_ts": ts, "p_start": p, "last_ts": ts, "p_end": p}
        else:
            if ts < slot["first_ts"]:
                slot["first_ts"] = ts
                slot["p_start"] = p
            if ts > slot["last_ts"]:
                slot["last_ts"] = ts
                slot["p_end"] = p

    out: Dict[str, float] = {}
    for mk, v in per_month.items():
        p_start = v.get("p_start")
        p_end = v.get("p_end")
        if p_start is None or p_end is None or p_start <= 0:
            continue
        R = p_end / p_start
        IL = il_from_R(R)
        if IL is None:
            continue
        out[mk] = float(IL)

    return out


def write_dat(
    all_series: Dict[str, Dict[str, float]],
    out_path: Path,
    order: List[str],
    drop_first_and_last_month: bool = True,
) -> None:
    """
    Writes a wide .dat:
      month  il_45  il_cb  il_4e ...
    where month is a simple index 1..N in chronological order of YYYY-MM keys.

    If drop_first_and_last_month=True, the earliest and latest months (global across all pools)
    are removed before writing.
    """
    months_all = sorted({mk for s in all_series.values() for mk in s.keys()})

    if drop_first_and_last_month and len(months_all) >= 3:
        months = months_all[1:-1]  # remove global first and global last month
    else:
        months = months_all

    month_to_idx = {mk: i + 1 for i, mk in enumerate(months)}

    cols = ["month"] + [f"il_{k}" for k in order if k in all_series]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for mk in months:
            row = [str(month_to_idx[mk])]
            for k in order:
                if k not in all_series:
                    continue
                v = all_series[k].get(mk)
                row.append("" if v is None else f"{v:.10g}")
            f.write("\t".join(row) + "\n")


def main() -> None:
    # discover JSON files
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files.extend(BASE_PATH.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No JSON files matched patterns {GLOB_PATTERNS} under {BASE_PATH}")

    # load pools in your desired set/order
    series: Dict[str, Dict[str, float]] = {}
    for p in files:
        short = _pool_short_from_filename(p)
        if short not in DEFAULT_SHORT_ORDER:
            continue
        series[short] = load_monthly_il(p, price_field="close")

    if not series:
        raise ValueError(
            f"Found JSON files, but none matched DEFAULT_SHORT_ORDER={DEFAULT_SHORT_ORDER} under {BASE_PATH}."
        )

    order = [k for k in DEFAULT_SHORT_ORDER if k in series]
    write_dat(series, OUT_PATH, order, drop_first_and_last_month=True)

    months_written = None
    # compute written months count (for logging)
    months_all = sorted({mk for s in series.values() for mk in s.keys()})
    if len(months_all) >= 3:
        months_written = len(months_all) - 2
    else:
        months_written = len(months_all)

    print(f"Wrote: {OUT_PATH}")
    print("Pools:", ", ".join(order))
    print("Months written:", months_written, "(dropped global first+last month)")


if __name__ == "__main__":
    main()
