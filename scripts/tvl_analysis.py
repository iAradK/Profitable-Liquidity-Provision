from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- user-provided paths ---
BASE_PATH = Path("./pool_data/updated")
OUT_DIR = Path("tvl_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "tvl_monthly_end.dat"

# Optional: keep the same pool ordering you use in figures/legends
DEFAULT_SHORT_ORDER = ["45", "cb", "4e", "11", "9d", "5a", "a6", "1d"]


# --- CHANGE THESE TO MATCH YOUR REAL FILENAMES ---
# Most robust: all json files that start with 0x
GLOB_PATTERNS = [
    "0x*.json",
    # If you want to narrow it (uncomment any that apply):
    # "0x*monthly*.json",
    # "0x*_monthly*.json",
    # "0x*poolDayData*.json",
]


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
    # Your data uses 'date' (UNIX seconds), so put it first.
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


def _find_list_of_dicts_recursive(data: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively search for a list whose elements are dicts.

    Preference order:
      1) data['data']['poolDayDatas'] if present (common GraphQL export)
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


def _pool_short_from_filename(path: Path) -> str:
    """
    Use first byte after 0x, e.g. 0x8a... -> '8a'
    """
    m = re.match(r"0x([0-9a-fA-F]{2})", path.name)
    return (m.group(1) if m else path.stem[:2]).lower()


def load_monthly_end_tvl(path: Path) -> Dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recs = _find_list_of_dicts_recursive(raw)
    if recs is None:
        raise ValueError(f"Could not find a list of records in JSON: {path}")

    best: Dict[str, Tuple[int, float]] = {}  # month -> (ts, tvl)
    for r in recs:
        ts = _parse_ts(r)
        if ts is None:
            continue

        tvl = None
        for key in ("tvlUSD", "tvlUsd", "totalValueLockedUSD", "totalValueLockedUsd"):
            if key in r:
                tvl = _to_float(r[key])
                break
        if tvl is None:
            continue

        mk = _month_key_from_ts(ts)
        if mk not in best or ts > best[mk][0]:
            best[mk] = (ts, tvl)

    return {mk: tvl for mk, (ts, tvl) in best.items()}


def write_dat(all_series: Dict[str, Dict[str, float]], out_path: Path, order: List[str]) -> None:
    months = sorted({mk for s in all_series.values() for mk in s.keys()})
    month_to_idx = {mk: i + 1 for i, mk in enumerate(months)}

    cols = ["month"] + [f"tvl_{k}" for k in order if k in all_series]
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
    # collect files using the glob patterns
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files.extend(BASE_PATH.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No JSON files matched patterns {GLOB_PATTERNS} under {BASE_PATH}")

    # Load all pools found
    series: Dict[str, Dict[str, float]] = {}
    for p in files:
        short = _pool_short_from_filename(p)
        series[short] = load_monthly_end_tvl(p)

    order = [k for k in DEFAULT_SHORT_ORDER if k in series] + sorted(
        [k for k in series if k not in DEFAULT_SHORT_ORDER]
    )

    write_dat(series, OUT_PATH, order)

    print(f"Wrote: {OUT_PATH}")
    print("Read files:", len(files))
    print("Pools:", ", ".join(order))


if __name__ == "__main__":
    main()
