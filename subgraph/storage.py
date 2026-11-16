"""JSON/CSV file helpers. All writes are atomic so an interrupted run never leaves a truncated file."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def iso(ts: int | str | float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def drop_unfinished(rows: list[dict], time_key: str, period_seconds: int, saved_path: Path) -> list[dict]:
    """Drop trailing rows whose hour/day had not ended when the file was saved, so they are fetched again.

    The subgraph keeps updating the current period's row until it ends, so a row saved mid-period is partial.
    """
    if not saved_path.exists():
        return rows
    saved_at = saved_path.stat().st_mtime
    end = len(rows)
    while end and int(rows[end - 1][time_key]) + period_seconds > saved_at:
        end -= 1
    return rows[:end]


def flatten(row: dict, prefix: str = "") -> dict:
    """{'token0': {'id': ..}} -> {'token0_id': ..}; values stay strings (no float conversion)."""
    out = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "_"))
        else:
            out[name] = value
    return out


def write_csv(path: Path, rows: list[dict], first_cols: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten(r) for r in rows]
    cols = list(first_cols)
    for row in flat:
        for key in row:
            if key not in cols:
                cols.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(flat)
    os.replace(tmp, path)
