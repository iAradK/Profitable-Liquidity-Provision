"""Writing results: atomic tables, finished-experiment markers, and per-item caching of parallel work."""

import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from subgraph.parallel import run_each
from subgraph.storage import load_json, save_json

log = logging.getLogger("experiments")

DONE_MARKER = "DONE"


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index, float_format="%.6g")
    os.replace(tmp, path)


def is_done(out_dir: Path) -> bool:
    return (out_dir / DONE_MARKER).exists()


def mark_done(out_dir: Path) -> None:
    (out_dir / DONE_MARKER).write_text("finished\n", encoding="utf-8")


def _compute_and_save(item: tuple[str, Any], worker: Callable[[Any], Any], cache_dir: Path) -> Any:
    key, payload = item
    result = worker(payload)
    save_json(cache_dir / f"{key}.json", result)
    return result


def cached_map(items: list[tuple[str, Any]], worker: Callable[[Any], Any], cache_dir: Path,
               max_workers: int, force_refresh: bool) -> dict[str, Any]:
    """Run `worker` on every (key, payload) in parallel processes, saving each result as <key>.json as soon
    as it is done. Items with a saved result are loaded instead of recomputed. Raises if any item failed
    (the finished ones stay saved, so the next run only redoes the failures).

    `worker` must be a top-level function and its results JSON-serializable.
    """
    pending = [item for item in items if force_refresh or not (cache_dir / f"{item[0]}.json").exists()]
    if len(pending) < len(items):
        log.info("  %d of %d items already computed", len(items) - len(pending), len(items))
    batch = run_each(pending, partial(_compute_and_save, worker=worker, cache_dir=cache_dir),
                     name=lambda item: item[0], max_workers=max_workers, use_processes=True)
    if batch.failures:
        raise RuntimeError(f"{len(batch.failures)} of {len(items)} items failed: {sorted(batch.failures)}")
    return {key: load_json(cache_dir / f"{key}.json") for key, _ in items}
