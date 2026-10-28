"""Run one task per item concurrently; a failing item is logged and reported, never fatal."""

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Generic, TypeVar

log = logging.getLogger("subgraph")

Item = TypeVar("Item")
Result = TypeVar("Result")


@dataclass
class BatchResult(Generic[Result]):
    results: dict[str, Result] = field(default_factory=dict)  # item name -> result
    failures: dict[str, str] = field(default_factory=dict)  # item name -> error message


def run_each(
    items: Iterable[Item],
    task: Callable[[Item], Result],
    name: Callable[[Item], str],
    max_workers: int,
    use_processes: bool = False,
) -> BatchResult[Result]:
    """Threads for I/O-bound tasks; processes (task must be picklable) for CPU-bound ones."""
    executor_type: type[Executor] = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    batch: BatchResult[Result] = BatchResult()
    executor = executor_type(max_workers=max_workers)
    try:
        futures = {executor.submit(task, item): name(item) for item in items}
        for future in as_completed(futures):
            item_name = futures[future]
            try:
                batch.results[item_name] = future.result()
            except Exception as exc:
                log.error("%s failed: %s: %s", item_name, type(exc).__name__, exc)
                batch.failures[item_name] = f"{type(exc).__name__}: {exc}"
    except KeyboardInterrupt:
        # Don't start queued items; items already running finish (and save) before exit.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return batch
