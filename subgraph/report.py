"""Sanity report per pool, and spot check of hourly closes vs the existing daily closes."""

import logging
import statistics
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .config import Paths, Pool
from .hourly import HOUR, load_hourly_rows
from .meta import PoolMeta
from .parallel import BatchResult, run_each
from .storage import iso, load_json, write_csv

log = logging.getLogger("subgraph")

DAY = 86400
STALE_FEE_GROWTH_LIMIT = 0.05  # flag pools where feeGrowth didn't move in >5% of hours with volume


@dataclass(frozen=True)
class PoolReport:
    sanity: dict
    spot_check: dict | None  # None when there are no daily closes to compare with


def _sanity(pool: Pool, created_ts: str | None, rows: list[dict]) -> dict:
    rec = {"pool": pool.address, "label": pool.label, "n_hours": len(rows)}
    if created_ts is not None:
        rec["created"] = iso(created_ts)
    if not rows:
        return rec
    ts = [int(r["periodStartUnix"]) for r in rows]
    gaps = [(b - a) // HOUR - 1 for a, b in zip(ts, ts[1:])]
    span = (ts[-1] - ts[0]) // HOUR + 1
    rec.update({
        "first": iso(ts[0]), "last": iso(ts[-1]),
        "span_hours": span, "missing_hours": span - len(rows),
        "coverage": round(len(rows) / span, 4),
        "max_gap_hours": max(gaps, default=0),
        "gaps_over_24h": sum(g > 24 for g in gaps),
        "duplicates_or_unsorted": sum(g < 0 for g in gaps),
    })
    fee_growth = {}
    for tag, key in (("fg0", "feeGrowthGlobal0X128"), ("fg1", "feeGrowthGlobal1X128")):
        if key not in rows[0]:
            continue
        values = fee_growth[tag] = [int(r[key]) for r in rows]
        decreases = sum(b < a for a, b in zip(values, values[1:]))
        rec[f"{tag}_monotonic"] = decreases == 0
        rec[f"{tag}_decreases"] = decreases
        rec[f"{tag}_zero_rows"] = sum(v == 0 for v in values)
    if len(fee_growth) == 2 and "volumeUSD" in rows[0]:
        # hours with swaps but neither feeGrowth moved -> the subgraph is not updating feeGrowth
        stale = active = 0
        fg0, fg1 = fee_growth["fg0"], fee_growth["fg1"]
        for i in range(1, len(rows)):
            if float(rows[i]["volumeUSD"]) > 0:
                active += 1
                stale += fg0[i] == fg0[i - 1] and fg1[i] == fg1[i - 1]
        rec["fg_stale_frac"] = round(stale / active, 4) if active else None
    return rec


def _spot_check(pool: Pool, hourly: list[dict], daily_dir: Path) -> dict | None:
    """Daily close vs close of the last hourly row of the same UTC day."""
    daily = load_json(daily_dir / f"{pool.address}.json", {}).get("data", {}).get("poolDayDatas", [])
    if not daily or not hourly:
        return None
    last_close = {}
    for row in hourly:  # sorted asc, so the last write per day wins
        last_close[int(row["periodStartUnix"]) // DAY * DAY] = float(row["close"])
    diffs = []
    for day_row in daily:
        day = int(day_row["date"])
        if day in last_close and float(day_row["close"]) != 0:
            diffs.append((abs(last_close[day] / float(day_row["close"]) - 1), day))
    if not diffs:
        return None
    rel = [d for d, _ in diffs]
    worst = max(diffs)
    return {"pool": pool.address, "label": pool.label, "days_compared": len(diffs),
            "median_rel_diff": statistics.median(rel), "max_rel_diff": worst[0],
            "worst_day": iso(worst[1])[:10],
            "frac_within_1e-6": round(sum(x < 1e-6 for x in rel) / len(rel), 4)}


def analyse_pool(pool: Pool, meta: PoolMeta, paths: Paths) -> PoolReport:
    """Runs in a worker process: loads the pool's hourly file once for both checks."""
    rows = load_hourly_rows(paths.hourly, pool.address)
    created = meta[pool.address]["createdAtTimestamp"] if pool.address in meta else None
    return PoolReport(_sanity(pool, created, rows), _spot_check(pool, rows, paths.daily_close_dir))


def _flags(rec: dict) -> list[str]:
    flags = []
    if rec.get("fg0_monotonic") is False or rec.get("fg1_monotonic") is False:
        flags.append("feeGrowth DECREASES")
    if (rec.get("fg_stale_frac") or 0) > STALE_FEE_GROWTH_LIMIT:
        flags.append(f"feeGrowth unchanged in {rec['fg_stale_frac']:.0%} of hours with volume")
    if rec["duplicates_or_unsorted"]:
        flags.append("duplicate/unsorted rows")
    return flags


def _log_sanity(rec: dict) -> None:
    if not rec["n_hours"]:
        return
    log.info("%-18s %6d h  %s -> %s  coverage %.1f%%  max gap %dh  %s",
             rec["label"], rec["n_hours"], rec["first"], rec["last"], 100 * rec["coverage"],
             rec["max_gap_hours"], "; ".join(_flags(rec)) or "OK")


def _log_spot_check(rec: dict) -> None:
    log.info("spot check %-18s %4d days  median |diff| %.2e  max %.2e (%s)  exact(1e-6) %.1f%%",
             rec["label"], rec["days_compared"], rec["median_rel_diff"], rec["max_rel_diff"],
             rec["worst_day"], 100 * rec["frac_within_1e-6"])


def write_reports(pools: list[Pool], meta: PoolMeta, paths: Paths, max_workers: int) -> BatchResult[PoolReport]:
    """Analyse pools in parallel processes, then log and write both reports in pool order."""
    batch = run_each(pools, partial(analyse_pool, meta=meta, paths=paths),
                     name=lambda p: p.label, max_workers=max_workers, use_processes=True)
    reports = [batch.results[p.label] for p in pools if p.label in batch.results]

    for report in reports:
        _log_sanity(report.sanity)
    write_csv(paths.reports / "sanity_report.csv", [r.sanity for r in reports],
              first_cols=("pool", "label", "n_hours"))
    log.info("sanity report -> %s", paths.reports / "sanity_report.csv")

    spot_checks = [r.spot_check for r in reports if r.spot_check]
    for rec in spot_checks:
        _log_spot_check(rec)
    if spot_checks:
        write_csv(paths.reports / "spot_check_daily_close.csv", spot_checks, first_cols=("pool", "label"))
    else:
        log.info("spot check: no pool has both hourly and daily close data yet")
    return batch
