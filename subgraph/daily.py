"""Dataset 1b: poolDayDatas per pool (daily volume and fees), resumed from the last saved day.

The hourly entity's volumeUSD/feesUSD are zero before 2024 in this subgraph, so the daily fees are the only
record of what each pool paid. The experiments use them to cap a modelled position's fees.
"""

import logging
from pathlib import Path

from .client import GraphClient
from .config import DownloadConfig, Pool
from .schema import Schema
from .storage import iso, load_json, save_json, write_csv

log = logging.getLogger("subgraph")

DAY = 86400
DAILY_FIELDS = ["date", "volumeToken0", "volumeToken1", "volumeUSD", "feesUSD", "tvlUSD", "txCount"]


def daily_fields(schema: Schema) -> list[str]:
    return schema.pick("PoolDayData", DAILY_FIELDS)


def _save(path: Path, rows: list[dict]) -> None:
    save_json(path, {"data": {"poolDayDatas": rows}})


def download_daily(pool: Pool, *, client: GraphClient, created_ts: int, fields: list[str],
                   settings: DownloadConfig, out_dir: Path) -> bool:
    """Fetch the days after the last saved one. Returns True if new rows were added."""
    path = out_dir / f"{pool.address}.json"
    csv_path = path.with_suffix(".csv")
    saved = {} if settings.force_refresh else load_json(path, {})
    rows = saved.get("data", {}).get("poolDayDatas", [])
    start = max(settings.start_ts, created_ts // DAY * DAY)
    cursor = rows[-1]["date"] if rows else start - 1

    new_rows = 0
    while True:
        query = f"""{{ poolDayDatas(first: {settings.page_size}, orderBy: date, orderDirection: asc,
            where: {{ pool: "{pool.address}", date_gt: {cursor}, date_lt: {settings.end_ts} }}) {{
            {" ".join(fields)} }} }}"""
        page = client.query(query)["poolDayDatas"]
        if not page:
            break
        if page[0]["date"] <= cursor:
            raise RuntimeError(f"[{pool.label}] cursor did not advance at {cursor}")
        rows.extend(page)
        new_rows += len(page)
        cursor = page[-1]["date"]
        if len(page) < settings.page_size:
            break

    if new_rows == 0 and csv_path.exists():
        log.info("[%s] daily up to date: %d days", pool.label, len(rows))
        return False
    _save(path, rows)
    write_csv(csv_path, [{**r, "date_utc": iso(r["date"])[:10]} for r in rows], first_cols=("date", "date_utc"))
    zero = sum(float(r.get("volumeUSD", 0)) == 0 for r in rows)
    log.info("[%s] daily done: %d days (+%d), %d with zero volumeUSD", pool.label, len(rows), new_rows, zero)
    return True
