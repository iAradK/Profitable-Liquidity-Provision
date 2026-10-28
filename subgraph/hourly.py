"""Dataset 1: poolHourDatas per pool, resumed from the last saved hour."""

import logging
from pathlib import Path

from .client import GraphClient
from .config import DownloadConfig, Pool
from .schema import Schema
from .storage import iso, load_json, save_json, write_csv

log = logging.getLogger("subgraph")

HOUR = 3600
HOURLY_FIELDS = [
    "periodStartUnix", "open", "high", "low", "close", "token0Price", "token1Price",
    "sqrtPrice", "tick", "liquidity", "feeGrowthGlobal0X128", "feeGrowthGlobal1X128",
    "volumeToken0", "volumeToken1", "volumeUSD", "feesUSD", "tvlUSD", "txCount",
]


def hourly_fields(schema: Schema) -> list[str]:
    return schema.pick("PoolHourData", HOURLY_FIELDS)


def load_hourly_rows(hourly_dir: Path, address: str) -> list[dict]:
    saved = load_json(hourly_dir / f"{address}.json", {})
    return saved.get("data", {}).get("poolHourDatas", [])


def _save(path: Path, rows: list[dict]) -> None:
    save_json(path, {"data": {"poolHourDatas": rows}})


def download_hourly(pool: Pool, *, client: GraphClient, created_ts: int, fields: list[str],
                    settings: DownloadConfig, out_dir: Path) -> bool:
    """Fetch the hours after the last saved one. Returns True if new rows were added."""
    path = out_dir / f"{pool.address}.json"
    csv_path = path.with_suffix(".csv")
    rows = [] if settings.force_refresh else load_hourly_rows(out_dir, pool.address)
    start = max(settings.start_ts, created_ts // HOUR * HOUR)
    cursor = rows[-1]["periodStartUnix"] if rows else start - 1
    total_hours = max(1, (settings.end_ts - start) // HOUR)
    if rows:
        log.info("[%s] resuming from %s (%d rows saved)", pool.label, iso(cursor), len(rows))

    pages = new_rows = 0
    while True:
        query = f"""{{ poolHourDatas(first: {settings.page_size}, orderBy: periodStartUnix,
            orderDirection: asc, where: {{ pool: "{pool.address}", periodStartUnix_gt: {cursor},
            periodStartUnix_lt: {settings.end_ts} }}) {{ {" ".join(fields)} }} }}"""
        page = client.query(query)["poolHourDatas"]
        if not page:
            break
        if page[0]["periodStartUnix"] <= cursor:
            raise RuntimeError(f"[{pool.label}] cursor did not advance at {cursor}")
        rows.extend(page)
        new_rows += len(page)
        cursor = page[-1]["periodStartUnix"]
        pages += 1
        log.info("[%s] %d rows, up to %s (%.0f%% of period)", pool.label, len(rows), iso(cursor),
                 100 * min(1, (cursor - start) / HOUR / total_hours))
        if pages % settings.checkpoint_every == 0:
            _save(path, rows)
        if len(page) < settings.page_size:
            break

    if new_rows == 0 and csv_path.exists():
        log.info("[%s] hourly up to date: %d rows", pool.label, len(rows))
        return False
    _save(path, rows)
    csv_rows = [{**r, "datetime_utc": iso(r["periodStartUnix"])} for r in rows]
    write_csv(csv_path, csv_rows, first_cols=("periodStartUnix", "datetime_utc"))
    log.info("[%s] hourly done: %d rows (+%d)", pool.label, len(rows), new_rows)
    return True
