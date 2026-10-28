"""Dataset 4: tokenDayDatas (daily USD prices) for every token in the selected pools."""

import logging
from dataclasses import dataclass
from pathlib import Path

from .client import GraphClient
from .config import DownloadConfig
from .meta import PoolMeta
from .schema import Schema
from .storage import load_json, save_json, write_csv

log = logging.getLogger("subgraph")

TOKEN_DAY_FIELDS = ["date", "priceUSD", "open", "high", "low", "close"]


@dataclass(frozen=True)
class Token:
    address: str
    symbol: str


def token_day_fields(schema: Schema) -> list[str]:
    return schema.pick("TokenDayData", TOKEN_DAY_FIELDS)


def tokens_in(meta: PoolMeta) -> list[Token]:
    tokens = {t["id"]: t["symbol"] for pool in meta.values() for t in (pool["token0"], pool["token1"])}
    return sorted((Token(a, s) for a, s in tokens.items()), key=lambda t: t.symbol)


def download_token_prices(token: Token, *, client: GraphClient, fields: list[str],
                          settings: DownloadConfig, out_dir: Path) -> bool:
    """Fetch the days after the last saved one. Returns True if new rows were added."""
    path = out_dir / f"{token.address}.json"
    csv_path = path.with_suffix(".csv")
    saved = {} if settings.force_refresh else load_json(path, {})
    rows = saved.get("data", {}).get("tokenDayDatas", [])
    cursor = rows[-1]["date"] if rows else settings.start_ts - 1

    new_rows = 0
    while True:
        query = f"""{{ tokenDayDatas(first: {settings.page_size}, orderBy: date, orderDirection: asc,
            where: {{ token: "{token.address}", date_gt: {cursor}, date_lt: {settings.end_ts} }}) {{
            {" ".join(fields)} }} }}"""
        page = client.query(query)["tokenDayDatas"]
        rows.extend(page)
        new_rows += len(page)
        if page:
            cursor = page[-1]["date"]
        if len(page) < settings.page_size:
            break

    if new_rows == 0 and csv_path.exists():
        log.info("token %s (%s): up to date, %d days", token.symbol, token.address, len(rows))
        return False
    save_json(path, {"symbol": token.symbol, "data": {"tokenDayDatas": rows}})
    write_csv(csv_path, rows, first_cols=("date",))
    log.info("token %s (%s): %d days (+%d)", token.symbol, token.address, len(rows), new_rows)
    return True
