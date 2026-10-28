"""
Download Uniswap v3 (Ethereum mainnet) subgraph data for the ICBC 2027 resubmission.

Datasets (see handoff_subgraph_download.md):
  1. hourly    - poolHourDatas for all 27 pools            -> pool_data/hourly/<pool>.json + .csv
     daily     - poolDayDatas (daily volume and fees)       -> pool_data/daily/<pool>.json + .csv
  2. meta      - pool metadata (tokens, decimals, fee)      -> pool_data/meta/pools.json + .csv
  3. positions - real full-range LP positions (+ snapshots) -> pool_data/positions/<pool>.json + .csv
  4. prices    - tokenDayDatas (priceUSD) for every token   -> pool_data/token_prices/<token>.json + .csv
  report       - sanity report per pool + spot check vs the existing daily closes

Usage:
  1. Put your The Graph API key in secrets.toml (copy secrets.example.toml).
  2. Choose the step (and optionally a subset of pools) in config.toml.
  3. python download_subgraph.py

Every download step resumes from the saved data: re-running only fetches what is missing or new.
Pools and tokens are downloaded concurrently; a failing pool is reported and retried on the next run.
feeGrowthGlobal*, liquidity, sqrtPrice etc. are 256-bit integers and are kept as strings
(in both JSON and CSV). Read the CSVs with dtype=str for those columns, then int(...).
"""

import logging
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import requests

from subgraph.client import GraphClient, GraphQLError
from subgraph.config import Config, ConfigError, Pool, load_config
from subgraph.daily import daily_fields, download_daily
from subgraph.hourly import download_hourly, hourly_fields
from subgraph.meta import PoolMeta, download_meta, load_or_download_meta, load_saved_meta
from subgraph.parallel import BatchResult, run_each
from subgraph.positions import build_position_query, download_positions
from subgraph.prices import download_token_prices, token_day_fields, tokens_in
from subgraph.report import write_reports
from subgraph.schema import Schema, SchemaError, introspect, load_schema

log = logging.getLogger("subgraph")

DownloadStep = Callable[[GraphClient, Schema, PoolMeta, list[Pool], Config], BatchResult[bool]]


def setup_logging(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(reports_dir / "download.log", encoding="utf-8")])


def run_hourly(client: GraphClient, schema: Schema, meta: PoolMeta, pools: list[Pool],
               config: Config) -> BatchResult[bool]:
    fields = hourly_fields(schema)

    def task(pool: Pool) -> bool:
        return download_hourly(pool, client=client, created_ts=int(meta[pool.address]["createdAtTimestamp"]),
                               fields=fields, settings=config.download, out_dir=config.paths.hourly)

    return run_each(pools, task, name=lambda p: p.label, max_workers=config.network.max_workers)


def run_daily(client: GraphClient, schema: Schema, meta: PoolMeta, pools: list[Pool],
              config: Config) -> BatchResult[bool]:
    fields = daily_fields(schema)

    def task(pool: Pool) -> bool:
        return download_daily(pool, client=client, created_ts=int(meta[pool.address]["createdAtTimestamp"]),
                              fields=fields, settings=config.download, out_dir=config.paths.daily)

    return run_each(pools, task, name=lambda p: p.label, max_workers=config.network.max_workers)


def run_positions(client: GraphClient, schema: Schema, meta: PoolMeta, pools: list[Pool],
                  config: Config) -> BatchResult[bool]:
    query = build_position_query(schema)

    def task(pool: Pool) -> bool:
        return download_positions(pool, client=client, fee_tier=int(meta[pool.address]["feeTier"]),
                                  query=query, settings=config.download, out_dir=config.paths.positions)

    return run_each(pools, task, name=lambda p: p.label, max_workers=config.network.max_workers)


def run_prices(client: GraphClient, schema: Schema, meta: PoolMeta, pools: list[Pool],
               config: Config) -> BatchResult[bool]:
    task = partial(download_token_prices, client=client, fields=token_day_fields(schema),
                   settings=config.download, out_dir=config.paths.prices)
    tokens = tokens_in({p.address: meta[p.address] for p in pools})
    return run_each(tokens, task, name=lambda t: t.symbol, max_workers=config.network.max_workers)


DOWNLOAD_STEPS: dict[str, DownloadStep] = {
    "hourly": run_hourly,
    "daily": run_daily,
    "positions": run_positions,
    "prices": run_prices,
}


def log_summary(step: str, batch: BatchResult) -> None:
    log.info("%s: %d succeeded, %d failed", step, len(batch.results), len(batch.failures))
    if all(isinstance(result, bool) for result in batch.results.values()):  # download steps
        updated = sum(batch.results.values())
        log.info("  %d with new data, %d already up to date", updated, len(batch.results) - updated)
    for name, error in batch.failures.items():
        log.info("  failed %s: %s (re-run to retry)", name, error)


def run_downloads(config: Config, pools: list[Pool]) -> int:
    """Run the configured download steps. Returns the number of failed items."""
    step = config.run.step
    client = GraphClient(config.subgraph, config.network)
    started = time.time()
    if step == "introspect":
        introspect(client, config)
        return 0

    schema = load_schema(client, config, refresh=config.run.refresh_schema or step == "all")
    meta_args = (client, pools, config.pools, config.paths.meta, config.download.page_size)
    meta = download_meta(*meta_args) if step in ("meta", "all") else load_or_download_meta(*meta_args)
    pools = [p for p in pools if p.address in meta]

    batches = {}
    for name, download in DOWNLOAD_STEPS.items():
        if step in (name, "all"):
            batches[name] = download(client, schema, meta, pools, config)
    if step == "all":
        batches["report"] = write_reports(pools, meta, config.paths, config.report.max_workers)

    for name, batch in batches.items():
        log_summary(name, batch)
    log.info("done: %d requests in %.1f min", client.n_requests, (time.time() - started) / 60)
    return sum(len(b.failures) for b in batches.values())


def run(config: Config) -> int:
    pools = config.selected_pools()
    if config.run.step == "report":
        batch = write_reports(pools, load_saved_meta(config.paths.meta), config.paths, config.report.max_workers)
        log_summary("report", batch)
        return len(batch.failures)
    return run_downloads(config, pools)


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}")
    setup_logging(config.paths.reports)
    try:
        failures = run(config)
    except (SchemaError, GraphQLError, requests.RequestException) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        raise SystemExit(1)
    if failures:
        log.error("%d item(s) failed; re-run to resume them", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
