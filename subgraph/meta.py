"""Dataset 2: pool metadata (tokens, decimals, fee tier, creation time)."""

import logging
from pathlib import Path

from .client import GraphClient
from .config import Pool
from .storage import load_json, save_json, write_csv

log = logging.getLogger("subgraph")

PoolMeta = dict[str, dict]  # pool address -> metadata row


def load_saved_meta(meta_dir: Path) -> PoolMeta:
    saved = load_json(meta_dir / "pools.json", {})
    return {r["id"]: r for r in saved.get("data", {}).get("pools", [])}


def download_meta(client: GraphClient, pools: list[Pool], universe: tuple[Pool, ...],
                  meta_dir: Path, page_size: int) -> PoolMeta:
    """Fetch metadata for `pools` and merge it into the saved file (ordered like `universe`)."""
    ids = ", ".join(f'"{p.address}"' for p in pools)
    query = f"""{{ pools(first: {page_size}, where: {{ id_in: [{ids}] }}) {{
        id feeTier createdAtTimestamp createdAtBlockNumber
        token0 {{ id symbol name decimals }} token1 {{ id symbol name decimals }} }} }}"""
    rows = client.query(query)["pools"]

    labels = {p.address: p.label for p in universe}
    found = {r["id"] for r in rows}
    for pool in pools:
        if pool.address not in found:
            log.error("pool %s (%s) not found in the subgraph", pool.address, pool.label)
    for row in rows:
        row["label"] = labels.get(row["id"], "")

    meta = load_saved_meta(meta_dir)
    meta.update({r["id"]: r for r in rows})
    order = {p.address: i for i, p in enumerate(universe)}
    merged = sorted(meta.values(), key=lambda r: order.get(r["id"], len(order)))
    save_json(meta_dir / "pools.json", {"data": {"pools": merged}})
    write_csv(meta_dir / "pools.csv", merged, first_cols=("id", "label", "feeTier"))
    log.info("pool metadata: %d/%d pools", len(rows), len(pools))
    return {r["id"]: r for r in merged}


def load_or_download_meta(client: GraphClient, pools: list[Pool], universe: tuple[Pool, ...],
                          meta_dir: Path, page_size: int) -> PoolMeta:
    """Saved metadata if it covers every pool, otherwise download it."""
    meta = load_saved_meta(meta_dir)
    if all(p.address in meta for p in pools):
        return meta
    return download_meta(client, pools, universe, meta_dir, page_size)
