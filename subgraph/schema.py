"""Subgraph schema introspection, so queries only ask for fields the deployed schema has."""

import logging
import time

from .client import GraphClient
from .config import Config
from .storage import iso, load_json, save_json

log = logging.getLogger("subgraph")

INTROSPECT_TYPES = ("PoolHourData", "PoolDayData", "Pool", "Position", "PositionSnapshot",
                    "Tick", "Token", "TokenDayData", "Transaction")

# Mainnet sanity check: USDC/WETH 0.05% must exist with mainnet USDC as token0
MAINNET_CHECK_POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
MAINNET_USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


class SchemaError(Exception):
    pass


class Schema:
    def __init__(self, types: dict[str, dict[str, dict]]):
        self.types = types  # {type: {field: {"kind": .., "name": ..}}}

    def has_type(self, type_name: str) -> bool:
        return type_name in self.types

    def pick(self, type_name: str, wanted: list[str]) -> list[str]:
        """The wanted fields that exist on the type; warns about the missing ones."""
        if type_name not in self.types:
            raise SchemaError(f"type {type_name} is not in the subgraph schema")
        fields = self.types[type_name]
        missing = [f for f in wanted if f not in fields]
        if missing:
            log.warning("%s: fields not in schema, skipped: %s", type_name, ", ".join(missing))
        return [f for f in wanted if f in fields]

    def is_object(self, type_name: str, field: str) -> bool:
        return self.types.get(type_name, {}).get(field, {}).get("kind") == "OBJECT"


def _unwrap(type_ref: dict) -> dict:
    while type_ref.get("ofType"):
        type_ref = type_ref["ofType"]
    return {"kind": type_ref["kind"], "name": type_ref["name"]}


def _fetch_types(client: GraphClient) -> dict[str, dict[str, dict]]:
    types = {}
    for name in INTROSPECT_TYPES:
        query = f"""{{ __type(name: "{name}") {{ name fields {{ name type {{ kind name
                ofType {{ kind name ofType {{ kind name ofType {{ kind name }} }} }} }} }} }} }}"""
        found = client.query(query)["__type"]
        if found is None:
            log.warning("type %s not found in schema", name)
            continue
        types[name] = {f["name"]: _unwrap(f["type"]) for f in found["fields"]}
    return types


def introspect(client: GraphClient, config: Config) -> Schema:
    """Fetch and cache the schema, and check that this is the Ethereum mainnet Uniswap v3 subgraph."""
    types = _fetch_types(client)
    meta = client.query("{ _meta { deployment hasIndexingErrors block { number timestamp } } }")["_meta"]
    pool = client.query(f"""{{ pool(id: "{MAINNET_CHECK_POOL}") {{ id feeTier
        token0 {{ id symbol }} token1 {{ id symbol }} }} }}""")["pool"]

    is_mainnet = bool(pool) and pool["token0"]["id"].lower() == MAINNET_USDC and str(pool["feeTier"]) == "500"
    block_ts = (meta.get("block") or {}).get("timestamp")
    log.info("subgraph deployment %s, indexed block %s (%s), indexing errors: %s",
             meta.get("deployment"), meta["block"]["number"],
             iso(block_ts) if block_ts else "?", meta.get("hasIndexingErrors"))
    if block_ts and int(block_ts) < config.download.end_ts:
        log.warning("subgraph is indexed only up to %s, before the end date %s",
                    iso(block_ts), iso(config.download.end_ts))
    for name in ("PoolHourData", "Position"):
        if name in types:
            log.info("%s fields: %s", name, ", ".join(sorted(types[name])))

    save_json(config.paths.schema_file, {
        "url": config.subgraph.safe_url, "fetched": iso(time.time()), "meta": meta,
        "mainnet_check": {"ok": is_mainnet, "pool": pool}, "types": types,
    })
    if not is_mainnet:
        raise SchemaError(f"mainnet check failed: pool {MAINNET_CHECK_POOL} -> {pool}. This is probably "
                          "not the Ethereum mainnet Uniswap v3 subgraph; check subgraph.subgraph_id.")
    log.info("mainnet check OK: %s is %s/%s fee %s", MAINNET_CHECK_POOL,
             pool["token0"]["symbol"], pool["token1"]["symbol"], pool["feeTier"])
    return Schema(types)


def load_schema(client: GraphClient, config: Config, refresh: bool) -> Schema:
    """Use the cached schema when it belongs to the same subgraph, else introspect."""
    cached = load_json(config.paths.schema_file)
    if cached and not refresh and cached.get("url") == config.subgraph.safe_url:
        return Schema(cached["types"])
    return introspect(client, config)
