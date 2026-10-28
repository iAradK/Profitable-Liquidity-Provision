"""Dataset 3: real full-range LP positions per pool, plus their snapshots (liquidity changes)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from .client import GraphClient
from .config import DownloadConfig, Pool
from .schema import Schema, SchemaError
from .storage import load_json, save_json, write_csv

log = logging.getLogger("subgraph")

# Full-range ticks per fee tier (MIN_TICK/MAX_TICK rounded to the tick spacing)
FULL_RANGE_TICK = {100: 887272, 500: 887270, 3000: 887220, 10000: 887200}

POSITION_FIELDS = [
    "id", "owner", "liquidity", "depositedToken0", "depositedToken1",
    "withdrawnToken0", "withdrawnToken1", "collectedToken0", "collectedToken1",
    "collectedFeesToken0", "collectedFeesToken1", "amountCollectedUSD",
    "feeGrowthInside0LastX128", "feeGrowthInside1LastX128",
]
SNAPSHOT_FIELDS = [
    "id", "timestamp", "blockNumber", "liquidity", "depositedToken0", "depositedToken1",
    "withdrawnToken0", "withdrawnToken1", "collectedFeesToken0", "collectedFeesToken1",
    "feeGrowthInside0LastX128", "feeGrowthInside1LastX128",
]


@dataclass(frozen=True)
class PositionQuery:
    """Selection sets for this schema version (built once, shared by all pools)."""
    selection: str
    tick_is_object: bool
    snapshot_selection: str | None  # None when the subgraph has no PositionSnapshot entity


def build_position_query(schema: Schema) -> PositionQuery:
    if not schema.has_type("Position"):
        raise SchemaError("no Position entity in this subgraph")
    fields = schema.pick("Position", POSITION_FIELDS)
    tick_is_object = schema.is_object("Position", "tickLower")
    fields += ["tickLower { tickIdx }", "tickUpper { tickIdx }"] if tick_is_object else ["tickLower", "tickUpper"]
    if schema.is_object("Position", "transaction"):
        tx_fields = schema.pick("Transaction", ["id", "timestamp", "blockNumber"])
        fields.append("transaction { " + " ".join(tx_fields) + " }")
    if schema.is_object("Position", "owner"):  # some versions make owner an entity
        fields = ["owner { id }" if f == "owner" else f for f in fields]

    snapshot_selection = None
    if schema.has_type("PositionSnapshot"):
        snapshot_selection = " ".join(schema.pick("PositionSnapshot", SNAPSHOT_FIELDS) + ["position { id }"])
    return PositionQuery(" ".join(fields), tick_is_object, snapshot_selection)


class _SavedPositions:
    """The pool's positions file: positions, snapshots and whether the position list is complete."""

    def __init__(self, path: Path, force_refresh: bool):
        self.path = path
        saved = {} if force_refresh else load_json(path, {})
        data = saved.get("data", {})
        self.positions: list[dict] = data.get("positions", [])
        self.snapshots: list[dict] = data.get("positionSnapshots", [])
        self.complete: bool = saved.get("complete", False)

    def save(self) -> None:
        save_json(self.path, {"complete": self.complete,
                              "data": {"positions": self.positions, "positionSnapshots": self.snapshots}})


def _tick_filter(pool: Pool, fee_tier: int, tick_is_object: bool) -> str:
    tick = FULL_RANGE_TICK[fee_tier]
    if tick_is_object:
        return f'tickLower: "{pool.address}#{-tick}", tickUpper: "{pool.address}#{tick}"'
    return f"tickLower: {-tick}, tickUpper: {tick}"


def _fetch_positions(pool: Pool, saved: _SavedPositions, client: GraphClient, tick_filter: str,
                     selection: str, settings: DownloadConfig) -> int:
    fetched = 0
    cursor = saved.positions[-1]["id"] if saved.positions else ""
    while not saved.complete and len(saved.positions) < settings.max_positions:
        first = min(settings.page_size, settings.max_positions - len(saved.positions))
        query = f"""{{ positions(first: {first}, orderBy: id, orderDirection: asc,
            where: {{ pool: "{pool.address}", {tick_filter}, id_gt: "{cursor}" }}) {{ {selection} }} }}"""
        page = client.query(query)["positions"]
        saved.positions.extend(page)
        fetched += len(page)
        if page:
            cursor = page[-1]["id"]
        saved.complete = len(page) < settings.page_size
        saved.save()
        log.info("[%s] %d full-range positions", pool.label, len(saved.positions))
    return fetched


def _fetch_snapshots(pool: Pool, saved: _SavedPositions, client: GraphClient, selection: str,
                     settings: DownloadConfig) -> int:
    """Snapshots give the timestamps of every liquidity change -> holding period and fees over time."""
    have = {s["position"]["id"] for s in saved.snapshots}
    todo = [p["id"] for p in saved.positions if p["id"] not in have]
    fetched = 0
    for i in range(0, len(todo), settings.snapshot_chunk):
        ids = ", ".join(f'"{x}"' for x in todo[i:i + settings.snapshot_chunk])
        cursor = ""
        while True:
            query = f"""{{ positionSnapshots(first: {settings.page_size}, orderBy: id, orderDirection: asc,
                where: {{ position_in: [{ids}], id_gt: "{cursor}" }}) {{ {selection} }} }}"""
            page = client.query(query)["positionSnapshots"]
            saved.snapshots.extend(page)
            fetched += len(page)
            if len(page) < settings.page_size:
                break
            cursor = page[-1]["id"]
        saved.save()
        log.info("[%s] snapshots for %d/%d positions", pool.label,
                 min(i + settings.snapshot_chunk, len(todo)), len(todo))
    return fetched


def _write_csvs(pool: Pool, saved: _SavedPositions, out_dir: Path) -> None:
    snapshot_span: dict[str, tuple[int, int]] = {}
    for snap in saved.snapshots:
        pid, ts = snap["position"]["id"], int(snap["timestamp"])
        first, last = snapshot_span.get(pid, (ts, ts))
        snapshot_span[pid] = (min(first, ts), max(last, ts))
    rows = []
    for position in saved.positions:
        row = dict(position)
        if position["id"] in snapshot_span:
            row["firstSnapshotTs"], row["lastSnapshotTs"] = snapshot_span[position["id"]]
        rows.append(row)
    write_csv(out_dir / f"{pool.address}.csv", rows, first_cols=("id",))
    if saved.snapshots:
        write_csv(out_dir / f"{pool.address}_snapshots.csv", saved.snapshots,
                  first_cols=("position_id", "timestamp"))


def download_positions(pool: Pool, *, client: GraphClient, fee_tier: int, query: PositionQuery,
                       settings: DownloadConfig, out_dir: Path) -> bool:
    """Fetch missing positions and snapshots. Returns True if anything new was added."""
    saved = _SavedPositions(out_dir / f"{pool.address}.json", settings.force_refresh)
    tick_filter = _tick_filter(pool, fee_tier, query.tick_is_object)
    fetched = _fetch_positions(pool, saved, client, tick_filter, query.selection, settings)
    if query.snapshot_selection:
        fetched += _fetch_snapshots(pool, saved, client, query.snapshot_selection, settings)

    csv_missing = saved.positions and not (out_dir / f"{pool.address}.csv").exists()
    if fetched == 0 and not csv_missing:
        log.info("[%s] positions up to date: %d positions, %d snapshots",
                 pool.label, len(saved.positions), len(saved.snapshots))
        return False
    if saved.positions:
        _write_csvs(pool, saved, out_dir)
    log.info("[%s] positions done: %d positions, %d snapshots",
             pool.label, len(saved.positions), len(saved.snapshots))
    return True
