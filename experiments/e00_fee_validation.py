"""E0. Fee accounting validation: do fees computed from feeGrowthGlobal match what real LPs earned?

Ours:   liquidity x change of feeGrowthGlobal on the hourly grid (interpolated linearly within the hour).
Truth:  the contract's own accounting for real full-range positions: liquidity x change of the position's
        feeGrowthInsideLast between two consecutive snapshots (mint / increase / decrease / collect).
For a full-range position feeGrowthInside = feeGrowthGlobal - a constant, so the two must agree up to the
intra-hour interpolation; any larger gap is an error in the hourly data or in our fee formula.

The subgraph's collectedFeesToken0/1 fields are not used: collectedFeesToken1 equals collectedFeesToken0 in
almost every position (a subgraph bug), so they are not a usable ground truth.
Snapshot pairs where the subgraph did not update feeGrowthInsideLast (zero change while the pool earned
fees) are counted as 'stale' and excluded from the error statistics.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.context import Context
from experiments.data import HOUR, Q128, PoolData
from experiments.output import cached_map, save_table
from experiments.plotting import CATEGORY_ORDER, INK_SECONDARY, apply_style, category_color, plt, save_figure

log = logging.getLogger("experiments")

FULL_RANGE_TICK = {100: 887272, 500: 887270, 3000: 887220, 10000: 887200}
U256 = 2**256
INSIDE_FIELDS = ("feeGrowthInside0LastX128", "feeGrowthInside1LastX128")


@dataclass(frozen=True)
class PoolPositions:
    pool: PoolData
    positions_csv: Path
    snapshots_csv: Path
    min_interval_hours: float


def snapshot_intervals(payload: PoolPositions) -> pd.DataFrame:
    """Consecutive snapshot pairs of full-range positions with the on-chain fees earned in between (raw units)."""
    meta = payload.pool.meta
    positions = pd.read_csv(payload.positions_csv, dtype=str)
    tick = FULL_RANGE_TICK[meta.fee_tier]
    full_range = positions.loc[(positions["tickLower_tickIdx"].astype(int) == -tick)
                               & (positions["tickUpper_tickIdx"].astype(int) == tick), "id"]
    snaps = pd.read_csv(payload.snapshots_csv, dtype=str)
    snaps = snaps[snaps["position_id"].isin(set(full_range))].copy()
    snaps["timestamp"] = snaps["timestamp"].astype(np.int64)
    snaps["block"] = snaps["blockNumber"].astype(np.int64)
    snaps = snaps.sort_values(["position_id", "block"])

    rows = []
    for position_id, group in snaps.groupby("position_id", sort=False):
        records = group.to_dict("records")
        for before, after in zip(records[:-1], records[1:]):
            liquidity = int(before["liquidity"])
            hours = (after["timestamp"] - before["timestamp"]) / HOUR
            if liquidity == 0 or hours < payload.min_interval_hours:
                continue
            earned = [liquidity * ((int(after[f]) - int(before[f])) % U256) / Q128 for f in INSIDE_FIELDS]
            rows.append({"position_id": position_id, "start_ts": int(before["timestamp"]),
                         "end_ts": int(after["timestamp"]), "hours": hours, "liquidity": float(liquidity),
                         "onchain0": earned[0], "onchain1": earned[1]})
    return pd.DataFrame(rows)


def compare_pool(payload: PoolPositions) -> list[dict]:
    meta, grid = payload.pool.meta, payload.pool.grid
    intervals = snapshot_intervals(payload)
    if intervals.empty:
        return []
    start0, start1 = grid.fee_growth_at(intervals["start_ts"].to_numpy())
    end0, end1 = grid.fee_growth_at(intervals["end_ts"].to_numpy())
    predicted0 = intervals["liquidity"] * (end0 - start0)
    predicted1 = intervals["liquidity"] * (end1 - start1)
    raw_price = grid.sqrt_price[grid.index_at(intervals["end_ts"].to_numpy())] ** 2
    to_token1 = 10.0 ** meta.decimals1
    out = intervals.assign(
        pool=meta.address, label=meta.label, category=meta.category,
        onchain_token1=(intervals["onchain0"] * raw_price + intervals["onchain1"]) / to_token1,
        predicted_token1=(predicted0 * raw_price + predicted1) / to_token1,
    )
    out["stale"] = (out["onchain0"] == 0) & (out["onchain1"] == 0) & (out["predicted_token1"] > 0)
    out["rel_error"] = out["predicted_token1"] / out["onchain_token1"] - 1.0
    return json.loads(out.drop(columns=["onchain0", "onchain1"]).to_json(orient="records"))


def summarize(group: pd.DataFrame, tolerance: float) -> pd.Series:
    ok = group[~group["stale"] & (group["onchain_token1"] > 0)]
    abs_error = ok["rel_error"].abs()
    return pd.Series({
        "n_intervals": len(group),
        "n_positions": group["position_id"].nunique(),
        "share_stale": group["stale"].mean(),
        "n_compared": len(ok),
        "median_hours": ok["hours"].median(),
        "median_abs_rel_error": abs_error.median(),
        "p90_abs_rel_error": abs_error.quantile(0.9),
        "aggregate_rel_error": ok["predicted_token1"].sum() / ok["onchain_token1"].sum() - 1.0,
        f"share_within_{tolerance:g}": (abs_error <= tolerance).mean(),
    })


def plot_scatter(intervals: pd.DataFrame, path_stem: Path) -> None:
    apply_style()
    ok = intervals[~intervals["stale"] & (intervals["onchain_token1"] > 0) & (intervals["predicted_token1"] > 0)]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for category in CATEGORY_ORDER:
        part = ok[ok["category"] == category]
        if len(part):
            ax.scatter(part["onchain_token1"], part["predicted_token1"], s=8, color=category_color(category),
                       label=category, edgecolors="white", linewidths=0.3)
    low, high = ok[["onchain_token1", "predicted_token1"]].min().min(), ok[["onchain_token1", "predicted_token1"]].max().max()
    ax.plot([low, high], [low, high], color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlabel="Fees earned on-chain (token1)", ylabel="Fees from hourly feeGrowthGlobal (token1)")
    ax.legend(loc="upper left")
    save_figure(fig, path_stem)


def run(ctx: Context, out_dir: Path) -> bool:
    cfg = ctx.config
    positions_dir = cfg.data.data_dir / "positions"
    items = []
    for pool in ctx.market.pools:
        positions_csv = positions_dir / f"{pool.meta.address}.csv"
        snapshots_csv = positions_dir / f"{pool.meta.address}_snapshots.csv"
        if positions_csv.exists() and snapshots_csv.exists():
            items.append((pool.meta.address, PoolPositions(pool, positions_csv, snapshots_csv,
                                                           cfg.fee_validation.min_interval_hours)))
        else:
            log.warning("E0: no positions for %s", pool.meta.label)
    results = cached_map(items, compare_pool, out_dir / "pools", cfg.max_workers, cfg.force_refresh)
    intervals = pd.DataFrame([row for rows in results.values() for row in rows])
    if intervals.empty:
        raise RuntimeError("no full-range snapshot intervals to compare")
    save_table(intervals, out_dir / "intervals.csv")

    tolerance = cfg.fee_validation.tolerance
    by_pool = intervals.groupby(["label", "category"]).apply(summarize, tolerance=tolerance, include_groups=False)
    overall = summarize(intervals, tolerance).to_frame().T.assign(label="ALL", category="")
    save_table(pd.concat([by_pool.reset_index(), overall]), out_dir / "fee_validation_summary.csv")
    plot_scatter(intervals, out_dir / "fee_validation_scatter")
    return True
