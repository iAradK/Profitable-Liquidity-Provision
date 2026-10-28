"""E11. Replication of the main experiment (E5) on another network.

Runs the E5 pipeline unchanged on the data in [experiments.replication].data_dir (same layout as pool_data,
downloaded with the same tool pointed at that network's subgraph). Pools are every pool in that folder's
meta/pools.csv; categories are taken from the main universe by label. Skipped (and retried on the next run)
until the folder exists.
"""

import logging
from pathlib import Path

from experiments.context import Context
from experiments.data import load_market
from experiments.e05_selection import run_selection_experiment

log = logging.getLogger("experiments")


def run(ctx: Context, out_dir: Path) -> bool:
    cfg, rep = ctx.config, ctx.config.replication
    if not (rep.data_dir / "meta" / "pools.csv").exists():
        log.warning("E11 skipped: no data in %s yet", rep.data_dir)
        return False
    market = load_market(rep.data_dir, rep.gas_file, rep.eth_token, universe=None,
                         label_categories={p.label: p.category for p in cfg.pools},
                         cache_dir=cfg.cache_dir / f"grids_{rep.data_dir.name}", max_workers=cfg.max_workers,
                         force_refresh=cfg.force_refresh, position_usd=cfg.costs.capital_usd)
    run_selection_experiment(market, cfg, out_dir)
    return True
