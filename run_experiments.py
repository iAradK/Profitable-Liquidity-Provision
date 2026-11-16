"""
Run the ICBC 2027 experiments (E0-E14) on the data saved by download_subgraph.py and the BigQuery exports.

Usage:
  1. Choose the experiments and settings in config.toml ([experiments] and its sub-sections).
  2. python run_experiments.py

Each experiment writes to results/<id>/ and marks itself finished there; the next run skips finished
experiments, and long experiments also resume from their saved per-pool / per-cell results. Hourly data is
parsed once and cached in results/cache/ (rebuilt automatically when the CSV changes). A failing experiment
is logged and the others still run. Requires numpy, pandas, scipy, matplotlib, pyarrow and statsmodels (E3).

  E0  fee accounting validation against real full-range positions
  E1  IL and net-yield profile with bootstrap confidence intervals
  E2  LVR decomposition (correlation vs volatility, Gaussian vs tail)
  E3  persistence: rolling AR(1), ADF/KPSS, AR(2), regime split, variance ratio
  E4  forecast calibration: Wilson intervals, Kupiec and Christoffersen tests
  E5  rolling out-of-sample pool selection vs baselines (main experiment)
  E6  sensitivity grid (tail level, horizon, calibration window, k)
  E7  horizon matched to the half-life vs fixed horizons
  E8  concentrated ranges, with swap-level validation
  E9  fee tiers vs the 6x turnover rule
  E10 comparison with RV / FARV
  E11 replication on another network
  E12 fee crediting rule and position size, with a swap-by-swap replay of the top months
  E13 correlation-aware portfolios: choosing the k pools jointly with the token correlation model
  E14 out-of-sample value of the token correlation in a pool score
"""

import importlib
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from experiments.config import ConfigError, ExperimentConfig, Horizon, load_config
from experiments.context import Context
from experiments.data import MarketData, load_market
from experiments.outcomes import sample_periods
from experiments.output import is_done, mark_done, save_table
from experiments.swaps import attach_replay, decode_swaps, months_table, replay_query, thin_liquidity_months

log = logging.getLogger("experiments")

MODULES = {
    "E0": "e00_fee_validation", "E1": "e01_risk_profile", "E2": "e02_lvr_decomposition", "E3": "e03_persistence",
    "E4": "e04_calibration", "E5": "e05_selection", "E6": "e06_sensitivity", "E7": "e07_horizon_matching",
    "E8": "e08_concentrated", "E9": "e09_fee_tiers", "E10": "e10_rv_farv", "E11": "e11_replication",
    "E12": "e12_crediting", "E13": "e13_portfolio", "E14": "e14_rho_score",
}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(output_dir / "experiments.log", encoding="utf-8")])


def run_one(experiment: str, ctx: Context, out_dir: Path) -> bool:
    """Imported lazily, so a missing optional dependency only fails the experiment that needs it."""
    module = importlib.import_module(f"experiments.{MODULES[experiment]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return module.run(ctx, out_dir)


def prepare_replay(market: MarketData, config: ExperimentConfig) -> MarketData:
    """Attach the swaps of the thin-liquidity pool-months, so their fees are computed swap by swap. Months the swap
    exports do not cover keep the hourly fees; the query that exports them is written to the output folder."""
    fees = config.fees
    starts, ends = sample_periods(config.sample, market.data_end_ts, Horizon(months=1))
    months = thin_liquidity_months(market, starts, ends, fees.replay_implied_ratio, fees.replay_min_fee_yield)
    missing = months
    if config.data.replay_swaps_files and months:
        swaps = decode_swaps(config.data.replay_swaps_files, config.cache_dir / "replay_swaps.parquet")
        market, missing = attach_replay(market, swaps, months)
    save_table(months_table(months, missing), config.output_dir / "thin_liquidity_months.csv")
    query_path = config.output_dir / "replay_query.sql"
    if missing:
        query_path.write_text(replay_query(missing), encoding="utf-8")
        log.warning("%d of %d thin-liquidity pool-months have no swap export and keep the hourly fees (an upper "
                    "bound); run %s in BigQuery and add the CSV to replay_swaps_files", len(missing), len(months),
                    query_path)
    else:
        query_path.unlink(missing_ok=True)
    log.info("thin-liquidity pool-months computed swap by swap: %d", len(months) - len(missing))
    return replace(market, thin_months=tuple((m.pool, m.start_ts) for m in months))


def run(config: ExperimentConfig) -> int:
    pending = [e for e in config.run if config.force_refresh or not is_done(config.output_dir / e)]
    finished_before = [e for e in config.run if e not in pending]
    if finished_before:
        log.info("already finished (delete the folder or set force_refresh to redo): %s", ", ".join(finished_before))
    if not pending:
        return 0

    market = load_market(config.data.data_dir, config.data.gas_file, config.data.eth_token, config.pools,
                         label_categories={}, cache_dir=config.cache_dir / "grids", max_workers=config.max_workers,
                         force_refresh=config.force_refresh, crediting=config.fees.main)
    market = prepare_replay(market, config)
    ctx = Context(config, market)
    done, skipped, failed = [], [], {}
    for experiment in pending:
        out_dir = config.output_dir / experiment
        started = time.time()
        log.info("%s: started", experiment)
        try:
            if run_one(experiment, ctx, out_dir):
                mark_done(out_dir)
                done.append(experiment)
                log.info("%s: finished in %.1f min -> %s", experiment, (time.time() - started) / 60, out_dir)
            else:
                skipped.append(experiment)
        except Exception as exc:
            log.exception("%s failed", experiment)
            failed[experiment] = f"{type(exc).__name__}: {exc}"

    log.info("summary: %d finished, %d skipped, %d already done, %d failed",
             len(done), len(skipped), len(finished_before), len(failed))
    for experiment, error in failed.items():
        log.info("  failed %s: %s (re-run to retry)", experiment, error)
    return len(failed)


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}")
    setup_logging(config.output_dir)
    sys.exit(1 if run(config) else 0)


if __name__ == "__main__":
    main()
