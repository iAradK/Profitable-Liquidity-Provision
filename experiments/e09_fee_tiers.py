"""E9. Fee tiers: does the paper's 6x turnover rule agree with correctly computed fees?

For every token pair traded in two fee tiers, each month:
  - the paper's rule: the low tier wins if its turnover (volume / TVL) is more than fee_high / fee_low times
    the high tier's turnover;
  - the corrected comparison: the low tier wins if its full-range fee yield (from feeGrowthGlobal) is higher.
TVL includes concentrated liquidity, so turnover overstates what a full-range LP earns; this table shows how
often the two disagree. Volume is derived from fee growth (fees / fee rate) because the subgraph's hourly
volumeUSD is empty before March 2024; months where the subgraph has volume give a cross-check.
Routers split order flow across tiers, so the tiers of one pair are not independent pools.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.context import Context
from experiments.data import PoolData, TokenPrices
from experiments.output import save_table


def monthly_volume_usd(pool: PoolData, token1_usd: TokenPrices, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Volume implied by fee growth: fees per unit liquidity x active liquidity / fee rate, in USD."""
    grid, meta = pool.grid, pool.meta
    fees_token1_raw = (np.diff(grid.fee_growth0) * grid.sqrt_price[1:] ** 2 + np.diff(grid.fee_growth1)) * grid.liquidity[:-1]
    hourly_usd = fees_token1_raw / meta.fee_rate / 10.0 ** meta.decimals1 * token1_usd.at(grid.t_end[1:])
    cumulative = np.concatenate([[0.0], np.nancumsum(hourly_usd)])
    i0, i1 = grid.index_at(starts), grid.index_at(ends)
    return np.where(i0 >= 0, cumulative[np.clip(i1, 0, None)] - cumulative[np.clip(i0, 0, None)], np.nan)


def monthly_mean_tvl(pool: PoolData, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    grid = pool.grid
    i0, i1 = grid.index_at(starts), grid.index_at(ends)
    return np.array([np.nanmean(grid.tvl_usd[a:b + 1]) if a >= 0 else np.nan for a, b in zip(i0, i1)])


def subgraph_monthly_volume(csv_path: Path, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    hourly = pd.read_csv(csv_path, usecols=["periodStartUnix", "volumeUSD"])
    ts, volume = hourly["periodStartUnix"].to_numpy(), hourly["volumeUSD"].to_numpy(float)
    out = np.array([volume[(ts >= a) & (ts < b)].sum() for a, b in zip(starts, ends)])
    return np.where(out > 0, out, np.nan)


def pool_month_table(ctx: Context) -> pd.DataFrame:
    panel = ctx.monthly_panel
    rows = []
    for p, pool in enumerate(ctx.market.pools):
        meta = pool.meta
        volume = monthly_volume_usd(pool, ctx.market.token_prices(meta.token1), panel.starts, panel.ends)
        tvl = monthly_mean_tvl(pool, panel.starts, panel.ends)
        reported = subgraph_monthly_volume(ctx.config.data.data_dir / "hourly" / f"{meta.address}.csv",
                                           panel.starts, panel.ends)
        for t in np.flatnonzero(panel.valid[p]):
            rows.append({"family": meta.family, "label": meta.label, "fee_tier": meta.fee_tier,
                         "fee_rate": meta.fee_rate, "period": pd.Timestamp(int(panel.starts[t]), unit="s").strftime("%Y-%m"),
                         "fee_yield": panel.fee[p, t], "net": panel.net[p, t], "volume_usd": volume[t],
                         "subgraph_volume_usd": reported[t], "mean_tvl_usd": tvl[t], "turnover": volume[t] / tvl[t]})
    return pd.DataFrame(rows)


def tier_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in table.groupby("family"):
        tiers = sorted(group["fee_tier"].unique())
        for low, high in itertools.combinations(tiers, 2):
            merged = group[group["fee_tier"] == low].merge(group[group["fee_tier"] == high], on="period",
                                                           suffixes=("_low", "_high"))
            tier_ratio = high / low
            for _, m in merged.iterrows():
                rows.append({
                    "family": family, "low_tier": m["label_low"], "high_tier": m["label_high"], "period": m["period"],
                    "turnover_ratio": m["turnover_low"] / m["turnover_high"], "tier_ratio": tier_ratio,
                    "fee_yield_low": m["fee_yield_low"], "fee_yield_high": m["fee_yield_high"],
                    "rule_says_low_wins": m["turnover_low"] / m["turnover_high"] > tier_ratio,
                    "low_has_higher_fee_yield": m["fee_yield_low"] > m["fee_yield_high"],
                    "low_has_higher_net": m["net_low"] > m["net_high"],
                })
    return pd.DataFrame(rows)


def summarize(comparisons: pd.DataFrame) -> pd.DataFrame:
    grouped = comparisons.groupby(["family", "low_tier", "high_tier"])
    return grouped.apply(lambda g: pd.Series({
        "n_months": len(g),
        "share_rule_says_low_wins": g["rule_says_low_wins"].mean(),
        "share_low_higher_fee_yield": g["low_has_higher_fee_yield"].mean(),
        "share_low_higher_net": g["low_has_higher_net"].mean(),
        "rule_agrees_with_fee_yield": (g["rule_says_low_wins"] == g["low_has_higher_fee_yield"]).mean(),
        "median_fee_yield_low": g["fee_yield_low"].median(),
        "median_fee_yield_high": g["fee_yield_high"].median(),
        "median_turnover_ratio": g["turnover_ratio"].median(),
    }), include_groups=False)


def run(ctx: Context, out_dir: Path) -> bool:
    table = pool_month_table(ctx)
    save_table(table, out_dir / "pool_months.csv")
    check = table.dropna(subset=["subgraph_volume_usd"])
    save_table(check.groupby("label").apply(lambda g: pd.Series({
        "n_months": len(g), "median_derived_over_subgraph": (g["volume_usd"] / g["subgraph_volume_usd"]).median()}),
        include_groups=False), out_dir / "volume_cross_check.csv", index=True)
    comparisons = tier_comparisons(table)
    save_table(comparisons, out_dir / "tier_comparisons.csv")
    save_table(summarize(comparisons), out_dir / "tier_summary.csv", index=True)
    return True
