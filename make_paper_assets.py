"""
Turn the experiment results into the LaTeX assets of the ICBC paper (overleaf/main_icbc2027.tex).

Writes to [paper].output_dir: numbers.tex (\newcommand macros for every number quoted in the text), one
tab_*.tex per table, and copies of the figures. Re-run after re-running the experiments; the paper updates
without editing the .tex.

Usage: set [paper] in config.toml, then `python make_paper_assets.py`.
"""

import logging
import os
import shutil
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paper_figures import cumulative_usd, make_figures

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"

log = logging.getLogger("paper")

CATEGORY_NAMES = {
    "stable-volatile": "Stable--volatile", "stable-stable": "Stable--stable", "correlated": "Correlated",
    "governance": "Governance", "high-vol": "High-vol./declining",
}
STRATEGY_NAMES = {
    "fee_minus_lvr": r"Fee $-\,\sigma^2/8$", "net_tail": "Net90", "median_fee": "Highest median fee", "fee_low_variance": r"Median fee, low $\sigma^2$",
    "low_variance": r"Lowest $\sigma^2$", "mean_net": "Mean net", "equal_weight": "Equal weight",
    "rv": "RV~\\cite{tangri2023generalizing}", "farv": "FARV~\\cite{tangri2023generalizing}",
}
@dataclass(frozen=True)
class PaperConfig:
    results_dir: Path
    data_dir: Path
    output_dir: Path
    risk_aversion: tuple[float, ...]
    top_k: tuple[int, ...]
    sensitivity_k: int
    main_k: int
    abstract_lambda: float
    table_risk_aversion: tuple[float, ...]
    table_top_k: tuple[int, ...]
    tail_quantile: float
    confidence: float
    capital_usd: float
    paper_dir: Path
    main_tex: str
    upload_zip: Path


def load_config(path: Path = CONFIG_PATH) -> PaperConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    paper, experiments = raw["paper"], raw["experiments"]
    risk_aversion = tuple(float(x) for x in experiments["selection"]["risk_aversion"])
    top_k = tuple(experiments["selection"]["top_k"])
    table_risk_aversion = tuple(float(x) for x in paper["table_risk_aversion"])
    table_top_k = tuple(paper["table_top_k"])
    if not set(table_risk_aversion) <= set(risk_aversion) or not set(table_top_k) <= set(top_k):
        raise ValueError("[paper] table_risk_aversion / table_top_k must be subsets of the [experiments.selection] grids")
    return PaperConfig(results_dir=ROOT / paper["results_dir"], data_dir=ROOT / experiments["data"]["data_dir"],
                       output_dir=ROOT / paper["output_dir"], risk_aversion=risk_aversion, top_k=top_k,
                       sensitivity_k=paper["sensitivity_k"], main_k=paper["main_k"],
                       abstract_lambda=float(paper["abstract_lambda"]),
                       table_risk_aversion=table_risk_aversion, table_top_k=table_top_k,
                       tail_quantile=float(experiments["selection"]["tail_quantile"]),
                       confidence=float(experiments["statistics"]["confidence"]),
                       capital_usd=float(experiments["costs"]["capital_usd"]),
                       paper_dir=ROOT / paper["paper_dir"], main_tex=paper["main_tex"],
                       upload_zip=ROOT / paper["upload_zip"])


def figures() -> dict[str, str]:
    """Figure (relative to the results dir) -> file name in the paper's generated/ folder. paper/ holds the figures
    drawn by paper_figures.py from the saved results."""
    return {"paper/fig_dollar.pdf": "fig_dollar.pdf",
            "paper/fig_fees_vs_il.pdf": "fig_fees_vs_il.pdf",
            "paper/fig_concentrated.pdf": "fig_concentrated.pdf",
            "paper/fig_picks.pdf": "fig_picks.pdf",
            "paper/fig_calibration.pdf": "fig_calibration.pdf"}


# ---------------------------------------------------------------- formatting

def num(x: float, digits: int = 2) -> str:
    return f"{round(x, digits) + 0.0:.{digits}f}"  # + 0.0 turns -0.0 into 0.0


def pct(x: float, digits: int = 2) -> str:
    return num(100 * x, digits)


def pvalue(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def tabular(columns: str, header: str, rows: list[str]) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{columns}}}", "\\toprule", header + r" \\", "\\midrule",
                      *[r + r" \\" for r in rows], "\\bottomrule", "\\end{tabular}", ""])


def lam_name(lam: float) -> str:
    return f"lvr_tail_{lam:g}"


# ---------------------------------------------------------------- tables

def pair_list(pools: list[dict]) -> str:
    """'WETH/USDT 0.05, 0.30\\%; DAI/WETH 0.30\\%': each pair once (in order of appearance), its fee tiers ascending.
    Labels are '<pair> <tier>%'."""
    tiers: dict[str, list[str]] = {}
    for pool in pools:
        pair, tier = pool["label"].split()
        tiers.setdefault(pair, []).append(tier.removesuffix("%"))
    return "; ".join(f"{pair} {', '.join(sorted(t, key=float))}\\%" for pair, t in tiers.items())


def pools_table(data_dir: Path, universe: list[dict]) -> str:
    meta = pd.read_csv(data_dir / "meta" / "pools.csv", dtype=str).set_index("id")
    rows = []
    for category, name in CATEGORY_NAMES.items():
        pools = [p for p in universe if p["category"] == category]
        rows.append(f"{name} & {len(pools)} & {pair_list(pools)}")
    assert all(p["address"] in meta.index for p in universe), "pool metadata is missing some pools"
    return tabular("@{}l@{\\hspace{3pt}}c@{\\hspace{3pt}}>{\\raggedright\\arraybackslash}p{0.66\\columnwidth}@{}",
                   r"Category & $n$ & Pools (pair and fee tier)", rows)


def categories_table(results: Path) -> str:
    e2 = pd.read_csv(results / "E2" / "summary_by_category.csv", index_col=0)
    rows = []
    for category, name in CATEGORY_NAMES.items():
        r = e2.loc[category]
        fee = r["mean_net"] - r["mean_il"]
        offset = "--" if pd.isna(r["covariance_offset_share"]) or r["mean_var_individual"] < 1e-4 \
            else f"{pct(r['covariance_offset_share'], 0)}\\%"
        rows.append(f"{name} & {int(r['n_pool_months'])} & {pct(fee, 3)} & {pct(r['mean_il'], 3)} & "
                    f"{pct(r['mean_predicted_il'], 3)} & {pct(r['il_tail'], 2)} & {pct(r['mean_net'], 3)} & {offset}")
    return tabular("@{}lrrrrrrr@{}", r"Category & $n$ & Fee & IL & $-\sigma^2/8$ & $P_{10}$ IL & Net & Corr.", rows)


def validation_table(results: Path) -> str:
    e0 = pd.read_csv(results / "E0" / "fee_validation_summary.csv").set_index("label").loc["ALL"]
    e8 = pd.read_csv(results / "E8" / "swap_validation_summary.csv").set_index("range")
    """Absolute relative fee error: real positions (median, P90) and swap replays per range (median, max). The two
    benchmarks count different units (position intervals, pool-months), so the counts are quoted in the text."""
    rows = [f"Real positions & {pct(e0['median_abs_rel_error'], 3)} & {pct(e0['p90_abs_rel_error'], 2)}",
            r"\midrule" + "\n" + r"\multicolumn{3}{@{}l}{Swap replay:} "]
    for name, label in (("full", "full range"), ("±5%", r"$\pm5\%$"), ("±1%", r"$\pm1\%$")):
        r = e8.loc[name]
        rows.append(f"\\quad {label} & {pct(r['median_abs_rel_error'], 2)} & {pct(r['max_abs_rel_error'], 2)}")
    return tabular("@{}lrr@{}", r"Benchmark & Median & P90/max", rows)


def calibration_table(results: Path) -> str:
    e4 = pd.read_csv(results / "E4" / "exceedance_tests.csv")
    pooled = e4[e4["label"] == "pooled"].set_index(["metric", "scheme"])
    names = {"il_tail_model": r"Model tail (ours)", "il_tail": r"Empirical $P_{10}$", "net_tail": "Net90 bound"}
    rows = []
    for metric, name in names.items():
        for scheme in sorted({s for _, s in pooled.index}, reverse=True):
            r = pooled.loc[(metric, scheme)]
            window = "12m" if scheme.startswith("rolling") else "exp."
            rows.append(f"{name} & {window} & {pct(r['rate'], 1)} & [{pct(r['wilson_lo'], 1)}, {pct(r['wilson_hi'], 1)}]"
                        f" & {pvalue(r['kupiec_p'])} & {pvalue(r['christoffersen_p'])}")
    return tabular("@{}llrcrr@{}", r"Forecast & Win. & Rate & 95\% CI & Kupiec & Christ.", rows)


# Scores whose IC is not comparable with the others: the screen ranks only the pools that pass it.
NO_IC = ("fee_low_variance", "equal_weight")
TABLE_BASELINES = ("median_fee", "fee_low_variance", "low_variance", "net_tail", "rv", "farv", "equal_weight")


def selection_table(results: Path, risk_aversion: tuple[float, ...], top_k: tuple[int, ...]) -> str:
    """IC and top-k mean / CVaR of S_lambda (lambda = 0 and each of risk_aversion) and of the baselines. The top-k
    mean / CVaR of RV and FARV come from E10, which runs the same backtest (its Net90 rows equal E5's)."""
    rv_farv = ["rv", "farv"]
    performance = pd.read_csv(results / "E10" / "performance.csv").query("strategy in @rv_farv")
    frontier = pd.concat([pd.read_csv(results / "E5" / "risk_return_frontier.csv"), performance])
    frontier = frontier.set_index(["strategy", "k"])
    ic = pd.read_csv(results / "E5" / "ic_summary.csv")
    ic = ic[ic["level"] == "pool"].set_index("score")
    random = pd.read_csv(results / "E5" / "random_portfolios.csv").set_index("k")

    def row(strategy: str, name: str) -> str:
        ic_value = "--" if strategy in NO_IC else num(ic.loc[strategy, "ic_mean"])
        cells = [f"{pct(frontier.loc[(strategy, k), 'mean'])} & {pct(frontier.loc[(strategy, k), 'cvar'])}"
                 for k in top_k]
        return f"{name} & {ic_value} & " + " & ".join(cells)

    rows = [row("fee_minus_lvr", r"$S_0$ (fee $-\,\sigma^2/8$)")]
    rows += [row(lam_name(lam), f"$S_{{{lam:g}}}$") for lam in risk_aversion]
    baselines = [row(strategy, STRATEGY_NAMES[strategy]) for strategy in TABLE_BASELINES]
    rows += ["\\midrule\n" + baselines[0], *baselines[1:]]
    rows.append("Random $k$ pools & -- & " + " & ".join(f"{pct(random.loc[k, 'random_mean'])} & --" for k in top_k))
    header_k = " & ".join(f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in top_k)
    sub = " & ".join("Mean & CVaR" for _ in top_k)
    cmid = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(top_k)))
    columns = "@{}lr" + "rr" * len(top_k) + "@{}"
    return tabular(columns, f" & & {header_k} \\\\ {cmid}\nStrategy & IC & {sub}", rows)


RULE_NAMES = {"fixed_volume": "Fixed volume", "proportional": "Proportional"}
CONCENTRATED_COLUMNS = ("median_fee", "fee_low_variance", "low_variance", "fee_minus_lvr", "lvr_tail_0.25",
                        "lvr_tail_1", "equal_weight")


def range_label(name: str) -> str:
    """'±1%' -> '$\\pm1\\%$', '±2σ' -> '$\\pm2\\sigma$'."""
    return "$" + name.replace("±", r"\pm").replace("%", r"\%").replace("σ", r"\sigma") + "$"


def concentrated_table(results: Path, k: int) -> str:
    """E8: mean monthly excess return after costs of the top-k pools held as concentrated positions, per range and
    full-range score; * marks a significant difference from the highest median fee (paired test, p < 0.05)."""
    top = pd.read_csv(results / "E8" / "top_k_by_range.csv").query("k == @k").set_index(["range", "strategy"])
    tests = pd.read_csv(results / "E8" / "tests_by_range.csv").query("k == @k and baseline == 'median_fee'")
    significant = set(zip(tests.loc[tests["mean_p_value"] < 0.05, "range"],
                          tests.loc[tests["mean_p_value"] < 0.05, "strategy"]))
    rows = []
    for name in dict.fromkeys(top.index.get_level_values(0)):
        cells = [pct(top.loc[(name, s), "mean_excess"]) + ("$^*$" if (name, s) in significant else "")
                 for s in CONCENTRATED_COLUMNS]
        rows.append(f"{range_label(name)} & " + " & ".join(cells))
    header = (r"Range & Fee & \makecell{Fee,\\low $\sigma^2$} & \makecell{Low\\$\sigma^2$} & $S_0$ & "
              r"$S_{0.25}$ & $S_1$ & All")
    return tabular("@{}lrrrrrrr@{}", header, rows)


def crediting_table(results: Path, k: int) -> str:
    """E12: IC and top-k mean excess return of fee - sigma^2/8 and of the highest median fee, per crediting rule and
    position size, with the S_0 mean when the thin-liquidity pool-months are ineligible."""
    perf = pd.read_csv(results / "E12" / "crediting_performance.csv").query("k == @k")
    perf = perf.set_index(["rule", "position_usd", "strategy"])
    ic = pd.read_csv(results / "E12" / "crediting_ic.csv").set_index(["rule", "position_usd", "score"])
    rows = []
    for rule, size in dict.fromkeys(zip(perf.index.get_level_values(0), perf.index.get_level_values(1))):
        lvr, fee = perf.loc[(rule, size, "fee_minus_lvr")], perf.loc[(rule, size, "median_fee")]
        rows.append(f"{RULE_NAMES[rule]} & {size / 1000:,.0f}K & {num(ic.loc[(rule, size, 'fee_minus_lvr'), 'ic_mean'])} & "
                    f"{num(ic.loc[(rule, size, 'median_fee'), 'ic_mean'])} & {pct(lvr['mean'])} "
                    f"[{pct(lvr['mean_lo'])}, {pct(lvr['mean_hi'])}] & {pct(lvr['mean_no_thin'])} & {pct(fee['mean'])}")
    header = (r"Rule & Size & \multicolumn{2}{c}{IC} & \multicolumn{3}{c}{Top-" + str(k) + r" mean (\%)} \\ "
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}" + "\n"
              r"& & $S_0$ & Fee & $S_0$ [95\% CI] & No thin & Fee")
    return tabular("@{}llrrrrr@{}", header, rows)


# ---------------------------------------------------------------- numbers quoted in the text

def numbers(results: Path, data_dir: Path, universe: list[dict], risk_aversion: tuple[float, ...],
            sensitivity_k: int, main_k: int, abstract_lambda: float, capital_usd: float) -> str:
    e0 = pd.read_csv(results / "E0" / "fee_validation_summary.csv").set_index("label").loc["ALL"]
    e2 = pd.read_csv(results / "E2" / "summary_by_category.csv", index_col=0)
    e4 = pd.read_csv(results / "E4" / "exceedance_tests.csv")
    pooled = e4[e4["label"] == "pooled"].set_index(["metric", "scheme"])
    rolling = next(s for _, s in pooled.index if s.startswith("rolling"))
    e5_ic = pd.read_csv(results / "E5" / "ic_summary.csv").query("level == 'pool'").set_index("score")
    eligible = pd.read_csv(results / "E5" / "eligible_pools_per_period.csv")
    tests = pd.read_csv(results / "E5" / "net_tail_vs_baselines.csv").query("measure == 'excess'")
    e6 = pd.read_csv(results / "E6" / "sensitivity_grid.csv")
    e8 = pd.read_csv(results / "E8" / "swap_validation_summary.csv").set_index("range")
    e3 = pd.read_csv(results / "E3" / "rolling_ar1_summary.csv")
    sanity = pd.read_csv(data_dir / "reports" / "sanity_report.csv")
    families = {"/".join(sorted(p["label"].split()[0].split("/"))) for p in universe}
    data_end = pd.Timestamp(sanity["last"].max())

    lvr_tests = tests[tests["baseline"].isin([lam_name(x) for x in risk_aversion if 0 < x <= 0.5])]
    lines = [
        "% Generated by make_paper_assets.py from the experiment results; do not edit by hand.",
        macro("NPools", str(len(universe))), macro("NFamilies", str(len(families))),
        macro("SensitivityK", str(sensitivity_k)),
        macro("DataEnd", f"{data_end:%B}~{data_end.day},~{data_end.year}"),
        macro("NHours", f"{int(sanity['n_hours'].sum()):,}".replace(",", "{,}")),
        macro("NDecisions", str(len(eligible))),
        macro("EligibleMin", str(int(eligible["n_eligible"].min()))),
        macro("EligibleMax", str(int(eligible["n_eligible"].max()))),
        # E0 / E8 fee-model validation
        macro("FeePositions", f"{int(e0['n_positions']):,}".replace(",", "{,}")),
        macro("FeeIntervals", f"{int(e0['n_compared']):,}".replace(",", "{,}")),
        macro("FeeMedianErr", pct(e0["median_abs_rel_error"], 3) + "\\%"),
        macro("FeeWithinFive", pct(e0["share_within_0.05"], 1) + "\\%"),
        macro("SwapErrFull", pct(e8.loc["full", "median_abs_rel_error"], 1) + "\\%"),
        macro("SwapErrFive", pct(e8.loc["±5%", "median_abs_rel_error"], 1) + "\\%"),
        macro("SwapErrOne", pct(e8.loc["±1%", "median_abs_rel_error"], 1) + "\\%"),
        macro("NSwapMonths", str(int(e8.loc["full", "n"]))),
        *volatile_pick_numbers(results / "E8" / "picks_by_category.csv", main_k),
        # E2 LVR decomposition
        macro("MeanIL", pct(e2.loc["ALL", "mean_il"], 2) + "\\%"),
        macro("MeanPredIL", pct(e2.loc["ALL", "mean_predicted_il"], 2) + "\\%"),
        macro("LvrSlope", num(e2.loc["ALL", "slope_ex_post"])),
        macro("LvrRsq", num(e2.loc["ALL", "r2_ex_post"])),
        macro("CorrOffset", f"{100 * e2.loc['correlated', 'covariance_offset_share']:.0f}\\%"),
        macro("CorrRho", num(e2.loc["correlated", "mean_rho"])),
        # Introduction: mean monthly IL and fee yield of correlated and stable-volatile pairs (fee = net - IL)
        macro("CatCorrIL", pct(-e2.loc["correlated", "mean_il"]) + "\\%"),
        macro("CatStableVolIL", pct(-e2.loc["stable-volatile", "mean_il"]) + "\\%"),
        macro("CatCorrFee", pct(e2.loc["correlated", "mean_net"] - e2.loc["correlated", "mean_il"]) + "\\%"),
        macro("CatStableVolFee",
              pct(e2.loc["stable-volatile", "mean_net"] - e2.loc["stable-volatile", "mean_il"]) + "\\%"),
        # E4 calibration
        macro("ModelTailRate", pct(pooled.loc[("il_tail_model", rolling), "rate"], 1) + "\\%"),
        macro("ModelTailKupiec", num(pooled.loc[("il_tail_model", rolling), "kupiec_p"])),
        macro("EmpTailRate", pct(pooled.loc[("il_tail", rolling), "rate"], 1) + "\\%"),
        macro("NetTailRate", pct(pooled.loc[("net_tail", rolling), "rate"], 1) + "\\%"),
        # E4 details quoted in the text now that the calibration table is a figure
        macro("ModelTailLo", pct(pooled.loc[("il_tail_model", rolling), "wilson_lo"], 1) + "\\%"),
        macro("ModelTailHi", pct(pooled.loc[("il_tail_model", rolling), "wilson_hi"], 1) + "\\%"),
        macro("ModelTailChrist", num(pooled.loc[("il_tail_model", rolling), "christoffersen_p"])),
        macro("EmpTailKupiec", pvalue(pooled.loc[("il_tail", rolling), "kupiec_p"])),
        macro("ModelTailRateExp", pct(pooled.loc[("il_tail_model", "expanding"), "rate"], 1) + "\\%"),
        macro("EmpTailRateExp", pct(pooled.loc[("il_tail", "expanding"), "rate"], 1) + "\\%"),
        macro("EmpTailChristExp", pvalue(pooled.loc[("il_tail", "expanding"), "christoffersen_p"])),
        *category_calibration_numbers(e4, rolling),
        *standardized_move_numbers(results / "E4" / "standardized_moves.csv"),
        # E5 selection
        macro("IcLvr", num(e5_ic.loc["fee_minus_lvr", "ic_mean"])),
        macro("IcLvrT", num(e5_ic.loc["fee_minus_lvr", "ic_nw_t"], 1)),
        macro("IcNetTail", num(e5_ic.loc["net_tail", "ic_mean"])),
        macro("IcNetTailT", num(e5_ic.loc["net_tail", "ic_nw_t"], 1)),
        macro("IcLowVar", num(e5_ic.loc["low_variance", "ic_mean"])),
        macro("NLvrTestsSig", str(int((lvr_tests["p_value"] < 0.05).sum()))),
        macro("NLvrTests", str(len(lvr_tests))),
        # E6 sensitivity
        macro("NSensCells", str(len(e6))),
        macro("NSensNetTailBehind", str(int((e6["diff_vs_fee_minus_lvr"] < 0).sum()))),
        # Abstract headline (E5, top-k)
        *abstract_numbers(results / "E5" / "risk_return_frontier.csv", main_k, abstract_lambda),
        # E8 concentrated ranges (IC of the full-range score against the realized concentrated net return)
        *range_numbers(results / "E8" / "ic_by_range.csv", risk_aversion),
        # E3 persistence
        macro("HalfLifeMin", f"{e3['mean_half_life'].min():.0f}"),
        macro("HalfLifeMax", f"{e3['mean_half_life'].max():.0f}"),
        *persistence_numbers(results / "E3"),
        # E5 fee - sigma^2/8 against the highest median fee
        *fee_baseline_numbers(results / "E5", main_k),
        *no_thin_numbers(results / "E5", main_k),
        # E10 RV / FARV
        *rv_numbers(results / "E10" / "ic_summary.csv"),
        # E8 top-k concentrated returns
        *range_top_k_numbers(results / "E8" / "top_k_by_range.csv", main_k),
        # E12 fee crediting and the swap-by-swap replay of the thin-liquidity months
        *crediting_numbers(results, main_k),
        # E5 the lambda frontier against the fee baselines (full range)
        *full_range_tail_numbers(results / "E5", main_k, abstract_lambda),
        # E8 concentrated ranges against the fee baselines, and who the scores pick
        *concentrated_numbers(results / "E8", main_k, abstract_lambda),
        # E13 / E14 a separate correlation model
        *correlation_model_numbers(results),
    ]
    lines += dollar_numbers(results, main_k, capital_usd, abstract_lambda)
    return "\n".join(lines) + "\n"


def usd(value: float, step: int) -> str:
    return f"{round(value / step) * step:,.0f}".replace(",", "{,}").replace("-", "$-$") + "~USD"


# Strategies of the dollar story (fig_dollar): E5 name -> macro suffix
DOLLAR_STRATEGIES = {"median_fee": "Fee", "fee_minus_lvr": "Lvr", "net_tail": "NetTail", "low_variance": "LowVar",
                     "equal_weight": "Equal"}


def dollar_numbers(results: Path, k: int, capital_usd: float, lam: float) -> list[str]:
    """The dollar story of fig_dollar: final gain over HODL (summed, not compounded) of the top-k pools of each
    strategy, and the largest drop from a previous high, for the configured capital (rounded to 10 USD)."""
    paths = cumulative_usd(results, k, capital_usd)
    names = DOLLAR_STRATEGIES | {lam_name(lam): "Tail"}
    lines = [macro("CapitalUsd", usd(capital_usd, 1000))]
    for strategy, suffix in names.items():
        path = paths[strategy]
        drop = path - path.cummax()
        lines += [macro(f"DollarTotal{suffix}", usd(path.iloc[-1], 10)),
                  macro(f"DollarDrop{suffix}", usd(-drop.min(), 10))]
    s0 = paths["fee_minus_lvr"]
    trough = (s0 - s0.cummax()).idxmin()
    peak = s0.loc[:trough].idxmax()
    lines += [macro("DollarDropStart", f"{peak:%B}~{peak.year}"),
              macro("DollarDropEnd", f"{trough:%B}~{trough.year}")]
    return lines


VOLATILE_CATEGORIES = ("stable-volatile", "governance", "high-vol")
HIGH_FEE_CATEGORIES = ("governance", "high-vol")


def volatile_pick_numbers(path: Path, k: int) -> list[str]:
    """Share of the top-k positions of S_0 and of the fee ranking held in governance and high-volatility pools."""
    picks = pd.read_csv(path).query("k == @k")
    share = picks[picks["category"].isin(HIGH_FEE_CATEGORIES)].groupby("strategy")["share"].sum()
    return [macro("HighFeeShareLvr", f"{100 * share['fee_minus_lvr']:.0f}\\%"),
            macro("HighFeeShareFee", f"{100 * share['median_fee']:.0f}\\%")]


def category_calibration_numbers(e4: pd.DataFrame, rolling: str) -> list[str]:
    """Exceedance rates of the 12-month forecasts pooled within each category (pool rows summed)."""
    rows = e4[(e4["label"] != "pooled") & (e4["scheme"] == rolling)].drop_duplicates(["label", "metric"])
    sums = rows.groupby(["metric", "category"])[["n", "exceedances"]].sum()
    rate = sums["exceedances"] / sums["n"]
    model, empirical = rate.loc["il_tail_model"], rate.loc["il_tail"]
    return [
        macro("ModelTailCatMin", pct(model.min(), 1) + "\\%"),
        macro("ModelTailCatMax", pct(model.max(), 1) + "\\%"),
        macro("ModelTailStableVol", pct(model.loc["stable-volatile"], 1) + "\\%"),
        macro("EmpTailCatMin", pct(empirical.min(), 1) + "\\%"),
        macro("NCatModelBetter", str(int(((model - 0.1).abs() < (empirical - 0.1).abs()).sum()))),
        macro("NCategories", str(len(model))),
    ]


def standardized_move_numbers(path: Path) -> list[str]:
    """Monthly log price change over the realized volatility of the same month (E4 standardized_moves.csv)."""
    moves = pd.read_csv(path).set_index("category")
    volatile = moves.loc[list(VOLATILE_CATEGORIES)]
    return [
        macro("StdMoveStd", num(moves.loc["ALL", "std"])),
        macro("StdMoveKurtMax", num(volatile["excess_kurtosis"].max())),
        macro("StdMoveShareMin", pct(volatile["tail_share"].min(), 1) + "\\%"),
        macro("StdMoveShareMax", pct(volatile["tail_share"].max(), 1) + "\\%"),
        macro("StdMoveCorrKurt", num(moves.loc["correlated", "excess_kurtosis"], 1)),
        macro("StdMoveCorrShare", pct(moves.loc["correlated", "tail_share"], 1) + "\\%"),
    ]


def abstract_numbers(path: Path, k: int, lam: float) -> list[str]:
    """Top-k headline numbers: fee - sigma^2/8 vs Net90, and what a moderate tail penalty adds."""
    frontier = pd.read_csv(path).set_index(["strategy", "k"])
    lvr, net90, tail = (frontier.loc[(name, k)] for name in ("fee_minus_lvr", "net_tail", lam_name(lam)))
    return [
        macro("MainK", str(k)),
        macro("AbsLvrMean", pct(lvr["mean"]) + "\\%"),
        macro("AbsNetTailMean", pct(net90["mean"]) + "\\%"),
        macro("AbsGainVsNetTail", f"{num(100 * (lvr['mean'] / net90['mean'] - 1), 0)}\\%"),
        macro("AbsCvarCut", f"{num(100 * (1 - tail['cvar'] / lvr['cvar']), 0)}\\%"),
        macro("AbsReturnKept", f"{num(100 * tail['mean'] / lvr['mean'], 0)}\\%"),
    ]


def range_numbers(path: Path, risk_aversion: tuple[float, ...]) -> list[str]:
    e8 = pd.read_csv(path)
    ic = e8.pivot(index="range", columns="score", values="full_score_ic_mean")
    net = e8.drop_duplicates("range").set_index("range")["mean_net"]
    sigma_ranges = [r for r in ic.index if r.endswith("σ")]
    wide = ["±10%", "±20%", *sigma_ranges]
    lam_max = max(risk_aversion)
    return [
        macro("IcRangeOne", num(ic.loc["±1%", "fee_minus_lvr"])),
        macro("IcRangeFive", num(ic.loc["±5%", "fee_minus_lvr"])),
        macro("IcRangeTwenty", num(ic.loc["±20%", "fee_minus_lvr"])),
        macro("IcRangeSigmaMin", num(ic.loc[sigma_ranges, "fee_minus_lvr"].min())),
        macro("IcRangeSigmaMax", num(ic.loc[sigma_ranges, "fee_minus_lvr"].max())),
        macro("IcRangeOneTail", num(ic.loc["±1%", lam_name(lam_max)])),
        macro("LamMax", f"{lam_max:g}"),
        macro("IcRangeNetTailMax", num(ic.loc[wide, "net_tail"].max())),
        macro("NetRangeOne", pct(net.loc["±1%"]) + "\\%"),
        macro("NetRangeTwenty", pct(net.loc["±20%"]) + "\\%"),
        macro("NetRangeSigma", pct(net.loc[sigma_ranges[-1]]) + "\\%"),
        macro("SigmaRangeMax", sigma_ranges[-1].removeprefix("±").removesuffix("σ")),
    ]


def persistence_numbers(e3_dir: Path) -> list[str]:
    rolling = pd.read_csv(e3_dir / "rolling_ar1_summary.csv")
    ratio = pd.read_csv(e3_dir / "variance_ratio.csv")
    regimes = pd.read_csv(e3_dir / "regimes_ar1_vs_ar2.csv").query("regime == 'full'")
    return [
        macro("AdfRejectShare", f"{100 * rolling['share_adf_rejects_unit_root'].mean():.0f}\\%"),
        macro("KpssRejectShare", f"{100 * rolling['share_kpss_rejects_stationarity'].mean():.0f}\\%"),
        macro("NVrTests", str(len(ratio))),
        macro("NVrRejects", str(int((ratio["p_value"] < 0.05).sum()))),
        macro("NArOneBic", str(int((regimes["bic_prefers"] == "AR(1)").sum()))),
        macro("NArPools", str(len(regimes))),
    ]


def fee_baseline_numbers(e5_dir: Path, k: int) -> list[str]:
    ic = pd.read_csv(e5_dir / "ic_summary.csv").query("level == 'pool'").set_index("score")
    tests = pd.read_csv(e5_dir / "paired_tests.csv").query(
        "reference == 'fee_minus_lvr' and baseline == 'median_fee' and measure == 'excess'").set_index("k")
    frontier = pd.read_csv(e5_dir / "risk_return_frontier.csv").set_index(["strategy", "k"])
    lvr, fee = frontier.loc[("fee_minus_lvr", k)], frontier.loc[("median_fee", k)]
    return [
        macro("IcMedianFee", num(ic.loc["median_fee", "ic_mean"])),
        macro("IcMedianFeeT", num(ic.loc["median_fee", "ic_nw_t"], 1)),
        macro("AbsFeeMean", pct(fee["mean"]) + "\\%"),
        macro("AbsLvrLo", pct(lvr["mean_lo"]) + "\\%"),
        macro("AbsLvrHi", pct(lvr["mean_hi"]) + "\\%"),
        macro("AbsLvrMeanExcl", pct(lvr["mean_excluding"]) + "\\%"),
        macro("LvrVsFeeDiff", pct(tests.loc[k, "mean_difference"]) + "\\%"),
        macro("LvrVsFeeP", pvalue(tests.loc[k, "p_value"])),
    ]


def no_thin_numbers(e5_dir: Path, k: int) -> list[str]:
    """Robustness check with the thin-liquidity pool-months ineligible for every strategy."""
    robust = pd.read_csv(e5_dir / "no_thin_months.csv").query("k == @k").set_index("strategy")
    return [
        macro("NoThinLvrMean", pct(robust.loc["fee_minus_lvr", "mean"]) + "\\%"),
        macro("NoThinLvrLo", pct(robust.loc["fee_minus_lvr", "mean_lo"]) + "\\%"),
        macro("NoThinLvrHi", pct(robust.loc["fee_minus_lvr", "mean_hi"]) + "\\%"),
        macro("NoThinFeeMean", pct(robust.loc["median_fee", "mean"]) + "\\%"),
        macro("NoThinNetTailMean", pct(robust.loc["net_tail", "mean"]) + "\\%"),
        macro("NoThinLvrVsNetTailP", pvalue(robust.loc["net_tail", "p_value"])),
        macro("NoThinLvrVsFeeP", pvalue(robust.loc["median_fee", "p_value"])),
    ]


def rv_numbers(path: Path) -> list[str]:
    ic = pd.read_csv(path).set_index("score")
    return [macro("IcRv", num(ic.loc["rv", "ic_mean"])), macro("IcRvT", num(ic.loc["rv", "ic_nw_t"], 1)),
            macro("IcFarv", num(ic.loc["farv", "ic_mean"])), macro("IcFarvT", num(ic.loc["farv", "ic_nw_t"], 1))]


def range_top_k_numbers(path: Path, k: int) -> list[str]:
    top = pd.read_csv(path).query("k == @k").set_index(["range", "strategy"])["mean_excess"]
    return [
        macro("TopRangeTwoSigma", pct(top.loc[("±2σ", "fee_minus_lvr")]) + "\\%"),
        macro("AllRangeTwoSigma", pct(top.loc[("±2σ", "equal_weight")]) + "\\%"),
        macro("TopRangeTwenty", pct(top.loc[("±20%", "fee_minus_lvr")]) + "\\%"),
        macro("AllRangeTwenty", pct(top.loc[("±20%", "equal_weight")]) + "\\%"),
        macro("TopRangeOne", pct(top.loc[("±1%", "fee_minus_lvr")]) + "\\%"),
        macro("AllRangeOne", pct(top.loc[("±1%", "equal_weight")]) + "\\%"),
    ]


def crediting_numbers(results: Path, k: int) -> list[str]:
    thin = pd.read_csv(results / "thin_liquidity_months.csv")
    replay = pd.read_csv(results / "E12" / "replay.csv")
    thin_keys = set(zip(thin["label"], thin["period"]))
    normal = replay[[(l, p) not in thin_keys for l, p in zip(replay["label"], replay["period"])]]
    perf = pd.read_csv(results / "E12" / "crediting_performance.csv").query("k == @k and strategy == 'fee_minus_lvr'")
    means = perf.set_index(["rule", "position_usd"])["mean"]
    ic = pd.read_csv(results / "E12" / "crediting_ic.csv").pivot_table(
        index=["rule", "position_usd"], columns="score", values="ic_mean")
    return [
        macro("NCreditCells", str(len(ic))),
        macro("NCreditFeeIcHigher", str(int((ic["median_fee"] > ic["fee_minus_lvr"]).sum()))),
        macro("NThinMonths", str(len(thin))),
        macro("NThinReplayed", str(int(thin["replayed"].sum()))),
        macro("NReplayNormal", str(len(normal))),
        macro("ReplayErrFixed", pct(normal["fixed_volume_rel_error"].abs().median(), 1) + "\\%"),
        macro("CreditMinMean", pct(means.min()) + "\\%"),
        macro("CreditMaxMean", pct(means.max()) + "\\%"),
    ]


def full_range_tail_numbers(e5_dir: Path, k: int, lam: float) -> list[str]:
    """lambda = abstract_lambda against the highest median fee (k and k = 5), S_0 against the variance screen."""
    tail = pd.read_csv(e5_dir / "tail_tests.csv").set_index(["k", "risk_aversion", "baseline"])
    perf = pd.read_csv(e5_dir / "performance.csv").query("measure == 'excess'").set_index(["k", "strategy"])
    frontier = pd.read_csv(e5_dir / "risk_return_frontier.csv").set_index(["strategy", "k"])
    k_max = max(tail.index.get_level_values(0))
    main, wide = tail.loc[(k, lam, "median_fee")], tail.loc[(k_max, lam, "median_fee")]
    screen = tail.loc[(k, 0.0, "fee_low_variance")]
    return [
        macro("TailLam", f"{lam:g}"), macro("TailWideK", str(k_max)),
        macro("TailMean", pct(frontier.loc[(lam_name(lam), k), "mean"]) + "\\%"),
        macro("TailCvar", pct(frontier.loc[(lam_name(lam), k), "cvar"]) + "\\%"),
        macro("FeeCvar", pct(frontier.loc[("median_fee", k), "cvar"]) + "\\%"),
        macro("TailVsFeeMean", pct(main["mean_difference"]) + "\\%"),
        macro("TailVsFeeMeanP", pvalue(main["mean_p_value"])),
        macro("TailVsFeeCvar", pct(main["cvar_difference"]) + "\\%"),
        macro("TailVsFeeCvarP", pvalue(main["cvar_p_value"])),
        macro("TailVsFeeMeanWide", pct(wide["mean_difference"]) + "\\%"),
        macro("TailVsFeeMeanWideP", pvalue(wide["mean_p_value"])),
        macro("TailVsFeeCvarWide", pct(wide["cvar_difference"]) + "\\%"),
        macro("TailVsFeeCvarWideP", pvalue(wide["cvar_p_value"])),
        macro("TailDrawdown", pct(perf.loc[(k, lam_name(lam)), "max_drawdown"], 1) + "\\%"),
        macro("FeeDrawdown", pct(perf.loc[(k, "median_fee"), "max_drawdown"], 1) + "\\%"),
        macro("ScreenMean", pct(frontier.loc[("fee_low_variance", k), "mean"]) + "\\%"),
        macro("ScreenCvar", pct(frontier.loc[("fee_low_variance", k), "cvar"]) + "\\%"),
        macro("LvrVsScreen", pct(screen["mean_difference"]) + "\\%"),
        macro("LvrVsScreenP", pvalue(screen["mean_p_value"])),
    ]


def concentrated_numbers(e8_dir: Path, k: int, lam: float) -> list[str]:
    top = pd.read_csv(e8_dir / "top_k_by_range.csv").query("k == @k").set_index(["range", "strategy"])
    tests = pd.read_csv(e8_dir / "tests_by_range.csv").query("k == @k and baseline == 'median_fee'")
    tests = tests.set_index(["range", "strategy"])
    picks = pd.read_csv(e8_dir / "picks_summary.csv").query("k == @k").set_index("strategy")
    categories = pd.read_csv(e8_dir / "picks_by_category.csv").query("k == @k")
    stable_share = categories[categories["category"].isin(["correlated", "stable-stable"])].groupby("strategy")["share"].sum()
    volatile_share = categories[categories["category"].isin(["governance", "stable-volatile"])].groupby("strategy")["share"].sum()
    ranges = list(dict.fromkeys(top.index.get_level_values(0)))
    sigma = next(r for r in ranges if r.endswith("σ"))
    tail = lam_name(lam)
    one, sig = tests.loc[("±1%", "lvr_tail_1")], tests.loc[(sigma, tail)]
    fee = top.xs("median_fee", level="strategy")["mean_excess"]
    tail_tests = tests.xs(tail, level="strategy")
    strong = tests.xs("lvr_tail_1", level="strategy")
    return [
        macro("NRanges", str(len(ranges))),
        macro("NConcFeeNegative", str(int((fee < 0).sum()))),
        macro("ConcFeeOne", pct(fee.loc["±1%"]) + "\\%"),
        macro("ConcStrongOne", pct(top.loc[("±1%", "lvr_tail_1"), "mean_excess"]) + "\\%"),
        macro("ConcStrongOneDiff", pct(one["mean_difference"]) + "\\%"),
        macro("ConcStrongOneCvar", pct(one["cvar_difference"], 1)),
        macro("ConcFeeOneCvar", pct(top.loc[("±1%", "median_fee"), "cvar"], 1) + "\\%"),
        macro("ConcStrongOneCvarAbs", pct(top.loc[("±1%", "lvr_tail_1"), "cvar"], 1) + "\\%"),
        macro("ConcSigmaName", range_label(sigma)),
        macro("ConcFeeSigma", pct(fee.loc[sigma]) + "\\%"),
        macro("ConcTailSigma", pct(top.loc[(sigma, tail), "mean_excess"]) + "\\%"),
        macro("ConcTailSigmaDiff", pct(sig["mean_difference"]) + "\\%"),
        macro("ConcTailSigmaP", pvalue(sig["mean_p_value"])),
        macro("ConcLowVarSigma", pct(top.loc[(sigma, "low_variance"), "mean_excess"]) + "\\%"),
        macro("ConcLowVarOne", pct(top.loc[("±1%", "low_variance"), "mean_excess"]) + "\\%"),
        macro("NConcTailMeanBetter", str(int(((tail_tests["mean_difference"] > 0)
                                              & (tail_tests["mean_p_value"] < 0.05)).sum()))),
        macro("NConcStrongCvarBetter", str(int(((strong["cvar_difference"] > 0)
                                                & (strong["cvar_p_value"] < 0.05)).sum()))),
        macro("RhoEligible", num(picks.loc["median_fee", "mean_rho_eligible"])),
        macro("RhoFee", num(picks.loc["median_fee", "mean_rho_held"])),
        macro("RhoTail", num(picks.loc[tail, "mean_rho_held"])),
        macro("RhoStrong", num(picks.loc["lvr_tail_1", "mean_rho_held"])),
        macro("LowRiskShareFee", f"{100 * stable_share.loc['median_fee']:.0f}\\%"),
        macro("LowRiskShareTail", f"{100 * stable_share.loc[tail]:.0f}\\%"),
        macro("LowRiskShareStrong", f"{100 * stable_share.loc['lvr_tail_1']:.0f}\\%"),
        macro("ScreenVolatileShare", f"{100 * volatile_share.loc['fee_low_variance']:.0f}\\%"),
    ]


def correlation_model_numbers(results: Path) -> list[str]:
    """E14: out-of-sample score with and without rho; E13: correlation-aware portfolios against top-k."""
    ic = pd.read_csv(results / "E14" / "ic.csv").set_index(["range", "model"])
    difference = ic.xs("with_minus_without", level="model")
    portfolio = pd.read_csv(results / "E13" / "tests.csv").query("variant == 'joint_corr' and baseline == 'topk'")
    better = ((portfolio["mean_difference"] > 0) & (portfolio["mean_p_value"] < 0.05)) | \
             ((portfolio["cvar_difference"] > 0) & (portfolio["cvar_p_value"] < 0.05))
    worse = ((portfolio["mean_difference"] < 0) & (portfolio["mean_p_value"] < 0.05)) | \
            ((portfolio["cvar_difference"] < 0) & (portfolio["cvar_p_value"] < 0.05))
    return [
        macro("IcWithRho", num(ic.loc[("full", "with_rho"), "ic_mean"])),
        macro("IcWithoutRho", num(ic.loc[("full", "without_rho"), "ic_mean"])),
        macro("NRhoLower", str(int((difference["ic_mean"] < 0).sum()))),
        macro("NRhoRanges", str(len(difference))),
        macro("RhoIcDiffMinT", num(difference["ic_nw_t"].min(), 1)),
        macro("NPortCells", str(len(portfolio))),
        macro("NPortBetter", str(int(better.sum()))),
        macro("NPortWorse", str(int(worse.sum()))),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)
    config = load_config()
    with CONFIG_PATH.open("rb") as f:
        universe = tomllib.load(f)["universe"]["pools"]
    results, out = config.results_dir, config.output_dir
    if not results.is_dir():
        raise SystemExit(f"results_dir not found: {results}")

    outputs = {
        "numbers.tex": lambda: numbers(results, config.data_dir, universe, config.risk_aversion,
                                       config.sensitivity_k, config.main_k, config.abstract_lambda,
                                       config.capital_usd),
        "tab_pools.tex": lambda: pools_table(config.data_dir, universe),
        "tab_categories.tex": lambda: categories_table(results),
        "tab_validation.tex": lambda: validation_table(results),
        "tab_calibration.tex": lambda: calibration_table(results),
        "tab_selection.tex": lambda: selection_table(results, config.table_risk_aversion, config.table_top_k),
        "tab_crediting.tex": lambda: crediting_table(results, config.main_k),
        "tab_concentrated.tex": lambda: concentrated_table(results, config.main_k),
    }
    failed = []
    for name, build in outputs.items():
        try:
            write_atomic(out / name, build())
        except (FileNotFoundError, KeyError) as exc:
            log.error("%s: %s (has the experiment run?)", name, exc)
            failed.append(name)
    try:
        make_figures(results, config.main_k, config.risk_aversion, config.tail_quantile, config.confidence,
                     config.capital_usd)
    except (FileNotFoundError, KeyError) as exc:
        log.error("paper figures: %s (has the experiment run?)", exc)
    figure_files = figures()
    for source, target in figure_files.items():
        tmp = out / (target + ".tmp")
        try:
            shutil.copyfile(results / source, tmp)
            os.replace(tmp, out / target)
        except FileNotFoundError as exc:
            log.error("%s: %s (has the experiment run?)", target, exc)
            failed.append(target)
        except PermissionError as exc:
            tmp.unlink(missing_ok=True)
            log.error("%s: %s (is the file open in another program?)", target, exc)
            failed.append(target)
    log.info("wrote %d of %d assets to %s", len(outputs) + len(figure_files) - len(failed),
             len(outputs) + len(figure_files), out)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
    write_upload_zip(config.paper_dir, config.main_tex, config.upload_zip)
    log.info("wrote the Overleaf upload %s", config.upload_zip)


def write_upload_zip(paper_dir: Path, main_tex: str, path: Path) -> None:
    """Zips the paper's main .tex, references.bib and generated/ for upload to Overleaf, atomically."""
    files = [paper_dir / main_tex, paper_dir / "references.bib", *sorted((paper_dir / "generated").glob("*.*"))]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(paper_dir).as_posix())
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
