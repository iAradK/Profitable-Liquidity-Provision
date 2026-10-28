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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"

log = logging.getLogger("paper")

CATEGORY_NAMES = {
    "stable-volatile": "Stable--volatile", "stable-stable": "Stable--stable", "correlated": "Correlated",
    "governance": "Governance", "high-vol": "High-vol./declining",
}
STRATEGY_NAMES = {
    "fee_minus_lvr": r"Fee $-\,\sigma^2/8$", "net_tail": "Net90", "median_fee": "Highest median fee",
    "low_variance": r"Lowest $\sigma^2$", "mean_net": "Mean net", "equal_weight": "Equal weight",
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


def load_config(path: Path = CONFIG_PATH) -> PaperConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    paper, experiments = raw["paper"], raw["experiments"]
    return PaperConfig(results_dir=ROOT / paper["results_dir"], data_dir=ROOT / experiments["data"]["data_dir"],
                       output_dir=ROOT / paper["output_dir"],
                       risk_aversion=tuple(float(x) for x in experiments["selection"]["risk_aversion"]),
                       top_k=tuple(experiments["selection"]["top_k"]), sensitivity_k=paper["sensitivity_k"],
                       main_k=paper["main_k"], abstract_lambda=float(paper["abstract_lambda"]))


def figures(sensitivity_k: int) -> dict[str, str]:
    """Experiment figure (relative to the results dir) -> file name in the paper's generated/ folder."""
    return {"E5/risk_return_frontier.pdf": "fig_frontier.pdf",
            "E2/variance_decomposition.pdf": "fig_variance_decomposition.pdf",
            "E0/fee_validation_scatter.pdf": "fig_fee_validation.pdf",
            f"E6/sensitivity_heatmap_k{sensitivity_k}.pdf": "fig_sensitivity.pdf",
            "E8/ic_vs_range.pdf": "fig_ic_by_range.pdf"}


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

def pools_table(data_dir: Path, universe: list[dict]) -> str:
    meta = pd.read_csv(data_dir / "meta" / "pools.csv", dtype=str).set_index("id")
    rows = []
    for category, name in CATEGORY_NAMES.items():
        pools = [p for p in universe if p["category"] == category]
        labels = ", ".join(p["label"].replace("%", r"\%") for p in pools)
        rows.append(f"{name} & {len(pools)} & {labels}")
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
    rows = [f"Real full-range positions (E0) & {int(e0['n_compared'])} & {pct(e0['median_abs_rel_error'], 3)} & "
            f"{pct(e0['p90_abs_rel_error'], 2)}"]
    for name, label in (("full", "full range"), ("±5%", r"$\pm5\%$"), ("±1%", r"$\pm1\%$")):
        r = e8.loc[name]
        rows.append(f"Swap replay, {label} & {int(r['n'])} & {pct(r['median_abs_rel_error'], 2)} & "
                    f"{pct(r['max_abs_rel_error'], 2)}")
    return tabular("@{}lrrr@{}", r"Benchmark & $n$ & Median $|$err$|$ & P90/max $|$err$|$", rows)


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


def selection_table(results: Path, risk_aversion: tuple[float, ...], top_k: tuple[int, ...]) -> str:
    frontier = pd.read_csv(results / "E5" / "risk_return_frontier.csv").set_index(["strategy", "k"])
    ic = pd.read_csv(results / "E5" / "ic_summary.csv")
    ic = ic[ic["level"] == "pool"].set_index("score")
    random = pd.read_csv(results / "E5" / "random_portfolios.csv").set_index("k")
    order = ["fee_minus_lvr", *(lam_name(lam) for lam in risk_aversion),
             "net_tail", "median_fee", "low_variance", "equal_weight"]
    rows = []
    for strategy in order:
        if strategy.startswith("lvr_tail_"):
            name = f"\\quad $\\lambda={strategy.removeprefix('lvr_tail_')}$"
        else:
            name = STRATEGY_NAMES[strategy] + (" ($\\lambda{=}0$)" if strategy == "fee_minus_lvr" else "")
        ic_value = num(ic.loc[strategy, "ic_mean"]) if strategy in ic.index else "--"
        cells = [f"{pct(frontier.loc[(strategy, k), 'mean'])} & {pct(frontier.loc[(strategy, k), 'cvar'])}"
                 for k in top_k]
        rows.append(f"{name} & {ic_value} & " + " & ".join(cells))
    rows.append("Random $k$ pools & -- & " + " & ".join(f"{pct(random.loc[k, 'random_mean'])} & --" for k in top_k))
    header_k = " & ".join(f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in top_k)
    sub = " & ".join("Mean & CVaR" for _ in top_k)
    cmid = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(top_k)))
    columns = "@{}lr" + "rr" * len(top_k) + "@{}"
    return tabular(columns, f" & & {header_k} \\\\ {cmid}\nStrategy & IC & {sub}", rows)


# ---------------------------------------------------------------- numbers quoted in the text

def numbers(results: Path, data_dir: Path, universe: list[dict], risk_aversion: tuple[float, ...],
            sensitivity_k: int, main_k: int, abstract_lambda: float) -> str:
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
        # E2 LVR decomposition
        macro("MeanIL", pct(e2.loc["ALL", "mean_il"], 2) + "\\%"),
        macro("MeanPredIL", pct(e2.loc["ALL", "mean_predicted_il"], 2) + "\\%"),
        macro("LvrSlope", num(e2.loc["ALL", "slope_ex_post"])),
        macro("LvrRsq", num(e2.loc["ALL", "r2_ex_post"])),
        macro("CorrOffset", f"{100 * e2.loc['correlated', 'covariance_offset_share']:.0f}\\%"),
        macro("CorrRho", num(e2.loc["correlated", "mean_rho"])),
        # E4 calibration
        macro("ModelTailRate", pct(pooled.loc[("il_tail_model", rolling), "rate"], 1) + "\\%"),
        macro("ModelTailKupiec", num(pooled.loc[("il_tail_model", rolling), "kupiec_p"])),
        macro("EmpTailRate", pct(pooled.loc[("il_tail", rolling), "rate"], 1) + "\\%"),
        macro("NetTailRate", pct(pooled.loc[("net_tail", rolling), "rate"], 1) + "\\%"),
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
    ]
    return "\n".join(lines) + "\n"


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
                                       config.sensitivity_k, config.main_k, config.abstract_lambda),
        "tab_pools.tex": lambda: pools_table(config.data_dir, universe),
        "tab_categories.tex": lambda: categories_table(results),
        "tab_validation.tex": lambda: validation_table(results),
        "tab_calibration.tex": lambda: calibration_table(results),
        "tab_selection.tex": lambda: selection_table(results, config.risk_aversion, config.top_k),
    }
    failed = []
    for name, build in outputs.items():
        try:
            write_atomic(out / name, build())
        except (FileNotFoundError, KeyError) as exc:
            log.error("%s: %s (has the experiment run?)", name, exc)
            failed.append(name)
    figure_files = figures(config.sensitivity_k)
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


if __name__ == "__main__":
    main()
