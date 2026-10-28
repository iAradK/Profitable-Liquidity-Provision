"""Shared figure style: validated categorical palette (fixed order), recessive axes, PNG + PDF output."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL_MID = "#f0efec"

CATEGORY_ORDER = ("stable-volatile", "stable-stable", "correlated", "governance", "high-vol")
STRATEGY_ORDER = ("net_tail", "fee_minus_lvr", "median_fee", "low_variance", "mean_net", "equal_weight", "rv", "farv")
STRATEGY_LABELS = {
    "net_tail": "Net90", "fee_minus_lvr": "Fee − σ²/8", "median_fee": "Highest median fee",
    "low_variance": "Lowest σ² (LVR)", "mean_net": "Mean net", "equal_weight": "Equal weight",
    "rv": "RV", "farv": "FARV", "hodl_5050": "HODL 50/50", "hodl_eth": "HODL ETH",
}

DIVERGING = LinearSegmentedColormap.from_list("diverging", [SERIES_COLORS[1], NEUTRAL_MID, SERIES_COLORS[0]])


def category_color(category: str) -> str:
    return SERIES_COLORS[CATEGORY_ORDER.index(category)] if category in CATEGORY_ORDER else INK_SECONDARY


def strategy_color(name: str) -> str:
    return SERIES_COLORS[STRATEGY_ORDER.index(name)] if name in STRATEGY_ORDER else INK_SECONDARY


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    """Writes <stem>.png and <stem>.pdf atomically, then closes the figure."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        final = path_stem.with_suffix(suffix)
        tmp = path_stem.with_name(path_stem.name + ".tmp" + suffix)
        fig.savefig(tmp)
        os.replace(tmp, final)
    plt.close(fig)
