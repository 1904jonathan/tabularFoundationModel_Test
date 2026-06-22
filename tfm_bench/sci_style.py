"""
Common scientific (publication-ready) plotting style for all figures.

Serif font, subtle grid, clean spines, consistent palette, high DPI.
Call set_scientific_style() once before plotting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Stable qualitative palette: one fixed color per model (cross-figure consistency).
MODEL_COLORS = {
    "TabICL":             "#D1495B",  # red -- the model of interest
    "LogisticRegression": "#3C6E9C",
    "XGBoost":            "#2A9D8F",
    "LightGBM":           "#E9C46A",
    "HistGB":             "#8D6A9F",
    "RandomForest":       "#F4A261",
    "KNN":                "#6C757D",
}

# Distinct markers per model (readability of curves in black & white).
MODEL_MARKERS = {
    "TabICL": "o", "LogisticRegression": "s", "XGBoost": "^",
    "LightGBM": "D", "HistGB": "v", "RandomForest": "P", "KNN": "X",
}


def color_for(name: str) -> str:
    return MODEL_COLORS.get(name, "#444444")


def marker_for(name: str) -> str:
    return MODEL_MARKERS.get(name, "o")


def set_scientific_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })
