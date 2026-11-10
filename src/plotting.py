"""Shared figure style: light surface, recessive grid, fixed series colours per arm."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID = "#1f1e1c", "#55534c", "#8a887f", "#e4e2dc"
# Categorical slots 1 and 2 of the validated reference palette; colour follows the arm, never its rank.
ARM_COLOR = {"baseline": "#2a78d6", "ensemble": "#eb6834"}
REFERENCE = "#8a887f"     # diagonal / target lines


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8.5, "font.size": 9.5, "lines.linewidth": 2, "text.color": INK,
    })
