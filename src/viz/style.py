"""Shared figure style: validated reference palette (dataviz skill) + NHL rink drawing.

Sequential (magnitude, e.g. V(s)) = one blue ramp light->dark.
Diverging (signed, e.g. PM-EPA deltas) = blue <-> red with a gray midpoint.
Categorical = fixed slot order (F = slot 1 blue, D = slot 2 orange).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POS_COLOR = {"F": SERIES[0], "D": SERIES[1]}
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#a32a29"])
RINK_LINE = "#8f8e89"
RED_LINE = "#d98a89"
BLUE_LINE = "#8fb3e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def draw_rink(ax, half: str | None = None, zorder: int = 5) -> None:
    """Draw NHL rink lines in feet (x -100..100, y -42.5..42.5), attacking +x."""
    lw = 1.1
    boards = FancyBboxPatch((-100, -42.5), 200, 85, boxstyle="round,pad=0,rounding_size=28", fill=False, ec=RINK_LINE, lw=1.4, zorder=zorder)
    ax.add_patch(boards)
    for artist in ax.collections:  # clip any heatmap drawn before the rink to the boards
        artist.set_clip_path(boards)
    for gx in (-89, 89):
        ax.plot([gx, gx], [-37.5, 37.5], color=RED_LINE, lw=lw, zorder=zorder)
    for bx in (-25, 25):
        ax.plot([bx, bx], [-42.5, 42.5], color=BLUE_LINE, lw=3, zorder=zorder)
    ax.plot([0, 0], [-42.5, 42.5], color=RED_LINE, lw=2.2, ls=(0, (6, 3)), zorder=zorder)
    ax.add_patch(Circle((0, 0), 15, fill=False, ec=BLUE_LINE, lw=lw, zorder=zorder))
    for cx in (-69, 69):
        for cy in (-22, 22):
            ax.add_patch(Circle((cx, cy), 15, fill=False, ec=RED_LINE, lw=lw, zorder=zorder))
            ax.add_patch(Circle((cx, cy), 1, color=RED_LINE, zorder=zorder))
    for cx in (-20, 20):
        for cy in (-22, 22):
            ax.add_patch(Circle((cx, cy), 1, color=RED_LINE, zorder=zorder))
    for gx, s in ((89, 1), (-89, -1)):
        ax.add_patch(Arc((gx, 0), 12, 12, theta1=90 if s > 0 else -90, theta2=270 if s > 0 else 90, ec=RED_LINE, lw=lw, zorder=zorder))
        ax.add_patch(Rectangle((gx if s > 0 else gx - 3.3, -3), 3.3, 6, fill=False, ec=RINK_LINE, lw=lw, zorder=zorder))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if half == "offense":
        ax.set_xlim(0, 100)
    elif half == "defense":
        ax.set_xlim(-100, 0)
    else:
        ax.set_xlim(-101, 101)
    ax.set_ylim(-43.5, 43.5)


def save(fig, path: Path, stem: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(path / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
