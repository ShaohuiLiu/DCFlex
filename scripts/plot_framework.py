#!/usr/bin/env python3
"""Generate a clean overview diagram for the robust co-optimization framework.

This script creates a publication-style block diagram in both PDF and PNG
formats without modifying any existing manuscript or figure source files.

Run from the repository root with `make framework`.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = FIG_DIR / "fig_problem_framework_overview.pdf"
OUT_PNG = FIG_DIR / "fig_problem_framework_overview.png"

PALETTE = {
    "bg": "#fbfaf7",
    "text": "#243240",
    "muted": "#5c6b78",
    "line": "#243240",
    "campus_fill": "#f7f5ef",
    "campus_edge": "#d6cfbf",
    "compute": "#c97a1d",
    "compute_fill": "#fff1dd",
    "compute_soft": "#fde5be",
    "bess": "#2f7d57",
    "bess_fill": "#e7f6ed",
    "bess_soft": "#c8ecd7",
    "poi": "#41576d",
    "poi_fill": "#edf2f7",
    "grid": "#2d6cdf",
    "grid_fill": "#eef4ff",
    "grid_soft": "#d8e6ff",
    "market_soft": "#dcecff",
    "scenario": "#d86b4d",
    "scenario_fill": "#fff0ea",
    "scenario_soft": "#ffd8cc",
    "control": "#0b7285",
    "control_fill": "#e4f6f8",
    "control_soft": "#d0eef2",
    "revenue_fill": "#d7f0df",
    "energy_fill": "#dbeafe",
    "qos_fill": "#fff0cf",
    "deg_fill": "#ffe1dd",
    "white": "#ffffff",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Avenir Next",
            "Avenir",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
    }
)


def add_shadow(patch, alpha=0.11):
    patch.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(2.8, -2.8), shadow_rgbFace=(0.08, 0.11, 0.14), alpha=alpha
            ),
            pe.Normal(),
        ]
    )


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    facecolor,
    edgecolor,
    linewidth=1.5,
    radius=0.18,
    linestyle="solid",
    zorder=1,
    shadow=False,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=zorder,
    )
    if shadow:
        add_shadow(patch)
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, w, h, text, *, facecolor, edgecolor="none", color=None, fontsize=9.1, zorder=5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={h/2.0}",
        linewidth=0.9 if edgecolor != "none" else 0.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or PALETTE["text"],
        zorder=zorder + 1,
        weight="semibold",
    )


def arrow(
    ax,
    start,
    end,
    *,
    color,
    lw=2.0,
    style="-|>",
    linestyle="solid",
    mutation_scale=15,
    connectionstyle="arc3,rad=0.0",
    zorder=7,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow_label(ax, x, y, text, *, color, fontsize=9.2, rotation=0, facecolor=None, zorder=8):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        rotation=rotation,
        zorder=zorder,
        bbox={
            "boxstyle": "round,pad=0.2,rounding_size=0.12",
            "facecolor": facecolor or PALETTE["bg"],
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )


def draw_grid_tower(ax, x, y, scale=1.0, color=None):
    color = color or PALETTE["grid"]
    lw = 2.0
    ax.plot([x, x + 0.35 * scale, x + 0.7 * scale], [y, y + 1.05 * scale, y], color=color, lw=lw)
    ax.plot([x + 0.18 * scale, x + 0.35 * scale], [y + 0.54 * scale, y + 1.05 * scale], color=color, lw=lw)
    ax.plot([x + 0.52 * scale, x + 0.35 * scale], [y + 0.54 * scale, y + 1.05 * scale], color=color, lw=lw)
    ax.plot([x + 0.35 * scale, x + 0.35 * scale], [y + 1.05 * scale, y + 1.34 * scale], color=color, lw=lw)
    for frac, width in [(0.85, 0.90), (0.60, 0.72), (0.37, 0.54)]:
        yi = y + frac * scale
        half = 0.35 * width * scale
        ax.plot([x + 0.35 * scale - half, x + 0.35 * scale + half], [yi, yi], color=color, lw=lw)
    ax.plot([x + 0.12 * scale, x + 0.28 * scale], [y, y], color=color, lw=lw)
    ax.plot([x + 0.42 * scale, x + 0.58 * scale], [y, y], color=color, lw=lw)


def draw_market_bars(ax, x, y, scale=1.0):
    widths = [0.14, 0.14, 0.14]
    heights = [0.45, 0.75, 1.05]
    colors = [PALETTE["grid_soft"], "#94b8ff", PALETTE["grid"]]
    for i, (width, height, color) in enumerate(zip(widths, heights, colors)):
        ax.add_patch(
            FancyBboxPatch(
                (x + i * 0.22 * scale, y),
                width * scale,
                height * scale,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=color,
                edgecolor="none",
                zorder=4,
            )
        )


def draw_server_cluster(ax, x, y, scale=1.0):
    rack_w = 0.24 * scale
    rack_h = 0.58 * scale
    rack_gap = 0.10 * scale
    for i in range(3):
        xi = x + i * (rack_w + rack_gap)
        rounded_box(
            ax,
            xi,
            y,
            rack_w,
            rack_h,
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["compute"],
            linewidth=1.2,
            radius=0.04 * scale,
            zorder=4,
        )
        for j in range(3):
            yj = y + rack_h - 0.12 * scale - j * 0.14 * scale
            ax.add_patch(Circle((xi + 0.05 * scale, yj), 0.014 * scale, facecolor=PALETTE["compute"], edgecolor="none", zorder=5))
            ax.plot(
                [xi + 0.09 * scale, xi + rack_w - 0.04 * scale],
                [yj, yj],
                color=PALETTE["compute"],
                lw=1.05,
                alpha=0.8,
                zorder=5,
            )

    chip_x = x + 0.87 * scale
    chip_y = y + 0.10 * scale
    chip_w = 0.30 * scale
    chip_h = 0.30 * scale
    ax.add_patch(
        FancyBboxPatch(
            (chip_x, chip_y),
            chip_w,
            chip_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=PALETTE["compute_soft"],
            edgecolor=PALETTE["compute"],
            linewidth=1.2,
            zorder=4,
        )
    )
    for offset in [0.03, 0.09, 0.15, 0.21]:
        ax.plot([chip_x + offset * scale, chip_x + offset * scale], [chip_y - 0.05 * scale, chip_y], color=PALETTE["compute"], lw=1.0, zorder=4)
        ax.plot([chip_x + offset * scale, chip_x + offset * scale], [chip_y + chip_h, chip_y + chip_h + 0.05 * scale], color=PALETTE["compute"], lw=1.0, zorder=4)
        ax.plot([chip_x - 0.05 * scale, chip_x], [chip_y + offset * scale, chip_y + offset * scale], color=PALETTE["compute"], lw=1.0, zorder=4)
        ax.plot([chip_x + chip_w, chip_x + chip_w + 0.05 * scale], [chip_y + offset * scale, chip_y + offset * scale], color=PALETTE["compute"], lw=1.0, zorder=4)
    ax.plot(
        [chip_x + 0.06 * scale, chip_x + 0.14 * scale, chip_x + 0.22 * scale],
        [chip_y + 0.10 * scale, chip_y + 0.19 * scale, chip_y + 0.12 * scale],
        color=PALETTE["compute"],
        lw=1.8,
        zorder=5,
    )


def draw_battery(ax, x, y, scale=1.0):
    body_w = 0.62 * scale
    body_h = 1.05 * scale
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            body_w,
            body_h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["bess"],
            linewidth=1.6,
            zorder=4,
        )
    )
    ax.add_patch(Rectangle((x + 0.22 * scale, y + body_h), 0.18 * scale, 0.08 * scale, facecolor=PALETTE["bess"], edgecolor="none", zorder=4))
    for i, width in enumerate([0.16, 0.28, 0.40]):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.10 * scale, y + 0.16 * scale + i * 0.24 * scale),
                width * scale,
                0.12 * scale,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=PALETTE["bess_soft"],
                edgecolor="none",
                zorder=5,
            )
        )
    arrow(ax, (x + 0.85 * scale, y + 0.22 * scale), (x + 0.85 * scale, y + 0.86 * scale), color=PALETTE["bess"], lw=1.6, mutation_scale=14)
    arrow(ax, (x + 1.05 * scale, y + 0.88 * scale), (x + 1.05 * scale, y + 0.24 * scale), color=PALETTE["bess"], lw=1.6, mutation_scale=14)


def draw_optimizer_icon(ax, x, y, scale=1.0):
    center = (x + 0.34 * scale, y + 0.34 * scale)
    ax.add_patch(Circle(center, 0.17 * scale, facecolor=PALETTE["control_soft"], edgecolor=PALETTE["control"], linewidth=1.4, zorder=4))
    ax.add_patch(Circle(center, 0.06 * scale, facecolor=PALETTE["control"], edgecolor="none", zorder=5))
    nodes = [
        (center[0], center[1] + 0.31 * scale),
        (center[0] + 0.27 * scale, center[1] + 0.13 * scale),
        (center[0] + 0.23 * scale, center[1] - 0.20 * scale),
        (center[0] - 0.23 * scale, center[1] - 0.20 * scale),
        (center[0] - 0.27 * scale, center[1] + 0.13 * scale),
    ]
    for node in nodes:
        ax.plot([center[0], node[0]], [center[1], node[1]], color=PALETTE["control"], lw=1.15, zorder=4)
        ax.add_patch(Circle(node, 0.04 * scale, facecolor=PALETTE["control"], edgecolor="none", zorder=5))


def draw_poi_icon(ax, x, y, scale=1.0):
    ax.add_patch(Circle((x + 0.15 * scale, y + 0.36 * scale), 0.07 * scale, facecolor=PALETTE["poi"], edgecolor="none", zorder=4))
    ax.add_patch(Circle((x + 0.48 * scale, y + 0.56 * scale), 0.07 * scale, facecolor=PALETTE["poi"], edgecolor="none", zorder=4))
    ax.add_patch(Circle((x + 0.80 * scale, y + 0.36 * scale), 0.07 * scale, facecolor=PALETTE["poi"], edgecolor="none", zorder=4))
    ax.plot([x + 0.22 * scale, x + 0.41 * scale], [y + 0.40 * scale, y + 0.53 * scale], color=PALETTE["poi"], lw=1.8, zorder=4)
    ax.plot([x + 0.54 * scale, x + 0.73 * scale], [y + 0.53 * scale, y + 0.40 * scale], color=PALETTE["poi"], lw=1.8, zorder=4)
    ax.plot([x + 0.15 * scale, x + 0.80 * scale], [y + 0.20 * scale, y + 0.20 * scale], color=PALETTE["poi"], lw=2.0, zorder=4)
    ax.plot([x + 0.48 * scale, x + 0.48 * scale], [y + 0.20 * scale, y + 0.49 * scale], color=PALETTE["poi"], lw=2.0, zorder=4)


def draw_scenario_stack(ax, x, y, scale=1.0):
    offsets = [(0.12, 0.12), (0.06, 0.06), (0.0, 0.0)]
    for i, (dx, dy) in enumerate(offsets):
        card = FancyBboxPatch(
            (x + dx * scale, y + dy * scale),
            0.88 * scale,
            1.05 * scale,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=PALETTE["white"] if i < 2 else PALETTE["scenario_fill"],
            edgecolor=PALETTE["scenario"],
            linewidth=1.1,
            zorder=3 + i,
            alpha=0.95,
        )
        ax.add_patch(card)
    for yy in [0.78, 0.60, 0.42]:
        ax.plot([x + 0.14 * scale, x + 0.68 * scale], [y + yy * scale, y + yy * scale], color=PALETTE["scenario"], lw=1.2, alpha=0.8, zorder=6)
    ax.add_patch(Circle((x + 0.72 * scale, y + 0.74 * scale), 0.08 * scale, facecolor=PALETTE["scenario_soft"], edgecolor=PALETTE["scenario"], linewidth=1.0, zorder=6))
    ax.plot([x + 0.72 * scale, x + 0.72 * scale], [y + 0.70 * scale, y + 0.78 * scale], color=PALETTE["scenario"], lw=1.2, zorder=7)
    ax.plot([x + 0.68 * scale, x + 0.76 * scale], [y + 0.74 * scale, y + 0.74 * scale], color=PALETTE["scenario"], lw=1.2, zorder=7)


def draw_objective_chip(ax, x, y, w, title, subtitle, facecolor):
    rounded_box(
        ax,
        x,
        y,
        w,
        0.54,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0.0,
        radius=0.18,
        zorder=4,
    )
    ax.text(x + w / 2.0, y + 0.33, title, ha="center", va="center", fontsize=8.8, color=PALETTE["text"], weight="bold", zorder=5)
    ax.text(x + w / 2.0, y + 0.15, subtitle, ha="center", va="center", fontsize=8.3, color=PALETTE["muted"], zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Main site boundary.
    rounded_box(
        ax,
        2.55, # x
        0.72, # y
        8.95, # width
        6.52, # height
        facecolor=PALETTE["campus_fill"],
        edgecolor=PALETTE["campus_edge"],
        linewidth=1.8,
        radius=0.24,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    pill(ax, 5.20, 0.80, 3.50, 0.56, "Co-located data center site", facecolor=PALETTE["white"], edgecolor=PALETTE["campus_edge"], color=PALETTE["text"], fontsize=14.0)

    # Left external card: grid and markets.
    rounded_box(
        ax,
        0.36,
        1.02,
        1.92,
        5.42,
        facecolor=PALETTE["grid_fill"],
        edgecolor=PALETTE["grid"],
        linewidth=1.6,
        radius=0.20,
        zorder=2,
        shadow=True,
    )
    ax.text(1.32, 6.35, "Grid and\nmarkets", ha="center", va="top", fontsize=13.0, color=PALETTE["grid"], weight="bold", zorder=5)
    draw_grid_tower(ax, 0.78, 4.50, scale=0.98, color=PALETTE["grid"])
    draw_market_bars(ax, 1.46, 4.58, scale=0.82)
    ax.text(1.32, 4.24, "Interconnection\nlimits", ha="center", va="center", fontsize=9.8, color=PALETTE["text"], zorder=5)
    pill(ax, 0.58, 3.47, 1.48, 0.38, r"Energy  $\lambda_t^E$", facecolor=PALETTE["market_soft"], color=PALETTE["grid"], fontsize=8.8)
    pill(ax, 0.58, 2.92, 1.48, 0.38, r"Reserve  $\lambda_t^{Res}$", facecolor=PALETTE["market_soft"], color=PALETTE["grid"], fontsize=8.5)
    pill(ax, 0.58, 2.37, 1.48, 0.38, r"Flex-ramp  $\lambda_t^{FRP}$", facecolor=PALETTE["market_soft"], color=PALETTE["grid"], fontsize=8.2)
    pill(ax, 0.58, 1.65, 1.48, 0.42, r"$D_t \leq \bar{D}$", facecolor=PALETTE["white"], edgecolor=PALETTE["grid"], color=PALETTE["text"], fontsize=8.8)
    pill(ax, 0.58, 1.16, 1.48, 0.42, r"$|\Delta D_t| \leq \bar{\Delta}$", facecolor=PALETTE["white"], edgecolor=PALETTE["grid"], color=PALETTE["text"], fontsize=8.3)

    # Right external card: uncertainty.
    rounded_box(
        ax,
        11.73,
        1.42,
        1.90,
        4.98,
        facecolor=PALETTE["scenario_fill"],
        edgecolor=PALETTE["scenario"],
        linewidth=1.6,
        radius=0.20,
        zorder=2,
        shadow=True,
    )
    ax.text(12.68, 6.05, "Scenario\nset  $\\mathcal{S}$", ha="center", va="top", fontsize=13.0, color=PALETTE["scenario"], weight="bold", zorder=5)
    draw_scenario_stack(ax, 12.02, 3.57, scale=1.20)
    ax.text(12.68, 3.12, "Ancillary-service\ndeployment and\nenergy settlement", ha="center", va="center", fontsize=9.5, color=PALETTE["text"], zorder=5)
    pill(ax, 11.96, 1.78, 1.44, 0.44, "Worst-case\nrealization", facecolor=PALETTE["white"], edgecolor=PALETTE["scenario"], color=PALETTE["text"], fontsize=8.5)

    # Optimizer card.
    rounded_box(
        ax,
        4.05,
        5.45,
        5.95,
        1.38,
        facecolor=PALETTE["control_fill"],
        edgecolor=PALETTE["control"],
        linewidth=1.7,
        radius=0.22,
        zorder=3,
        shadow=True,
    )
    draw_optimizer_icon(ax, 4.34, 5.78, scale=0.90)
    ax.text(5.25, 6.55, "Robust day-ahead coordinator", ha="left", va="center", fontsize=14.0, color=PALETTE["control"], weight="bold", zorder=5)
    ax.text(5.25, 6.30, "One schedule shared across all scenarios", ha="left", va="center", fontsize=10.1, color=PALETTE["muted"], zorder=5)
    ax.text(4.50, 5.74, "Worst-case net value:", ha="left", va="center", fontsize=9.6, color=PALETTE["text"], weight="bold", zorder=5)
    draw_objective_chip(ax, 6.40, 5.56, 0.96, "+ Revenue", r"$R^{Res},\;R^{FRP}$", PALETTE["revenue_fill"])
    draw_objective_chip(ax, 7.43, 5.56, 0.88, "- Energy", r"$C_s^E$", PALETTE["energy_fill"])
    draw_objective_chip(ax, 8.38, 5.56, 0.72, "- QoS", r"$C^{job}$", PALETTE["qos_fill"])
    draw_objective_chip(ax, 9.17, 5.56, 0.76, "- Wear", r"$C_s^{BESS}$", PALETTE["deg_fill"])

    # Compute block.
    rounded_box(
        ax,
        3.16,
        3.50,
        3.55,
        1.62,
        facecolor=PALETTE["compute_fill"],
        edgecolor=PALETTE["compute"],
        linewidth=1.6,
        radius=0.18,
        zorder=3,
        shadow=True,
    )
    draw_server_cluster(ax, 3.42, 3.86, scale=0.98)
    ax.text(4.92, 4.90, "Flexible compute", ha="center", va="center", fontsize=13.0, color=PALETTE["compute"], weight="bold", zorder=5)
    ax.text(5.42, 4.52, "Fixed IT load, schedulable jobs,\nand DVFS coordination", ha="center", va="center", fontsize=9.6, color=PALETTE["text"], zorder=5)
    pill(ax, 5.04, 3.95, 0.84, 0.34, r"Fixed $\hat{P}^o$", facecolor=PALETTE["compute_soft"], color=PALETTE["compute"], fontsize=8.7)
    pill(ax, 4.46, 3.55, 0.96, 0.34, r"Jobs  $w_{j,t}$", facecolor=PALETTE["compute_soft"], color=PALETTE["compute"], fontsize=8.7)
    pill(ax, 5.52, 3.55, 1.15, 0.34, r"DVFS  $a_t$", facecolor=PALETTE["compute_soft"], color=PALETTE["compute"], fontsize=8.7)

    # BESS block.
    rounded_box(
        ax,
        7.34,
        3.50,
        3.55,
        1.62,
        facecolor=PALETTE["bess_fill"],
        edgecolor=PALETTE["bess"],
        linewidth=1.6,
        radius=0.18,
        zorder=3,
        shadow=True,
    )
    draw_battery(ax, 7.62, 3.78, scale=0.92)
    ax.text(9.56, 4.73, "Co-located BESS", ha="center", va="center", fontsize=13.0, color=PALETTE["bess"], weight="bold", zorder=5)
    ax.text(9.76, 4.42, "Dispatch, state of charge,\nand reserve / ramp offers", ha="center", va="center", fontsize=9.6, color=PALETTE["text"], zorder=5)
    pill(ax, 8.35, 3.55, 1.13, 0.34, r"$P_{\alpha,t}^B,\;P_{\beta,t}^B$", facecolor=PALETTE["bess_soft"], color=PALETTE["bess"], fontsize=8.3)
    pill(ax, 9.60, 3.55, 1.20, 0.34, r"$E_t^B,\;P_t^{B,R/\uparrow/\downarrow}$", facecolor=PALETTE["bess_soft"], color=PALETTE["bess"], fontsize=7.8)

    # POI / bus block.
    rounded_box(
        ax,
        5.18,
        1.67,
        3.58,
        1.28,
        facecolor=PALETTE["poi_fill"],
        edgecolor=PALETTE["poi"],
        linewidth=1.6,
        radius=0.17,
        zorder=3,
        shadow=True,
    )
    draw_poi_icon(ax, 5.46, 1.98, scale=0.82)
    ax.text(7.08, 2.58, "POI / grid interface", ha="center", va="center", fontsize=13.0, color=PALETTE["poi"], weight="bold", zorder=5)
    ax.text(7.48, 2.30, r"Net load $D_t$ and ramp $\Delta D_t$", ha="center", va="center", fontsize=10.0, color=PALETTE["text"], zorder=5)
    pill(ax, 6.22, 1.82, 1.10, 0.34, r"Load limit  $\bar{D}$", facecolor=PALETTE["white"], edgecolor=PALETTE["poi"], color=PALETTE["poi"], fontsize=8.3)
    pill(ax, 7.44, 1.82, 1.28, 0.34, r"Ramp limit  $\bar{\Delta}$", facecolor=PALETTE["white"], edgecolor=PALETTE["poi"], color=PALETTE["poi"], fontsize=8.1)

    # Arrows and labels.
    arrow(ax, (6.95, 5.43), (5.18, 4.96), color=PALETTE["control"], lw=1.8, linestyle=(0, (5, 3)), mutation_scale=14)
    arrow(ax, (7.12, 5.43), (8.86, 4.96), color=PALETTE["control"], lw=1.8, linestyle=(0, (5, 3)), mutation_scale=14)
    arrow_label(ax, 5.88, 5.25, "schedule", color=PALETTE["control"], fontsize=8.2, facecolor=PALETTE["bg"])
    arrow_label(ax, 8.19, 5.25, "set-points", color=PALETTE["control"], fontsize=8.2, facecolor=PALETTE["bg"])

    arrow(ax, (4.94, 3.46), (6.22, 2.86), color=PALETTE["line"], lw=2.2, mutation_scale=14)
    arrow(ax, (8.90, 3.46), (8.05, 2.87), color=PALETTE["bess"], lw=2.2, style="<|-|>", mutation_scale=14)
    arrow_label(ax, 5.64, 3.06, r"$\hat{P}_t$", color=PALETTE["line"], fontsize=9.0, facecolor=PALETTE["bg"])
    arrow_label(ax, 8.72, 3.06, r"$P_{\alpha,t}^B - P_{\beta,t}^B$", color=PALETTE["bess"], fontsize=8.3, facecolor=PALETTE["bg"])

    arrow(ax, (5.14, 2.26), (2.28, 3.20), color=PALETTE["line"], lw=2.3, mutation_scale=15)
    arrow_label(ax, 2.98, 3.14, r"$D_t,\;\Delta D_t$", color=PALETTE["line"], fontsize=8.9, facecolor=PALETTE["bg"])

    arrow(ax, (2.28, 4.10), (4.03, 6.17), color=PALETTE["grid"], lw=1.8, linestyle=(0, (5, 3)), mutation_scale=14)
    arrow_label(ax, 3.56, 6.02, "prices", color=PALETTE["grid"], fontsize=8.7, facecolor=PALETTE["bg"])

    arrow(ax, (11.72, 4.50), (10.02, 6.10), color=PALETTE["scenario"], lw=1.8, linestyle=(0, (5, 3)), mutation_scale=14)
    arrow_label(ax, 10.32, 6.05, r"$\beta_s$", color=PALETTE["scenario"], fontsize=9.0, facecolor=PALETTE["bg"])

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08, facecolor=PALETTE["bg"])
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.08, dpi=320, facecolor=PALETTE["bg"])
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - debug path for batch rendering
        raise SystemExit(f"Figure generation failed: {exc}")
