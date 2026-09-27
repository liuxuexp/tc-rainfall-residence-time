#!/usr/bin/env python3
"""Fig. 6 - conceptual model schematic (single panel, no data dependency).

Output: figures/Fig6_conceptual_model.png (183 mm).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import save, FIG_DIR, C_DYN, C_THE

C_RT  = "#6A4C93"      # purple – residence-time bridge (conceptual; _style.C_RT is data-blue)
C_PER = "#1a1a1a"      # neutral outcome box (_style.C_PER vermilion is for PEP data points)


def box(ax, x, y, w, h, text, fc, ec, fs=8, weight="normal", tc="#111"):

    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, lw=lw, color=color, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.2); ax.axis("off")


    ax.text(5, 8.90, "Conceptual model — how TC residence time\nproduces Persistent Extreme Rainfall",
            ha="center", fontsize=9, weight="normal")
    ax.text(0.15, 6.55, "Dynamic pathway", color=C_DYN, weight="bold", fontsize=9.5)
    ax.text(0.15, 6.30, "(controls DURATION)", color=C_DYN, fontsize=8, style="italic")
    ax.text(9.85, 6.55, "Thermodynamic pathway", color=C_THE, weight="bold", fontsize=9.5, ha="right")
    ax.text(9.85, 6.30, "(controls ACCUMULATION)", color=C_THE, fontsize=8, style="italic", ha="right")


    # bold reserved for the pathway ends; all-casps trimmed to the two outcomes
    box(ax, 0.2, 5.3, 2.7, 0.85, "Persistent circulation anomaly\n(WPSH shift / blocking /\ntrack rotation)",
        "#eaf2fb", C_DYN, 8)
    box(ax, 0.2, 4.1, 2.7, 0.85, "Weak / rotated\nsteering flow", "#eaf2fb", C_DYN, 8)
    box(ax, 0.2, 2.6, 2.7, 1.0, "Long TC residence\ntime (RT)", "#efe9f8", C_RT, 8.5, "bold")
    box(ax, 0.2, 1.3, 2.7, 0.85, "Slow rainband\ndisplacement", "#eaf2fb", C_DYN, 8)
    box(ax, 0.2, 0.25, 2.7, 0.8, "Long rainfall\nDURATION", "#d7e6f7", C_DYN, 8.5, "bold")
    for (y1, y2) in [(5.3, 4.95), (4.1, 3.6), (2.6, 2.15), (1.3, 1.05)]:
        arrow(ax, 1.55, y1, 1.55, y2, C_DYN)

    # RIGHT thermo chain
    box(ax, 7.1, 5.3, 2.7, 0.85, "Warming atmosphere\n(Clausius–Clapeyron)", "#fbeae9", C_THE, 8)
    box(ax, 7.1, 4.1, 2.7, 0.85, "Higher PW /\nstronger IVT", "#fbeae9", C_THE, 8)
    box(ax, 7.1, 2.6, 2.7, 1.0, "Greater moisture\nsupply &\nconvergence (MFC)", "#fbeae9", C_THE, 8)
    box(ax, 7.1, 1.3, 2.7, 0.85, "Repeated convective\nregeneration", "#fbeae9", C_THE, 8)
    box(ax, 7.1, 0.25, 2.7, 0.8, "Large event\nACCUMULATION", "#f6d8d6", C_THE, 8.5, "bold")
    for (y1, y2) in [(5.3, 4.95), (4.1, 3.6), (2.6, 2.15), (1.3, 1.05)]:
        arrow(ax, 8.45, y1, 8.45, y2, C_THE)


    box(ax, 3.6, 1.9, 2.8, 1.2, "Persistent Extreme\nRainfall (PEP)\n$D>P_{90}$ and $A>P_{90}$",
        "#f3f3f3", C_PER, 9.5, "bold")
    box(ax, 3.6, 0.35, 2.8, 0.8, "Flood risk amplified", "#f3f3f3", C_PER, 8.5)
    arrow(ax, 5.0, 1.9, 5.0, 1.15, C_PER)
    arrow(ax, 2.9, 0.65, 3.6, 2.1, C_DYN, 1.8)
    arrow(ax, 7.1, 0.65, 6.4, 2.1, C_THE, 1.8)

    # RT bridge label
    box(ax, 3.7, 4.55, 2.6, 0.95, "Residence time —\nthe dynamical bridge\n(circulation → rainfall org.)",
        "#fbfbfd", C_RT, 8, "bold", tc=C_RT)
    arrow(ax, 2.9, 3.1, 3.8, 4.55, C_RT, 1.5)


    # re-flowed as three clearly separated regular-weight lines that stop clear of
    # the side boxes' top edge (short lines run inside the centre corridor; the
    # wide last line rides above the box-top plane, clear of the side subtitles)
    ax.text(5.0, 8.15, "Empirical anchors (1960–2024)", ha="center", fontsize=8.5,
            weight="normal", color="#444")
    ax.text(5.0, 7.96,
            "PEP = 41/1132 (3.6%) · RT PEP 150 h vs 54 h\n"
            "Extreme−Short RT: IVT +40, MFC +1.4, TCWV +1.6\n"
            "post-1979: TCWV +0.50/dec (p=.004),\nthermodynamic Δq·V̄ +0.52/dec (p=.002)",
            ha="center", va="top", fontsize=8, color="#333", linespacing=1.25)

    info = save(fig, "Fig6_conceptual_model", column="double", checks=True)
    plt.close(fig)
    return info


if __name__ == "__main__":
    main()
