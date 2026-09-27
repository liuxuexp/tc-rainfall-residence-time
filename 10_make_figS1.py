#!/usr/bin/env python3
"""Fig. S1 - threshold-sensitivity matrices (3x3 per panel, 2x2 grid).

Reads figure_data/figS1.csv (= Table S1).
Output: figures/FigS1_threshold_sensitivity.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import ImageGrid

from _style import (natural_save, label_panel, FS_ANNOT, FS_TITLE, FS_TICK,
                    FS_AXIS, FS_CBAR_TICK)
FD = Path(__file__).resolve().parents[1] / "figure_data"

DQ = [85, 90, 95]
QLAB = ["P85", "P90", "P95"]
# plan Fig.S1(c,d): in-cell body text ~9 pt (all four matrices, one size)
CELL_FS = 9.0


def mat(df, col):
    z = np.full((3, 3), np.nan)
    for _, r in df.iterrows():
        i = DQ.index(r.D_q); j = DQ.index(r.A_q)
        z[i, j] = r[col]
    return z


def annot_color(val, norm, cmap):
    """Dark or white text by the cell's rendered luminance (any cmap/norm)."""
    if not np.isfinite(val):
        return "0.1"
    rgba = cmap(norm(val))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.55 else "0.1"


def add_cbar(ax, im):
    """Colorbar on the ImageGrid's dedicated cax (no axes-space stealing)."""
    cb = ax.cax.colorbar(im)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=2)
    cb.outline.set_linewidth(0.5)          # plan: clean boundary, no heavy frame
    return cb


def main():
    df = pd.read_csv(FD / "figS1.csv")


    # ImageGrid keeps the matrices SQUARE (aspect=True with 3×3 data), so the

    # absolute and the canvas/rect were tuned empirically around it. The pads
    # cannot go much tighter: the horizontal gap hosts the left panels' colourbar
    # tick numbers (~7.4 mm) and the vertical gap hosts the bottom row's title

    fig = plt.figure(figsize=(6.6, 6.0))
    grid = ImageGrid(
        fig, [0.060, 0.09, 0.91, 0.83], nrows_ncols=(2, 2),
        axes_pad=(0.42, 0.36),
        cbar_mode="each", cbar_location="right",
        cbar_pad=0.02, cbar_size="4%",
        label_mode="L",   # tick labels only on left column + bottom row
        share_all=True,   # one shared D×A grid across all four panels
    )
    a, b, c, d = grid

    # (a) PEP count + pct annotation
    z = mat(df, "n_PER")
    im = a.imshow(z, cmap="YlOrRd", origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            r = df[(df.D_q == DQ[i]) & (df.A_q == DQ[j])].iloc[0]
            a.text(j, i, f"{int(r.n_PER)}\n({r.pct}%)", ha="center", va="center",
                   fontsize=CELL_FS, color=annot_color(z[i, j], im.norm, im.cmap))
    a.set_title("PEP event count", fontsize=FS_TITLE)
    add_cbar(a, im)
    label_panel(a, "a")

    # (b) volume contribution
    z = mat(df, "vol_contrib")
    im = b.imshow(z, cmap="YlGnBu", origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            b.text(j, i, f"{z[i, j]:.1f}%", ha="center", va="center",
                   fontsize=CELL_FS, color=annot_color(z[i, j], im.norm, im.cmap))
    b.set_title("Rainfall-volume contribution", fontsize=FS_TITLE)
    add_cbar(b, im)
    # right-column letters hug their own matrix: default x=-0.14 drops them in

    label_panel(b, "b", x=-0.09)

    # (c) RT contrast  (cividis = RT semantic, matches main fig)
    zc = mat(df, "medianRT_PER") - mat(df, "medianRT_nonPER")
    im = c.imshow(zc, cmap="cividis", origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            # plan Fig.S1c: unit separated from the value by a space
            c.text(j, i, f"+{zc[i, j]:.0f} h", ha="center", va="center",
                   fontsize=CELL_FS, color=annot_color(zc[i, j], im.norm, im.cmap))
    c.set_title("median RT: PEP − non-PEP (h)", fontsize=FS_TITLE)
    add_cbar(c, im)
    label_panel(c, "c")

    # (d) corr(RT_local, D) within PEP  (BrBG diverging around the 0.8 baseline)
    zd = mat(df, "corr_RTD")
    vm = max(0.001, np.nanmax(np.abs(zd - 0.8)))
    norm = TwoSlopeNorm(vmin=0.8 - vm, vcenter=0.8, vmax=0.8 + vm)
    im = d.imshow(zd, cmap="BrBG", norm=norm, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            v = zd[i, j]
            d.text(j, i, f"{v:.2f}" if np.isfinite(v) else "—",
                   ha="center", va="center", fontsize=CELL_FS,
                   color=annot_color(v, im.norm, im.cmap))
    d.set_title("corr(RT$_{local}$, D) within PEP", fontsize=FS_TITLE)
    add_cbar(d, im)
    label_panel(d, "d", x=-0.09)


    for ax in grid:
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"A{l}" for l in QLAB], fontsize=FS_TICK)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"D{l}" for l in QLAB], fontsize=FS_TICK)
    # axis titles only on the outer edge
    for ax in (c, d):
        ax.set_xlabel("Accumulation threshold", fontsize=FS_AXIS)
    for ax in (a, c):
        ax.set_ylabel("Duration threshold", fontsize=FS_AXIS)


    for ax in grid:
        ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False,
                                   edgecolor="k", lw=1.1, zorder=5))

    # Natural-width save: no journal column lock; grid fills the figure via rect.
    # ImageGrid is incompatible with bbox="tight", so use the standard save path.
    return natural_save(fig, "FigS1_threshold_sensitivity", tight=False)


if __name__ == "__main__":
    main()
