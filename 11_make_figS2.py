#!/usr/bin/env python3
"""Fig. S2 - characteristics of severe PEP events.

Reads data/tc_events_full.csv and figure_data/figS2_spatial.npz.
Output: figures/FigS2_severe_PEP.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _style import (natural_save, label_panel, add_china, C_PER, C_NPER, C_ACCENT,
                    FS_LEGEND, FS_PANEL, FS_ANNOT, FS_TICK, FS_GRID, FS_CBAR_LAB,
                    FS_CBAR_TICK)
from config import DATA_DIR
FD = Path(__file__).resolve().parents[1] / "figure_data"
# house map extent (same as FigS3/S8 maps); (d) carries the bottom-right SCS inset
EXTENT = [95, 134, 16, 50]
# plan Fig.S2a: yellow (primary PEP) deepened one step for print contrast
ACCENT_DEEP = "#CC8A00"


def _scs_inset(ax, proj):
    """Bottom-right South China Sea inset, hugging the map's bottom-right corner
    (same construction as FigS3: GeoAxes settles the inset box at its data aspect
    (23°/17°), NARROWER than the requested rect — anchor "SE" + y0 just inside
    the frame tucks it into the corner)."""
    w, h, y0 = 0.245, 0.345, 0.005
    axins = ax.inset_axes([1 - w, y0, w, h], projection=proj)
    axins.set_anchor("SE")
    add_china(axins)
    axins.set_extent([105, 122, 2, 25], crs=proj)
    axins.tick_params(length=0, labelsize=0)
    for side in ("top", "left"):                       # inner border only …
        axins.spines[side].set_edgecolor("0.35"); axins.spines[side].set_linewidth(0.5)
    for side in ("right", "bottom"):                   # … outer edges merge with the map
        axins.spines[side].set_visible(False)
    return axins


def knn(rt, flag, grid, k=120, seed=0):
    rng = np.random.default_rng(seed)
    xs, pm = [], []
    for g in grid:
        idx = np.argsort(np.abs(np.asarray(rt) - g))[:k]
        xs.append(g); pm.append(np.asarray(flag, float)[idx].mean())
    return np.array(xs), np.array(pm)


def main():
    df = pd.read_csv(DATA_DIR / "tc_events_full.csv").dropna(
        subset=["duration_days", "accumulation_Ae_mm", "RT_e_hours"])
    Dp90 = float(np.quantile(df.duration_days, 0.90))
    Ap90 = float(np.quantile(df.accumulation_Ae_mm, 0.90))
    Ap95 = float(np.quantile(df.accumulation_Ae_mm, 0.95))
    primary = (df.duration_days > Dp90) & (df.accumulation_Ae_mm > Ap90)
    severe = (df.duration_days > Dp90) & (df.accumulation_Ae_mm > Ap95)


    # right; 6.9 gave a 185.6 mm tight bbox, 6.75 lands back under 183 mm
    fig = plt.figure(figsize=(6.75, 6.4))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.97, top=0.95, bottom=0.07,
                          width_ratios=[1, 1.14], wspace=0.15, hspace=0.16)
    LAB_X = -0.08  # panel-label x offset (axes fraction) for b/d

    # (a) D-A scatter
    a = fig.add_subplot(gs[0, 0])
    a.scatter(df.duration_days[~primary], df.accumulation_Ae_mm[~primary],
              s=8, c=C_NPER, alpha=0.35, edgecolor="none", rasterized=True, label="non-PEP")
    a.scatter(df.duration_days[primary], df.accumulation_Ae_mm[primary],
              s=16, c=ACCENT_DEEP, alpha=0.8, edgecolor="none", label="primary PEP")
    # plan Fig.S2a: severe (orange) keeps its light stroke so overlaps separate
    a.scatter(df.duration_days[severe], df.accumulation_Ae_mm[severe],
              s=22, c=C_PER, alpha=0.9, edgecolor="w", lw=0.4, label="severe PEP")
    # plan Fig.S2a: threshold lines thinned; all three kept, tags staggered
    a.axvline(Dp90, color="k", ls="--", lw=0.6); a.axhline(Ap90, color="k", ls=":", lw=0.6)
    a.axhline(Ap95, color=C_PER, ls=":", lw=0.7)
    a.text(Dp90 + 0.1, 0.04, "P90(D)", fontsize=FS_ANNOT, color="k",
           transform=a.get_xaxis_transform())
    a.text(0.98, Ap90 - 30, "P90(A)", fontsize=FS_ANNOT, color="k",
           transform=a.get_yaxis_transform(), ha="right")
    a.text(0.98, Ap95 + 10, "P95(A)", fontsize=FS_ANNOT, color=C_PER,
           transform=a.get_yaxis_transform(), ha="right")
    a.set_xlabel("Duration $D_e$ (days)"); a.set_ylabel("Accumulation $A_e$ (mm)")
    a.legend(fontsize=FS_LEGEND, frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="upper left",
             bbox_to_anchor=(-0.01, 1.0), handletextpad=0.1)
    label_panel(a, "a")

    # (b) RT violin 3 groups
    b = fig.add_subplot(gs[0, 1])
    data = [df.RT_e_hours[~primary], df.RT_e_hours[primary], df.RT_e_hours[severe]]
    pos = [1, 2, 3]; cols = [C_NPER, C_ACCENT, C_PER]
    vp = b.violinplot(data, positions=pos, widths=0.7, showmedians=False, showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        # plan Fig.S2b: softer fill, crisp same-colour outline
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(0.30); body.set_linewidth(0.9)
    for p, d, col in zip(pos, data, cols):
        b.hlines(float(np.median(d)), p - 0.3, p + 0.3, colors=col, lw=1.7, zorder=4)
        b.text(p, np.max(d) * 1.03, f"{np.median(d):.0f} h", ha="center",
               fontsize=FS_ANNOT, color=col)
    b.set_xticks(pos); b.set_xticklabels([f"non-PEP\n({len(data[0])})",
                                          f"primary PEP\n({len(data[1])})",
                                          f"severe PEP\n({len(data[2])})"], fontsize=FS_TICK)
    b.margins(y=0.12)
    b.set_ylabel("$RT_e$ (h)")
    label_panel(b, "b", x=LAB_X)

    # (c) P(PEP|RT) primary vs severe
    c = fig.add_subplot(gs[1, 0])
    rt = df.RT_e_hours.values
    grid = np.arange(0, 301, 15)
    xs1, p1 = knn(rt, primary.values.astype(int), grid, seed=1)
    xs2, p2 = knn(rt, severe.values.astype(int), grid, seed=2)
    # plan Fig.S2c: colour + line style both separate the two curves
    c.plot(xs1, p1 * 100, color=ACCENT_DEEP, lw=1.5, ls="-", label="primary PEP")
    c.plot(xs2, p2 * 100, color=C_PER, lw=1.5, ls="--", label="severe PEP")
    c.set_xlabel("$RT_e$ (h)"); c.set_ylabel("P(PEP | RT) (%)")
    c.legend(fontsize=FS_LEGEND, frameon=False)
    label_panel(c, "c")

    # (d) severe spatial footprint
    sp = np.load(FD / "figS2_spatial.npz", allow_pickle=True)
    lat = sp["lat"]; lon = sp["lon"]
    LON, LAT = np.meshgrid(lon, lat)
    proj = ccrs.PlateCarree()
    d = fig.add_subplot(gs[1, 1], projection=proj)
    # NO set_box_aspect: with the colourbar divider it flips the axes to
    # adjustable="datalim" and can stretch the drawn extent (cf. figS8


    d.set_anchor("C")
    rain = np.ma.masked_where(sp["per_rain"] <= 0, sp["per_rain"])
    vmax = 1500.0  # severe-PEP rain upper bound (mm)
    im = d.pcolormesh(LON, LAT, rain, cmap="YlOrRd", vmin=0, vmax=vmax,
                      shading="auto", transform=proj, zorder=2)
    add_china(d); d.set_extent(EXTENT, crs=proj)
    gl = d.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False; gl.right_labels = False; gl.bottom_labels = True; gl.left_labels = True
    gl.xlocator = mticker.MultipleLocator(15)
    gl.xlabel_styles = gl.ylabel_styles = {"fontsize": FS_GRID}
    _scs_inset(d, proj)                     # bottom-right SCS mini-map (house mode)
    cax = make_axes_locatable(d).append_axes("right", size="5%", pad=0.04, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=2)
    cb.set_label("Severe PEP rain (mm)", fontsize=FS_CBAR_LAB)
    cb.locator = mticker.MultipleLocator(500)      # plan: neat even ticks
    cb.update_ticks()
    # figS3/figS8 pattern: divider slots track the CELL, not the aspect-settled

    # beside the applied map box.
    fig.canvas.draw()
    fd = d.get_position(); cx = cax.get_position(); pad = cx.x0 - fd.x1
    d.set_axes_locator(None); cax.set_axes_locator(None)
    fig.canvas.draw()
    mf = d.get_position()
    cax.set_position([mf.x1 + pad, mf.y0, cx.width, mf.height])
    # align (d) label x with (b): d is a centered cartopy axes (inset left edge),
    # so place its label at b's figure-x after layout is finalized.
    fig.canvas.draw()
    pos_b = b.get_position(); pos_d = d.get_position()
    x_d = (pos_b.x0 + LAB_X * pos_b.width - pos_d.x0) / pos_d.width
    d.text(x_d, 1.02, "(d)", transform=d.transAxes,
           fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    return natural_save(fig, "FigS2_severe_PEP")


if __name__ == "__main__":
    main()
