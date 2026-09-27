#!/usr/bin/env python3
"""Fig. S3 - influence-radius sensitivity (500 km baseline vs 400/600 km).

Reads data/si_radius.csv, figure_data/figS3_spatial.npz and the cached
radius grids in figure_data/_si_radius_cache/.
Output: figures/FigS3_radius_sensitivity.png.
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

from _style import (natural_save, label_panel, add_china, halo_text, C_PER, OKABE_ITO,
                    FS_PANEL, FS_AXIS, FS_TICK, FS_TITLE, FS_ANNOT, FS_LEGEND,
                    FS_GRID, FS_CBAR_LAB, FS_CBAR_TICK)
from config import DATA_DIR
FD = Path(__file__).resolve().parents[1] / "figure_data"
CACHE = FD / "_si_radius_cache"
EXTENT = [95, 134, 16, 50]
RADII = [400, 500, 600]


def rt_contrast_by_radius():
    """median RT_local(PEP) - median RT_local(non-PEP) per radius, via tc_id match."""
    base = pd.read_csv(DATA_DIR / "tc_events_full.csv")
    base_rt = base.groupby("tc_id")["RT_local_hours"].median()   # tc-level median RT
    out = {}
    for R in RADII:
        if R == 500:
            df = base[["tc_id", "PER_flag"]].rename(columns={"PER_flag": "PER"})
        else:
            z = np.load(CACHE / f"radius_{R}.npz", allow_pickle=True)
            df = pd.DataFrame({"tc_id": z["tc_id"], "PER": z["PER"]})
        rt = df["tc_id"].astype(int).map(base_rt)
        med_per = float(np.nanmedian(rt[df.PER == 1])) if (df.PER == 1).any() else np.nan
        med_nper = float(np.nanmedian(rt[df.PER == 0]))
        out[R] = med_per - med_nper
    return out


def _scs_inset(ax, proj):
    """Bottom-right South China Sea inset, hugging the map's bottom-right corner."""
    # GeoAxes settles the box at the inset's data aspect (23°/17°), NARROWER than

    # slit at the right frame and dipping the bottom below the map edge. Anchor
    # "SE" + y0 just inside the frame tucks it into the corner (user nudge

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


def main():
    rad = pd.read_csv(DATA_DIR / "si_radius.csv")
    sp = np.load(FD / "figS3_spatial.npz", allow_pickle=True)
    dRT = rt_contrast_by_radius()

    fig = plt.figure(figsize=(6.4, 8.2))


    # horizontal gaps opened back up): one gridspec per ROW so wspace is tuned

    # ~11 mm, mostly clear air between the left panel and the right map's OWN
    # latitude labels. Map-row cells stay near the map's data aspect
    # (39°lon/34°lat) so the maps FILL their cells (measured: GeoAxes renders

    # cell geometry alone controls the dead space).


    gs_rows = fig.add_gridspec(3, 1, left=0.11, right=0.94, top=0.95, bottom=0.06,
                               height_ratios=[0.877, 1.0, 1.0], hspace=0.26)
    row = [gs_rows[0].subgridspec(1, 2, wspace=0.145),
           gs_rows[1].subgridspec(1, 2, wspace=0.185),
           gs_rows[2].subgridspec(1, 2, wspace=0.185)]
    # reading order: a(r0c0) b(r0c1) c(r1c0); maps d(r1c1) e(r2c0) f(r2c1)

    x = np.arange(3); w = 0.38
    xlab = [f"{r}" for r in RADII]

    # (a) counts
    a = fig.add_subplot(row[0][0, 0])
    # plan Fig.S3a: blue total bars recede slightly so the PEP orange carries
    # the contrast; both get thin outlines (no value labels invented)
    a.bar(x - w / 2, rad.n_events, w, color=OKABE_ITO[2], alpha=0.85,
          edgecolor="0.25", linewidth=0.6, label="all TC events")
    a.bar(x + w / 2, rad.n_PER, w, color=C_PER,
          edgecolor="0.25", linewidth=0.6, label="PEP")
    a.set_xticks(x); a.set_xticklabels(xlab, fontsize=FS_TICK)
    a.set_xlabel("influence radius (km)", fontsize=FS_AXIS)
    a.set_ylabel("event count", fontsize=FS_AXIS)
    a.tick_params(labelsize=FS_TICK)
    a.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    label_panel(a, "a")

    # (b) thresholds (twin y)
    b = fig.add_subplot(row[0][0, 1])
    b.plot(x, rad.P90_D, "o-", color=OKABE_ITO[2], lw=1.4, label="P90(D) (d)")
    b.set_ylabel("P90(D) (days)", color=OKABE_ITO[2], fontsize=FS_AXIS, fontweight="bold")
    b.tick_params(axis="y", labelcolor=OKABE_ITO[2], labelsize=FS_TICK)
    b2 = b.twinx()
    b2.plot(x, rad.P90_A, "s-", color=C_PER, lw=1.4, label="P90(A) (mm)")
    b2.set_ylabel("P90(A) (mm)", color=C_PER, fontsize=FS_AXIS, fontweight="bold")
    b2.tick_params(axis="y", labelcolor=C_PER, labelsize=FS_TICK)
    # plan Fig.S3b: 500 km tick not separately over-bolded (plain, same as a)
    b.set_xticks(x); b.set_xticklabels(xlab, fontsize=FS_TICK)
    b.set_xlabel("influence radius (km)", fontsize=FS_AXIS)
    b.tick_params(axis="x", labelsize=FS_TICK)
    label_panel(b, "b")

    # (c) RT contrast
    c = fig.add_subplot(row[1][0, 0])
    vals = [dRT[R] for R in RADII]
    c.bar(x, vals, color=OKABE_ITO[6], width=0.55)
    for i, v in enumerate(vals):
        c.text(i, v + 1.5, f"{v:.0f} h", ha="center", fontsize=FS_ANNOT)
    c.set_xticks(x); c.set_xticklabels(xlab, fontsize=FS_TICK)
    c.set_xlabel("influence radius (km)", fontsize=FS_AXIS)
    # plan Fig.S3c: long ylabel stays in place as two lines, size follows ladder
    c.set_ylabel("Δ median $RT_{local}$\nPEP − non-PEP (h)", fontsize=FS_AXIS)
    c.margins(y=0.15)
    c.tick_params(labelsize=FS_TICK)
    label_panel(c, "c")


    proj = ccrs.PlateCarree()
    lat = sp["lat"]; lon = sp["lon"]
    LON, LAT = np.meshgrid(lon, lat)
    vmax = float(np.nanpercentile(sp["per_contrib_500"], 98))
    im = None
    map_axes = []
    map_cells = [row[1][0, 1], row[2][0, 0], row[2][0, 1]]        # d,e,f cells
    for k, R in enumerate(RADII):
        ax = fig.add_subplot(map_cells[k], projection=proj)       # d,e,f
        ax.set_anchor("N")   # top-anchor safety if the cell aspect drifts from the
                             # map's data aspect: slack stays at the bottom where the


        for t in (ax.title, ax._left_title, ax._right_title):
            t.set_visible(False)   # empty titles render nothing, but their NaN extents poison

        field = np.ma.masked_where(sp[f"per_contrib_{R}"] <= 0, sp[f"per_contrib_{R}"])
        im = ax.pcolormesh(LON, LAT, field, cmap="YlGnBu", vmin=0, vmax=vmax,
                           shading="auto", transform=proj, zorder=2)
        add_china(ax); ax.set_extent(EXTENT, crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
        gl.top_labels = False; gl.right_labels = False
        gl.left_labels = True

        gl.bottom_labels = True           # …and lon labels on the bottom of every map
        gl.xlocator = mticker.MultipleLocator(15)
        gl.xlabel_styles = gl.ylabel_styles = {"fontsize": FS_GRID}
        label_panel(ax, "def"[k])         # letters unified with a/b/c: bold, outside the
                                          # axes top-left (was halo text inside the map)
        halo_text(ax, 0.5, 0.95, f"{R} km", fs=FS_TITLE, va="top", ha="center")  # inside, upper-center
        _scs_inset(ax, proj)                            # South China Sea mini-map, bottom-right
        map_axes.append(ax)


    # bar on f's right edge (e has none of its own). All bars share the same
    # vmax (inter-radius comparability); both land in the right margin.
    caxes = []
    for ax in (map_axes[0], map_axes[2]):            # d and f
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.06,
                                                  axes_class=plt.Axes)
        cb = fig.colorbar(im, cax=cax)   # im = last map's mappable; vmin/vmax shared
        cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=2)
        cb.set_label("PEP rain contrib. (%)", fontsize=FS_CBAR_LAB)
        caxes.append(cax)
    fig.canvas.draw()
    for ax, cax in zip((map_axes[0], map_axes[2]), caxes):
        fd = ax.get_position()           # divider slot (map+pad+cbar block)
        cx = cax.get_position()          # divider-derived cax box: right of the map, 4% wide
        pad = cx.x0 - fd.x1              # divider gap between map and colourbar
        ax.set_axes_locator(None)        # size the map by gs+anchor again (same as pre-divider);
        cax.set_axes_locator(None)       # divider locators would re-derive both from the CELL on redraw
        fig.canvas.draw()
        mf = ax.get_position()           # applied (aspect-settled) map box
        cax.set_position([mf.x1 + pad, mf.y0, cx.width, mf.height])

    for t in b.get_yticklabels():        # bold b's y ticks (draw above has materialised them)
        t.set_fontweight("bold")

    return natural_save(fig, "FigS3_radius_sensitivity")


if __name__ == "__main__":
    main()
