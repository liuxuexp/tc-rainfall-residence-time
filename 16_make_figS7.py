#!/usr/bin/env python3
"""Fig. S7 - composite comparison, PEP-minus-non-PEP vs Extreme-minus-Short-RT.

Reads data/composite_anomaly_wide.nc, composite_RTquartile_wide.nc and
event_moisture_budget.csv/.npz. Output: figures/FigS7_composite_comparison.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

from _style import (label_panel, add_china, C_PER, C_NPER, OKABE_ITO,
                    smooth_field, FIG_DIR)
from config import DATA_DIR
EXTENT = [60, 180, -10, 70]
                              # tick range for all composite-map panels across

                              # frame when Fig4 widened to the synoptic context.
GEO_ASPECT = 80.0 / 120.0
CBAR_W = 0.038


F_LETTER = 12.0
F_TITLE = 11.0
F_AXIS = 10.5
F_SMALL = 9.5


def load():
    ca = xr.open_dataset(DATA_DIR / "composite_anomaly_wide.nc")     # PEP - non-PEP
    cr = xr.open_dataset(DATA_DIR / "composite_RTquartile_wide.nc")  # Extreme - Short
    lat = ca["lat"].values; lon = ca["lon"].values
    out = {}
    for k in ["IVT", "IVT_u", "IVT_v", "steer_u", "steer_v", "MFC"]:
        out[f"per_{k}"] = ca[f"diff_{k}"].values
        out[f"rt_{k}"] = cr[f"diff_{k}"].values
    out["lat"] = lat; out["lon"] = lon
    out["budget"] = pd.read_csv(DATA_DIR / "event_moisture_budget.csv")
    npz = np.load(DATA_DIR / "event_moisture_budget.npz")
    out["P_onset"] = npz["P_onset"]; out["t"] = npz["t"]; out["per_flag"] = npz["per_flag"]
    return out


def _map(ax, proj, title, letter, bottom, left):
    add_china(ax)
    ax.set_extent(EXTENT, crs=proj)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False; gl.right_labels = False
    gl.bottom_labels = bottom; gl.left_labels = left   # lon labels bottom row only,
    gl.xlocator = mticker.MultipleLocator(20)          # lat left column only (Fig.4
    gl.ylocator = mticker.MultipleLocator(20)          # tick spec: 20° both axes)
    gl.xlabel_styles = gl.ylabel_styles = {"fontsize": F_SMALL}
    ax.set_title(title, fontsize=F_TITLE, pad=2, y=1.0)   # y=1.0: cartopy 0.25 + mpl 3.11 autotitlepos workaround
    label_panel(ax, letter, fs=F_LETTER)
    return gl


def _m(z):
    return np.ma.masked_invalid(z)


def _cbar(im, label, cax):
    cb = cax.figure.colorbar(im, cax=cax)

    # over the range, so the top tick can land a full spacing ABOVE the cax top and
    # collide with the panel above once the a–f grid is tight. Keep only ticks
    # inside the norm range; endpoint labels at the limits survive.
    cb.locator = mticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    lo, hi = im.norm.vmin, im.norm.vmax
    cb.set_ticks([t for t in cb.get_ticks() if lo - 1e-9 <= t <= hi + 1e-9])
    cb.set_label(label, fontsize=F_SMALL, labelpad=1.5)
    cb.ax.tick_params(labelsize=F_SMALL, length=2)
    return cb


def _skip(s):
    return (slice(None, None, s), slice(None, None, s))


def ivt_panel(ax, d, key, proj, cax, letter, title, bottom, left):
    ivt = d[key]; u = d[key.replace("IVT", "IVT_u")]; v = d[key.replace("IVT", "IVT_v")]
    vmax = float(np.nanpercentile(np.abs(ivt), 98))
    im = ax.pcolormesh(d["LON"], d["LAT"], _m(ivt),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto", transform=proj, zorder=2)
    sk = 2
                # point = 5° spacing on the 2.5° grid
    ax.quiver(d["LON"][_skip(sk)], d["LAT"][_skip(sk)], u[_skip(sk)], v[_skip(sk)],
              transform=proj, color="0.25", scale=1200, width=0.0036,
              headwidth=5.5, headlength=6.0, headaxislength=4.6, zorder=5)
    gl = _map(ax, proj, title, letter, bottom, left)
    _cbar(im, "IVT anomaly (kg m$^{-1}$ s$^{-1}$)", cax)
    return gl


def steer_panel(ax, d, key, proj, cax, letter, title, bottom, left):
    # key like 'per_steer_u' -> base 'per_steer' -> both components
    base = key[:-2] if key.endswith("_u") else key
    uu = d[base + "_u"]; vv = d[base + "_v"]
    spd = np.hypot(uu, vv)
    vmax = float(np.nanpercentile(spd, 98))
    im = ax.pcolormesh(d["LON"], d["LAT"], _m(spd),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto", transform=proj, zorder=2)
    ax.quiver(d["LON"][_skip(2)], d["LAT"][_skip(2)], uu[_skip(2)], vv[_skip(2)],
              transform=proj, color="0.25", scale=16, width=0.0036,
              headwidth=5.5, headlength=6.0, headaxislength=4.6, zorder=5)
    gl = _map(ax, proj, title, letter, bottom, left)
    _cbar(im, "|ΔV$_{steer}$| (m s$^{-1}$)", cax)
    return gl


def mfc_panel(ax, d, key, proj, cax, letter, title, bottom, left):
    z = smooth_field(d[key])   # display-only smoothing (#52); norm taken from the display field
    vm = float(np.nanpercentile(np.abs(z), 97))
    norm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
    im = ax.pcolormesh(d["LON"], d["LAT"], _m(z), cmap="BrBG", norm=norm,
                       shading="auto", transform=proj, zorder=2)
    gl = _map(ax, proj, title, letter, bottom, left)
    _cbar(im, "MFC anomaly (mm day$^{-1}$)", cax)
    return gl


def main():
    d = load()
    LON, LAT = np.meshgrid(d["lon"], d["lat"])
    d["LON"], d["LAT"] = LON, LAT
    proj = ccrs.PlateCarree()

    # 2 composite-splits × 3 variable-panels re-flowed into a 3×2 map grid + g/h
    # closing row (= 2 cols × 4 rows overall, #52), mirroring Fig. 4: the flat
    # split-major order a..f is re-flowed row-major, so every letter keeps its

    # the MFC panels). Rows = composite split (left row label), panels titled by


    fig = plt.figure(figsize=(7.2, 8.3))
    gs = fig.add_gridspec(2, height_ratios=[2.55, 1.0], hspace=0.12,
                          left=0.09, right=0.93, top=0.965, bottom=0.05)
    gs_maps = gs[0].subgridspec(3, 2, wspace=0.03, hspace=0.185)   # 3×2 maps, shared axes
    gs_gh = gs[1].subgridspec(1, 2, wspace=0.20)                   # bottom: g | h
    map_axes, map_caxes, map_gls = [], [], []

    rows = [("per", "PEP − non-PEP"), ("rt", "Extreme − Short RT")]
    cols = [(ivt_panel, "IVT", "IVT + transport vectors"),
            (steer_panel, "steer_u", "Steering-flow anomaly"),
            (mfc_panel, "MFC", "Moisture-flux convergence")]
    letters = ["a", "b", "c", "d", "e", "f"]
    for r, (prefix, _rlab) in enumerate(rows):
        for c, (fn, var, vtitle) in enumerate(cols):
            k = r * 3 + c                    # flat panel order a..f (split-major)
            jr, jc = k // 2, k % 2
            key = f"{prefix}_{var}"
            ax = fig.add_subplot(gs_maps[jr, jc], projection=proj)
            # Fig.4 _main_and_cbar pattern: west-pinned height-limited map + a PLAIN
            # independent colourbar axes pinned beside it post-draw. (A
            # make_axes_locatable divider child flips the GeoAxes to
            # adjustable="datalim", centring the box mid-cell and re-deriving its

            ax.set_anchor("W")
            ax.set_box_aspect(GEO_ASPECT)
            cax = fig.add_axes([0.0, 0.0, 0.01, 0.1])
            gl = fn(ax, d, key, proj, cax, letters[k], vtitle,
                    bottom=(jr == 2), left=(jc == 0))
            map_axes.append(ax); map_caxes.append(cax); map_gls.append(gl)

    # (g) lifecycle
    g = fig.add_subplot(gs_gh[0, 0])
    P = d["P_onset"]; flag = d["per_flag"]; rtq = d["budget"]["RT_quartile"].values
    rs = np.random.RandomState(0)
    # plan Fig.S7g: the two blues now differ in depth AND dash length
    # (Extreme = deep blue long dash, Short = light blue short dash)
    BLUE_SHORT = "#B3DDF2"
    curves = [
        (flag == 1, C_PER, "PEP", "-", True),
        (flag == 0, C_NPER, "non-PEP", "-", False),
        (rtq == "Extreme", OKABE_ITO[2], "Extreme RT", (0, (6, 2)), True),
        (rtq == "Short", BLUE_SHORT, "Short RT", (0, (2, 1.5)), False),
    ]
    for m, col, lab, ls, ci in curves:
        sub = P[m]
        if len(sub) < 2:
            continue
        mean = np.nanmean(sub, axis=0)
        if ci:
            draws = sub[rs.randint(0, len(sub), size=(500, len(sub)))]
            lo, hi = np.nanpercentile(np.nanmean(draws, axis=1), [2.5, 97.5], axis=0)
            g.fill_between(d["t"], lo, hi, color=col, alpha=0.15, lw=0)
        g.plot(d["t"], mean, color=col, lw=1.4, ls=ls, label=f"{lab} (n={len(sub)})")
    g.axvline(0, color="0.5", lw=0.5, ls="--")
    g.set_xlabel("days since onset", fontsize=F_AXIS)
    g.set_ylabel("Area-mean precip (mm day$^{-1}$)", fontsize=F_AXIS)
    g.tick_params(labelsize=F_SMALL)
    # plan Fig.S7g: legend rows tightened but not shrunk below the ladder
    g.legend(fontsize=F_SMALL, frameon=False, loc="upper left", ncol=1,
             labelspacing=0.3, handlelength=1.8)
    g.set_title("Rainfall lifecycle", fontsize=F_TITLE, pad=2)
    g.grid(axis="y", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        g.spines[sp].set_visible(False)
    label_panel(g, "g", x=-0.14, fs=F_LETTER)

    # (h) moisture budget
    h = fig.add_subplot(gs_gh[0, 1])
    dfb = d["budget"]; terms = ["P", "MFC", "storage", "E"]
    groups = [(dfb.PER_flag == 1, C_PER, "PEP"),
              (dfb.PER_flag == 0, C_NPER, "non-PEP"),
              (dfb.RT_quartile == "Extreme", OKABE_ITO[2], "Extreme"),
              (dfb.RT_quartile == "Short", "#B3DDF2", "Short")]
    x = np.arange(len(terms)); w = 0.20
    for k, (mask, col, lab) in enumerate(groups):
        vals = [dfb.loc[mask, t].mean() for t in terms]
        # plan Fig.S7h: thin outlines so near-zero terms read; order untouched
        h.bar(x + (k - 1.5) * w, vals, w, color=col, label=lab,
              edgecolor="0.25", linewidth=0.6)
    h.axhline(0, color="0.3", lw=0.8)
    h.set_xticks(x); h.set_xticklabels(terms, fontsize=F_SMALL)
    h.set_ylabel("mm day$^{-1}$", fontsize=F_AXIS); h.tick_params(labelsize=F_SMALL)
    h.legend(fontsize=F_SMALL, frameon=False, loc="upper right", ncol=1,
             labelspacing=0.3, handlelength=1.4)
    h.set_title("Moisture budget", fontsize=F_TITLE, pad=2)
    h.grid(axis="y", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        h.spines[sp].set_visible(False)
    label_panel(h, "h", x=-0.14, fs=F_LETTER)


    # Colourbar pinning + stack clearance: pin each plain cax beside its map once
    # the GeoAxes have settled (west-pinned, height-limited: the box fills the
    # cell height, the cell's extra width is east slack). If a stack still crosses
    # its column boundary (col 1: right map's drawn left edge + gutter; col 2:
    # gs.right) shrink every map by max(overshoot)/(1+CBAR_W) and re-pin.
    fig.canvas.draw()
    r_ = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gut = 0.04 / fig.get_size_inches()[0]
    pad = 0.05 / fig.get_size_inches()[0]

    def pin():
        for ax, cax in zip(map_axes, map_caxes):
            b = ax.get_window_extent(r_).transformed(inv)
            cax.set_position([b.x1 + pad, b.y0, CBAR_W * b.width, b.height])
    pin()
    fig.canvas.draw()
    boxes = [ax.get_window_extent(r_).transformed(inv) for ax in map_axes]
    overs = []
    for k, cax in enumerate(map_caxes):
        ends = [cax.yaxis.label.get_window_extent(r_).x1]
        ends += [t.get_window_extent(r_).x1
                 for t in cax.yaxis.get_ticklabels() if t.get_text()]
        end = inv.transform((max(ends), 0.0))[0]           # stack end, fraction
        tgt = (boxes[k + 1].x0 - gut if k % 2 == 0 else gs.right)
        overs.append(end - tgt)
    over = max(max(overs), 0.0) / (1.0 + CBAR_W)
    if over > 0:
        for ax, b in zip(map_axes, boxes):
            ax.set_position([b.x0, b.y0, b.width - over, b.height])
        fig.canvas.draw()  # GeoAxes re-settle height = new width × GEO_ASPECT
        pin()              # re-pin the caxes to the shrunk maps
        fig.canvas.draw()

    # Tighten the column gutter (Fig.4 pattern): the right-column maps sit at
    # their cells' west edge, leaving a wide empty gutter past the left column's


    # gs.right, so the clearance above cannot fire.
    fig.canvas.draw()
    left_end = []
    for k in (0, 2, 4):
        cax = map_caxes[k]
        ends = [cax.yaxis.label.get_window_extent(r_).x1]
        ends += [t.get_window_extent(r_).x1
                 for t in cax.yaxis.get_ticklabels() if t.get_text()]
        left_end.append(inv.transform((max(ends), 0.0))[0])
    boxes2 = [ax.get_window_extent(r_).transformed(inv) for ax in map_axes]
    shift = min(boxes2[1].x0 - e for e in left_end) - 0.08 / fig.get_size_inches()[0]
    if shift > 0:
        for k in (1, 3, 5):
            b = boxes2[k]
            map_axes[k].set_position([b.x0 - shift, b.y0, b.width, b.height])
        fig.canvas.draw()
        pin()
        fig.canvas.draw()


    # 3×2 re-flow each split's three panels span two grid rows (a,b,c / d,e,f), so
    # centre each label on its split's band: midway between the first and last map
    # of the split (drawn boxes). x_lab clears the leftmost maps' latitude tick
    # labels, measured from the Gridliners' left label artists (#52 successor: the

    # font enlargement would have broken it).
    fig.canvas.draw()
    left_edge = min(t.get_window_extent(r_).x0
                    for gl in (map_gls[0], map_gls[2], map_gls[4])
                    for t in gl.ylabel_artists
                    if t.get_text() and t.get_visible()
                    and t.get_window_extent(r_).width > 0)
    x_lab = inv.transform((left_edge, 0.0))[0] - 0.08 / fig.get_size_inches()[0]
    for r, (_prefix, rlab) in enumerate(rows):
        b0 = map_axes[r * 3].get_window_extent(r_).transformed(inv)
        b2 = map_axes[r * 3 + 2].get_window_extent(r_).transformed(inv)
        y_lab = (b0.y0 + b0.height / 2 + b2.y0 + b2.height / 2) / 2
        fig.text(x_lab, y_lab, rlab, rotation=90,
                 va="center", ha="center", fontsize=F_SMALL)

    # Left-align the bottom row with the map block above (Fig.4 pattern): (g)
    # starts exactly under the top-left map, (h)'s right edge aligns with the
    # widest right-column colourbar STACK end (one clean right edge).
    fig.canvas.draw()
    r2 = fig.canvas.get_renderer()
    inv2 = fig.transFigure.inverted()
    mb = [ax.get_window_extent(r2).transformed(inv2) for ax in map_axes]
    x0 = mb[0].x0                                # left edge of top-left map (panel a)
    ends = []
    for k in (1, 3, 5):
        ends += [map_caxes[k].yaxis.label.get_window_extent(r2).x1]
        ends += [t.get_window_extent(r2).x1
                 for t in map_caxes[k].yaxis.get_ticklabels() if t.get_text()]
    x_right = inv2.transform((max(ends), 0.0))[0]
    gap = 0.05
    each = ((x_right - x0) - gap) / 2.0
    g_pos = g.get_position()
    h_pos = h.get_position()
    g.set_position([x0, g_pos.y0, each, g_pos.height])
    h.set_position([x0 + each + gap, h_pos.y0, each, h_pos.height])
    # g/h titles sit over the map columns above: centre each on the corresponding
    # map's drawn centre (e for g, f for h) instead of default axes-centre.
    for gx, mbb in zip((g, h), (mb[4], mb[5])):
        gp = gx.get_position()
        gx.title.set_x((mbb.x0 + mbb.width / 2 - gp.x0) / gp.width)
    # (b)/(d)/(f) letters share one left edge with the (h) letter (Fig.4 pattern):
    # measure (h)'s settled letter and re-fraction the right-column letters to the

    hx = h.texts[0].get_window_extent(r2).transformed(inv2).x0
    for k in (1, 3, 5):
        map_axes[k].texts[0].set_x((hx - mb[k].x0) / mb[k].width)


    # rescaling to a fixed mm column width distorts the panel spacing). PNG

    base = str(Path(__file__).resolve().parents[1] / "figures" / "FigS7_composite_comparison")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(FIG_DIR / "pdf" / "FigS7_composite_comparison.pdf", dpi=600)
    w_in, h_in = fig.get_size_inches()
    print(f"[save] FigS7_composite_comparison (natural size, no width lock) -> "
          f"{w_in:.2f}x{h_in:.2f} in = {w_in*25.4:.1f}x{h_in*25.4:.1f} mm")
    plt.close(fig)


if __name__ == "__main__":
    main()
