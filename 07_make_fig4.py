#!/usr/bin/env python3
"""Fig. 4 - circulation and moisture mechanism composites (a-h).

Wide-domain (60-180E / 10S-70N) maps from data/composite_RTquartile_wide.nc
and data/era5_event_composite_wide.nc, plus the event moisture budget from
data/event_moisture_budget.csv/.npz. Output: figures/Fig4_mechanism.png.
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

from _style import (save, label_panel, add_china, C_PER, C_NPER, smooth_field,
                    FIG_DIR)
from config import DATA_DIR

EXTENT = [60, 180, -10, 70]

                                 # anomaly pattern. The fields were recomputed on
                                 # exactly this frame (data/*_wide.nc), so the
                                 # colour fills and vectors tile the frame with no
                                 # blank margin. S7 shares this frame since the

GEO_ASPECT = 80.0 / 120.0
CBAR_W = 0.038
JULIAN_JJAS = slice(152, 274)


# letter 12, title 11, axis 10.5, ticks/legend/cbar/grid 9.5.
F_LETTER = 12.0
F_TITLE = 11.0
F_AXIS = 10.5
F_SMALL = 9.5


def load():
    comp = xr.open_dataset(DATA_DIR / "composite_RTquartile_wide.nc")
    era = xr.open_dataset(DATA_DIR / "era5_event_composite_wide.nc")
    lat = comp["lat"].values; lon = comp["lon"].values
    d = {f"diff_{k}": comp[f"diff_{k}"].values
         for k in ["IVT", "IVT_u", "IVT_v", "TCWV", "MFC", "omega500", "steer_u", "steer_v", "hgt500"]}
    # 500-hPa ridge lines: 5880 dagpm is absent inside this core domain (max ~5874),
    # so the 5860-dagpm ridge is used to mark the high-pressure position.
    d["clim_jjas_hgt500"] = era["clim_hgt500"].sel(doy=JULIAN_JJAS).mean("doy").values
    d["per_hgt500"] = era["per_hgt500"].values
    comp.close(); era.close()
    npz = np.load(DATA_DIR / "event_moisture_budget.npz")
    d["life_t"] = npz["t"]; d["life_P"] = npz["P_onset"]; d["life_flag"] = npz["per_flag"]
    d["budget"] = pd.read_csv(DATA_DIR / "event_moisture_budget.csv")
    return d, lat, lon


def derive(lat, lon):
    LON, LAT = np.meshgrid(lon, lat)
    return dict(LON=LON, LAT=LAT)


# ---------- shared map scaffolding ----------
def _map(ax, proj, title, letter, bottom=True, left=True):
    add_china(ax)
    ax.set_extent(EXTENT, crs=proj)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False; gl.right_labels = False
    gl.bottom_labels = bottom; gl.left_labels = left   # shared-look labels: all six maps draw
                                                       # the identical EXTENT, so lon labels sit
                                                       # on the bottom row only, lat on the left

    gl.xlocator = mticker.MultipleLocator(20)

    gl.ylocator = mticker.MultipleLocator(20)
                                                # labelled — matches the lon density
    gl.xlabel_styles = gl.ylabel_styles = {"fontsize": F_SMALL}
    ax.set_title(title, fontsize=F_TITLE, pad=2, y=1.0)   # y=1.0: cartopy 0.25 + mpl 3.11 autotitlepos workaround
    label_panel(ax, letter, fs=F_LETTER)
    return gl


def _cbar(im, label, cax):
    cb = cax.figure.colorbar(im, cax=cax)

    # range, so a 0–2.08 bar gets a 2.5 tick that renders a full spacing ABOVE

    # panel above once the a–f grid tightened to ~0.28 in (measured). Keep only
    # ticks inside the norm range; endpoint labels at the limits survive.
    cb.locator = mticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    lo, hi = im.norm.vmin, im.norm.vmax
    cb.set_ticks([t for t in cb.get_ticks() if lo - 1e-9 <= t <= hi + 1e-9])
    cb.set_label(label, fontsize=F_SMALL, labelpad=1.5)
    cb.ax.tick_params(labelsize=F_SMALL, length=2)
    return cb


def _m(z):
    return np.ma.masked_invalid(z)


def _skip(s):
    return (slice(None, None, s), slice(None, None, s))


# ---------- map panels (top group: shared x/y, individual colourbars) ----------
def panel_a(ax, d, s, proj, cax, bottom=True, left=True):
    ivt = d["diff_IVT"]
    vmax = float(np.nanpercentile(np.abs(ivt), 98))
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(ivt), cmap="YlOrRd", vmin=0, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    sk = 2
                # point = 5° spacing on the 2.5° grid
    # plan Fig.4(a)(c): deep grey (not near-black) vectors, shared shaft/head spec
    ax.quiver(s["LON"][_skip(sk)], s["LAT"][_skip(sk)],
              d["diff_IVT_u"][_skip(sk)], d["diff_IVT_v"][_skip(sk)],
              transform=proj, color="0.25", scale=1200, width=0.0036,
              headwidth=5.5, headlength=6.0, headaxislength=4.6, zorder=5)
    _map(ax, proj, "IVT + transport vectors", "a", bottom, left)
    _cbar(im, "IVT anomaly (kg m$^{-1}$ s$^{-1}$)", cax)


def panel_b(ax, d, s, proj, cax, bottom=True, left=True):
    z = smooth_field(d["diff_hgt500"])   # display-only smoothing; ridge contours below stay raw
    vm = float(np.nanpercentile(np.abs(z), 97))
    norm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(z), cmap="PuOr", norm=norm,
                       shading="auto", transform=proj, zorder=2)
    # 5860-dagpm ridge: climatology (dashed grey) vs PEP composite (solid black)
    # plan Fig.4(b): meanings kept, ONE unified contour width
    ax.contour(s["LON"], s["LAT"], d["clim_jjas_hgt500"], levels=[5860],
               colors="0.45", linewidths=0.9, linestyles="--", transform=proj, zorder=4)
    ax.contour(s["LON"], s["LAT"], d["per_hgt500"], levels=[5860],
               colors="k", linewidths=0.9, transform=proj, zorder=5)
    _map(ax, proj, "500-hPa height anomaly", "b", bottom, left)
    _cbar(im, "Z500 anomaly (m)", cax)


def panel_c(ax, d, s, proj, cax, bottom=True, left=True):
    spd = np.hypot(d["diff_steer_u"], d["diff_steer_v"])
    vmax = float(np.nanpercentile(spd, 98))
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(spd), cmap="YlOrRd", vmin=0, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    sk = 2
    ax.quiver(s["LON"][_skip(sk)], s["LAT"][_skip(sk)],
              d["diff_steer_u"][_skip(sk)], d["diff_steer_v"][_skip(sk)],
              transform=proj, color="0.25", scale=16, width=0.0036,
              headwidth=5.5, headlength=6.0, headaxislength=4.6, zorder=5)
    _map(ax, proj, "Steering-flow anomaly", "c", bottom, left)
    _cbar(im, "|ΔV$_{steer}$| (m s$^{-1}$)", cax)


def panel_d(ax, d, s, proj, cax, bottom=True, left=True):
    tcwv = smooth_field(d["diff_TCWV"])
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(tcwv), cmap="YlGnBu", vmin=0,
                       vmax=float(np.nanpercentile(tcwv, 98)),
                       shading="auto", transform=proj, zorder=2)
    _map(ax, proj, "Column water vapor", "d", bottom, left)
    _cbar(im, "TCWV anomaly (mm)", cax)


def panel_e(ax, d, s, proj, cax, bottom=True, left=True):
    mfc = smooth_field(d["diff_MFC"])
    vm = float(np.nanpercentile(np.abs(mfc), 97))
    norm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(mfc), cmap="BrBG", norm=norm,
                       shading="auto", transform=proj, zorder=2)
    _map(ax, proj, "Moisture-flux convergence", "e", bottom, left)
    _cbar(im, "MFC anomaly (mm day$^{-1}$)", cax)


def panel_f(ax, d, s, proj, cax, bottom=True, left=True):
    om0 = smooth_field(d["diff_omega500"]); om = om0 * 100.0   # smooth, then Pa/s -> 10⁻² Pa/s
    vm = float(np.nanpercentile(np.abs(om), 97))
    norm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(om), cmap="RdBu", norm=norm,
                       shading="auto", transform=proj, zorder=2)
    _map(ax, proj, "500-hPa vertical motion", "f", bottom, left)
    _cbar(im, r"$\omega_{500}$ anomaly (10$^{-2}$ Pa s$^{-1}$)", cax)


# ---------- non-map panels (bottom group) ----------
def panel_g(ax, d, s, proj, cax):
    if cax is not None:
        cax.set_visible(False)
    P, flag, t = d["life_P"], d["life_flag"], d["life_t"]
    rs = np.random.RandomState(0)
    for f, color, lab in [(1, C_PER, "PEP"), (0, C_NPER, "non-PEP")]:
        sub = P[flag == f]
        mean = np.nanmean(sub, axis=0)
        n = len(sub)
        draws = sub[rs.randint(0, n, size=(1000, n))]
        boot = np.nanmean(draws, axis=1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        ax.fill_between(t, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(t, mean, color=color, lw=1.4, label=f"{lab} (n={n})")
    ax.axvline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("days since onset", fontsize=F_AXIS)
    ax.set_ylabel("Area-mean precip (mm day$^{-1}$)", fontsize=F_AXIS)
    ax.tick_params(labelsize=F_SMALL)
    ax.set_title("Rainfall lifecycle", fontsize=F_TITLE, pad=2)
    ax.legend(fontsize=F_SMALL, frameon=False, loc="upper left",
              handlelength=1.6, labelspacing=0.3)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    label_panel(ax, "g", x=-0.14, fs=F_LETTER)


def panel_h(ax, d, s, proj, cax):
    if cax is not None:
        cax.set_visible(False)
    df = d["budget"]
    terms = ["P", "MFC", "storage", "E"]
    per = np.array([df.loc[df.PER_flag == 1, t].mean() for t in terms])
    nper = np.array([df.loc[df.PER_flag == 0, t].mean() for t in terms])
    x = np.arange(len(terms)); w = 0.38
    # plan Fig.4(h): thin bar outlines (0.6 pt) make near-zero terms legible;
    # zero line emphasized; no axis break, no negative cropping
    ax.bar(x - w / 2, per, w, color=C_PER, label="PEP",
           edgecolor="0.25", linewidth=0.6)
    ax.bar(x + w / 2, nper, w, color=C_NPER, alpha=0.75, label="non-PEP",
           edgecolor="0.25", linewidth=0.6)
    ax.tick_params(labelsize=F_SMALL)   # before set_xticklabels so the term labels win
    ax.set_xticks(x); ax.set_xticklabels(terms, fontsize=F_SMALL)
    ax.set_ylabel("mm day$^{-1}$", fontsize=F_AXIS)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_title("Moisture budget", fontsize=F_TITLE, pad=2)
    ax.legend(fontsize=F_SMALL, frameon=False, loc="upper right",
              handlelength=1.6, labelspacing=0.3)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    label_panel(ax, "h", x=-0.14, fs=F_LETTER)


MAP_PANELS = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]


def _main_and_cbar(fig, subspec, proj, geo):
    # NOTE: cartopy GeoAxes forbid matplotlib sharex/sharey (adjustable='datalim' clash),
    # so axes are "shared" visually: identical EXTENT + edge-only gridline labels.
    ax = fig.add_subplot(subspec, projection=proj) if geo else fig.add_subplot(subspec)
    if geo:
        # height-limited maps: pin to the cell's WEST edge so the east slack
        # hosts the colourbar stack (anchor "C" would split the slack and push

        ax.set_anchor("W")
        ax.set_box_aspect(GEO_ASPECT)
        # cax = PLAIN independent axes (make_figS8 pattern), pinned beside the
        # map once the layout settles. A make_axes_locatable divider child (a)
        # flips the GeoAxes to adjustable="datalim", which ignores the anchor


        # position at every draw, unstable across the measure+render passes.
        cax = fig.add_axes([0.0, 0.0, 0.01, 0.1])
    else:
        cax = None
    return ax, cax


def compose(d, lat, lon):
    s = derive(lat, lon)
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.2, 8.3))

    # are HEIGHT-limited (box fills the cell height; the cell's extra width is
    # slack), so the row gap comes from hspace alone: inner hspace 0.185 ≈ 0.28 in

    # 0.12 houses the bottom-row lon labels + (g)/(h) titles. ratios 2.55/1
    # (was 3.1/1) absorb the freed block height into (g)/(h) ≈ 2.0 in tall so


    # the ~91-px colourbar stack (strip + 9.5-pt numbers + rotated label) fits

    # full height-limited 340×227 px (2.27×1.51 in); outer whitespace is
    # tight-cropped away.
    gs = fig.add_gridspec(2, height_ratios=[2.55, 1.0], hspace=0.12,
                          left=0.09, right=0.93, top=0.965, bottom=0.05)
    gs_maps = gs[0].subgridspec(3, 2, wspace=0.03, hspace=0.185)  # 3×2 maps, shared axes
    gs_gh = gs[1].subgridspec(1, 2, wspace=0.20)                  # bottom: g | h
    map_axes = []
    map_caxes = []
    for k, fn in enumerate(MAP_PANELS):
        r, c = k // 2, k % 2
        ax, cax = _main_and_cbar(fig, gs_maps[r, c], proj, True)
        fn(ax, d, s, proj, cax, bottom=(r == 2), left=(c == 0))
        map_axes.append(ax)
        map_caxes.append(cax)
    gh_axes = []
    for c, fn in [(0, panel_g), (1, panel_h)]:
        ax, cax = _main_and_cbar(fig, gs_gh[0, c], proj, False)
        fn(ax, d, s, proj, cax)
        gh_axes.append(ax)

    # independent axes (see _main_and_cbar), so pin each beside its map once
    # the GeoAxes have settled (west-pinned, height-limited: the box fills the
    # cell height, the cell's extra width is east slack). With the 10°S–70°N

    # drawn boxes as window extents (get_position returns the CELL rect) and
    # only if a stack still crosses its column boundary (col 1: right map's
    # drawn left edge + 0.04 in gutter; col 2: gs.right) shrink every map by

    # re-pin. Clamped ≥0: a negative overshoot must not widen the maps.
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


    # maps sit at their cells' west edge, leaving a wide empty gutter past the


    # move further inboard of gs.right, so the clearance above cannot fire.
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

    # Left-align the bottom row with the map block above. The maps are GeoAxes
    # pinned west + box_aspect + a right colourbar, so measure the DRAWN boxes

    # panel (g) starts exactly under the top-left map, while (h)'s right edge
    # aligns with the widest right-column colourbar STACK end (the stacks end

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
    g_pos = gh_axes[0].get_position()
    h_pos = gh_axes[1].get_position()
    gh_axes[0].set_position([x0, g_pos.y0, each, g_pos.height])
    gh_axes[1].set_position([x0 + each + gap, h_pos.y0, each, h_pos.height])
    # g/h titles sit over the map columns above: centre each on the corresponding
    # map's drawn centre (e for g, f for h) instead of default axes-centre.
    for gx, mbb in zip(gh_axes, (mb[4], mb[5])):
        gp = gx.get_position()
        gx.title.set_x((mbb.x0 + mbb.width / 2 - gp.x0) / gp.width)


    # clear of the stacks (numbers/rotated labels stay inside the cax height).
    hx = gh_axes[1].texts[0].get_window_extent(r2).transformed(inv2).x0
    for k in (1, 3, 5):
        map_axes[k].texts[0].set_x((hx - mb[k].x0) / mb[k].width)

    # rescales to a fixed mm column width, which distorts the panel spacing).

    base = str(Path(__file__).resolve().parents[1] / "figures" / "Fig4_mechanism")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(FIG_DIR / "pdf" / "Fig4_mechanism.pdf", dpi=600)
    w_in, h_in = fig.get_size_inches()
    print(f"[save] Fig4_mechanism (natural size, no width lock) -> "
          f"{w_in:.2f}x{h_in:.2f} in = {w_in*25.4:.1f}x{h_in*25.4:.1f} mm")
    plt.close(fig)


def main():
    d, lat, lon = load()
    compose(d, lat, lon)


if __name__ == "__main__":
    main()
