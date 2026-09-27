#!/usr/bin/env python3
"""Fig. 3 - spatial patterns of PEP rainfall, counts, RT and trends (a-f).

Reads the cached grids figure_data/fig3_grids.npz (built on first run by
03_build_fig3_grids.py). Run with --panels for single-panel debug renders
under figures/_panels/. Output: figures/Fig3_PEP_spatial.png.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _style import (save, label_panel, add_china, FIG_DIR, C_PER,
                    FS_TITLE, FS_AXIS, FS_TICK, FS_CBAR_LAB, FS_CBAR_TICK,
                    FS_GRID, FS_ANNOT)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "fig3_grids", Path(__file__).resolve().parent / "03_build_fig3_grids.py")
fig3_grids = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(fig3_grids)
build_grids = fig3_grids.build

EXTENT = [95, 134, 16, 50]             # China focus (W edge at 95°E)
GEO_ASPECT = 1.04
RAIN_THR = 100.0                       # mm/64 yr; masks noisy low-TC-rain cells
CBAR_W = 0.0425                        # cbar sub-axis width ratio (vs main = 1)
REGIONS = [
    ("South\nChina",  (18, 27), (104, 122)),
    ("East\nChina",   (27, 34), (116, 122)),
    ("Yangtze",       (27, 34), (105, 116)),
    ("North\nChina",  (34, 43), (110, 122)),
    ("NE\nChina",     (43, 53), (115, 134)),
]
# panel-f x tick codes; the caption names the five regions in this order
REGION_ABBR = {
    "South\nChina": "SC", "East\nChina": "EC", "Yangtze": "YZ",
    "North\nChina": "NC", "NE\nChina": "NEC",
}


def load():
    build_grids()
    z = np.load(FIG_DIR.parent / "figure_data" / "fig3_grids.npz")
    return {k: z[k] for k in z.files}


def derive(d):
    lat, lon = d["lat"], d["lon"]
    LON, LAT = np.meshgrid(lon, lat)
    with np.errstate(invalid="ignore", divide="ignore"):
        contrib = np.where(d["all_rain"] > RAIN_THR,
                           100.0 * d["per_rain"] / d["all_rain"], np.nan)
    return dict(lat=lat, lon=lon, LON=LON, LAT=LAT, contrib=contrib,
                rt_count=d["rt_count"])


# ---------- shared helpers ----------
def _scs_inset(ax, proj):
    """Bottom-right South China Sea inset, shifted to hug the map's bottom-right corner."""
    w, h, y0 = 0.245, 0.345, -0.012
    axins = ax.inset_axes([1 - w, y0, w, h], projection=proj)
    add_china(axins)
    axins.set_extent([105, 122, 2, 25], crs=proj)
    axins.tick_params(length=0, labelsize=0)
    for side in ("top", "left"):                       # inner border only …
        axins.spines[side].set_edgecolor("0.35"); axins.spines[side].set_linewidth(0.5)
    for side in ("right", "bottom"):                   # … outer edges merge with the map
        axins.spines[side].set_visible(False)
    return axins


def _base_map(ax, proj, title, letter):
    add_china(ax)
    ax.set_extent(EXTENT, crs=proj)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(10)            # lon ticks every 10°
    gl.xlabel_style = gl.ylabel_style = {"fontsize": FS_GRID}

    # pushes titles to y=inf (silently dropped) when gridliner labels are on; an
    # explicit y sets _autotitlepos=False so the title actually renders.
    ax.set_title(title, y=1.0, fontsize=FS_TITLE, pad=2)
    label_panel(ax, letter)
    _scs_inset(ax, proj)
    return gl


def _cbar(im, label, cax, log=False, labelpad=3.0, fmt=None, integer=False):
    """Vertical colorbar in the dedicated cax (outside the map).

    integer=True → clean whole-number ticks (panel a frequency)."""
    from matplotlib.ticker import MaxNLocator
    cb = cax.figure.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FS_CBAR_LAB, labelpad=labelpad)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=2)
    if log:
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    elif fmt is not None:
        cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    if integer:
        cb.locator = MaxNLocator(nbins=5, integer=True)
        cb.update_ticks()
    return cb


def _m(z):
    return np.ma.masked_invalid(z)


# ---------- per-panel drawers ----------
def panel_a(ax, d, s, proj, cax):
    pg = d["per_grid"]
    vmax = max(float(np.percentile(pg[pg > 0], 98)), 5)
    vmax = float(np.ceil(vmax / 5) * 5)          # round → even linear ticks
    # plan Fig.3a: same monotone ramp/range, darkest end lifted (start 12% in)
    # so low-frequency cell edges stay visible; high/low direction untouched
    magma_lite = mpl.colors.LinearSegmentedColormap.from_list(
        "magma_lite", plt.get_cmap("magma")(np.linspace(0.12, 1.0, 256)))
    im = ax.pcolormesh(s["LON"], s["LAT"], np.ma.masked_where(pg == 0, pg),
                       cmap=magma_lite, vmin=0, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    _base_map(ax, proj, "PEP event frequency", "a")
    _cbar(im, "PEP events (count)", cax, integer=True)


def panel_b(ax, d, s, proj, cax):
    pr = d["per_rain"]
    vmax = float(np.percentile(pr[pr > 0], 98))
    vmax = float(np.ceil(vmax / 1000) * 1000)    # round to 1000s → even ticks
    im = ax.pcolormesh(s["LON"], s["LAT"], np.ma.masked_where(pr <= 0, pr),
                       cmap="YlOrRd", vmin=0, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    _base_map(ax, proj, "PEP cumulative rainfall", "b")
    _cbar(im, "PEP rainfall (mm)", cax,
          fmt=lambda v, _: "0" if abs(v) < 1 else f"{v / 1000:g}k")


def panel_c(ax, d, s, proj, cax):
    c = s["contrib"]
    vmax = float(np.nanpercentile(c, 98))
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(c),
                       cmap="YlGnBu", vmin=0, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    _base_map(ax, proj, "PEP rainfall contribution", "c")
    _cbar(im, "PEP share of TC rain (%)", cax)


def panel_d(ax, d, s, proj, cax):
    rtc, rtm = d["rt_count"], d["rt_mean"]
    rt = np.where(rtc >= 3, rtm, np.nan)
    vmin = float(np.nanpercentile(rt, 2))
    vmax = float(np.nanpercentile(rt, 98))
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(rt),
                       cmap="Blues", vmin=vmin, vmax=vmax,
                       shading="auto", transform=proj, zorder=2)
    _base_map(ax, proj, "Mean residence time", "d")
    _cbar(im, "Mean RT$_e$ (h)", cax)


def panel_e(ax, d, s, proj, cax):
    slope, pval, nyr = d["trend_slope"], d["trend_pval"], d["trend_nyr"]   # cached (builder)
    vmax = 10.0                                      # symmetric ±10 mm/dec (≥ P97 of |slope|)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    full = plt.get_cmap("RdBu_r")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "RdBu_lite", full(np.linspace(0.12, 0.88, 256)))   # drop dark ends → no black
    im = ax.pcolormesh(s["LON"], s["LAT"], _m(slope),
                       cmap=cmap, norm=norm,
                       shading="auto", transform=proj, zorder=2)
    _base_map(ax, proj, "PEP rainfall trend", "e")
    _cbar(im, "mm/dec (OLS)", cax, labelpad=0)
    sig = (pval < 0.05) & (nyr >= 5)
    if sig.any():
        # plan Fig.3e: stipple set unchanged; display only (tone/alpha) tuned
        ax.contourf(s["LON"], s["LAT"], sig.astype(float), levels=[0.5, 1.5],
                    colors="0.35", alpha=0.45, transform=proj, zorder=5)
    # legend: grey patch (matches the stipple) + label, top-left corner
    ax.add_patch(mpl.patches.Rectangle((0.02, 0.915), 0.040, 0.048, transform=ax.transAxes,
                                       facecolor="0.35", alpha=0.45, edgecolor="none", zorder=6))
    ax.text(0.075, 0.91, f"p<0.05, ≥5 PEP-yr\n({int(sig.sum())} cells)",
            transform=ax.transAxes, fontsize=FS_ANNOT, va="center", ha="left",
            color="0.25", zorder=6)


def panel_f(ax, d, s, proj, cax):
    if cax is not None:
        cax.set_visible(False)                   # no colorbar for the violin panel
    lat, lon, contrib = s["lat"], s["lon"], s["contrib"]
    data, pos, labs = [], [], []
    for k, (name, (la0, la1), (lo0, lo1)) in enumerate(REGIONS):
        lm = (lat >= la0) & (lat < la1)
        om = (lon >= lo0) & (lon < lo1)
        v = contrib[lm[:, None] & om[None, :] & np.isfinite(contrib)]
        if v.size:
            data.append(v); pos.append(k + 1); labs.append(name)
    vp = ax.violinplot(data, positions=pos, widths=0.72,
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        # plan Fig.3f: calmer fill, darker outline (median below stays visible)
        body.set_facecolor(C_PER); body.set_edgecolor("#A84A00")
        body.set_alpha(0.30); body.set_linewidth(0.9)
    vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.4)
    ax.set_xticks(pos); ax.set_xticklabels([REGION_ABBR[s] for s in labs], fontsize=FS_TICK)
    ax.set_ylabel("PEP share of TC rain (%)", fontsize=FS_AXIS, labelpad=1)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title("Regional PEP contribution", fontsize=FS_TITLE, pad=2)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    label_panel(ax, "f", x=-0.14)


PANELS = [("a", panel_a, True), ("b", panel_b, True), ("c", panel_c, True),
          ("d", panel_d, True), ("e", panel_e, True), ("f", panel_f, False)]


# ---------- assembly ----------
def _main_and_cbar(fig, subspec, proj, geo):
    """Map axes (undistorted box-aspect) + a colorbar hugging its right edge (0.05 in).
    The non-map (violin) panel takes no cax and no divider child — a hidden divider
    spacer re-derives the parent position at every draw, which would fight the
    explicit pinning compose() applies to (f) afterwards."""
    ax = fig.add_subplot(subspec, projection=proj) if geo else fig.add_subplot(subspec)
    ax.set_box_aspect(GEO_ASPECT)
    ax.set_anchor("C")
    if geo:
        cax = make_axes_locatable(ax).append_axes(
            "right", size=f"{CBAR_W * 100:.1f}%", pad=0.05, axes_class=mpl.axes.Axes)
    else:
        cax = None
    return ax, cax


def compose(d):
    """3×2 composed figure, equal-sized panels, outside colorbars."""
    s = derive(d)
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 2)
    gs.update(left=0.09, right=0.90, top=0.95, bottom=0.05,
              wspace=0.20, hspace=0.16)
    axes = []
    for k, (letter, fn, geo) in enumerate(PANELS):
        ax, cax = _main_and_cbar(fig, gs[k // 2, k % 2], proj, geo)
        fn(ax, d, s, proj, cax)
        axes.append(ax)

    # The violin's aspect-shrunk box sits right of the map frames, so its "(f)"
    # letter misses the (b)/(d) letters by ~6 mm. Measure the settled layout
    # (cartopy applies aspect via set_position, so get_position() IS the drawn map
    # rect) and place (f) with its left spine exactly on b's map frame, width =
    # map width ×1.10 (the delivered violin equalled the map width) growing
    # rightward into the space where b/d carry their colourbars. The letter is
    # re-fractioned so "(f)" lands exactly under "(b)"/"(d)".
    fig.canvas.draw()
    b_ax, f_ax = axes[1], axes[5]
    bp = b_ax.get_position()
    fb = f_ax.get_window_extent(
        fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    f_ax.set_box_aspect(None)      # else the next draw re-shrinks width to the old aspect
    f_ax.set_position([bp.x0, fb.y0, bp.width * 1.10, fb.height])
    f_ax.texts[0].set_x(-0.14 / 1.10)   # texts[0] = the "(f)" from label_panel
    base = str(FIG_DIR / "Fig3_PEP_spatial")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(FIG_DIR / "pdf" / "Fig3_PEP_spatial.pdf", dpi=600)
    print(f"[save] Fig3_PEP_spatial  7.2×8.6 in -> "
          f"Fig3_PEP_spatial.png")
    plt.close(fig)
    return {"width_in": 7.2, "height_in": 8.6}


def render_panels(d, outdir):
    """Each panel standalone to PNG with an identical canvas size for inspection."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    s = derive(d)
    proj = ccrs.PlateCarree()
    SIZE = (2.6, 2.6 / GEO_ASPECT * 1.05)        # identical canvas per panel
    with mpl.rc_context({"savefig.bbox": "standard"}):
        for letter, fn, geo in PANELS:
            fig = plt.figure(figsize=SIZE)
            gs = fig.add_gridspec(1, 2, width_ratios=[1, CBAR_W])
            gs.update(left=0.02, right=0.97, top=0.96, bottom=0.04, wspace=0.01)
            ax = (fig.add_subplot(gs[0, 0], projection=proj) if geo
                  else fig.add_subplot(gs[0, 0]))
            cax = fig.add_subplot(gs[0, 1])
            fn(ax, d, s, proj, cax)
            p = outdir / f"fig3_{letter}.png"
            fig.savefig(p, dpi=600); plt.close(fig)
            print(f"  panel {letter} -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", action="store_true",
                    help="render each panel standalone to figures/_panels/")
    args = ap.parse_args()
    d = load()
    if args.panels:
        render_panels(d, FIG_DIR / "_panels")
        return
    compose(d)


if __name__ == "__main__":
    main()
