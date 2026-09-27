#!/usr/bin/env python3
"""Fig. S8 - leave-one-out and trend robustness.

Reads figure_data/figS8_loo.csv (= Table S4), figS8_trend.csv and
figS8_spatial.npz. Output: figures/FigS8_leave_one_out.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

from _style import (natural_save, label_panel, add_china, halo_text, C_PER,
                    FS_PANEL, FS_AXIS, FS_TICK, FS_TITLE, FS_ANNOT,
                    FS_GRID, FS_CBAR_LAB, FS_CBAR_TICK)
FD = Path(__file__).resolve().parents[1] / "figure_data"
EXTENT = [95, 134, 16, 50]


# room left of the data. Display-only: the annotated stats in _maps() still slice
# the ORIGINAL EXTENT, so every printed number (mean / max / "% grid |Δ|<1%")
# stays bit-identical.
VIEW = [98, 134, 16, 50]
CBAR_W = 0.05                          # e/f cbar width ratio (house style, cf. Fig3/Fig4)


def _pfmt(p):
    """Journal-style p string with leading zero dropped (.022, <.001)."""
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def _scs_inset(parent):
    """South-China-Sea inset (lower-right of a China map): nine-dotted line,
    the standard cartographic element for China figures."""
    proj = ccrs.PlateCarree()


    axi = parent.inset_axes([0.79, 0.002, 0.21, 0.3], projection=proj)
    axi.set_extent([104, 123, 0, 25], crs=proj)
    add_china(axi, provinces=False, nine_dotted=True)
    for sp in axi.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("0.35")
        sp.set_linewidth(0.5)     # plan: harmonized with the other insets
    return axi


def _loo_strip(ax, df, col, full, ylabel, letter, unit, scale=1.0):
    """Leave-one-out deviation strip: Δ = LOO − full, vs replicate index.

    scale: display multiplier — (b) passes 1e3 and puts (×10⁻³) in the ylabel so
    the ticks are plain mantissas hugging the axis title; the mpl ×10⁻³ offset
    text it replaces pushed the ylabel ~12 pt off the ticks (measured render,
    """
    dev = (df[col].values - full) * scale
    xs = np.arange(len(df))
    maxabs = float(np.max(np.abs(dev)))          # scaled — drives band/ylim/tags
    raw = maxabs / scale                         # raw units — for the halo note
    pct = 100.0 * raw / abs(full) if full else 0.0

    # light LOO envelope ± zero reference (plan: band even more transparent)
    ax.axhspan(-maxabs, maxabs, color=C_PER, alpha=0.08, zorder=1)
    ax.axhline(0, color="0.30", lw=1.1, zorder=2)
    ax.scatter(xs, dev, s=16, c=C_PER, alpha=0.85, edgecolor="white",
               linewidth=0.3, zorder=3)

    halo_text(ax, 0.03, 0.97,
              f"full = {full:g}{unit}\nmax |$\\Delta$| = {raw:g}{unit}  ({pct:.1f}%)",
              fs=FS_ANNOT, color="0.12", va="top", ha="left")

    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.set_xlabel(f"LOO replicate (n = {len(df)})", fontsize=FS_AXIS)
    ax.set_xticks([])
    ax.tick_params(labelsize=FS_TICK)
    ax.locator_params(axis="y", nbins=8)
    ax.margins(x=0.03)
    ax.set_ylim(-1.25 * maxabs, 1.80 * maxabs)
    # Heavily-tied statistic (e.g. 6-hour-binned RT median) => the dots sit on
    # a few horizontal lines. Tag each line with its count so the stripes read
    # as real tied data, not a rendering artefact.
    udev = np.unique(np.round(dev, 3))
    if len(udev) <= 4:
        for val in udev:
            n = int(np.sum(np.abs(dev - val) < 1e-6))
            halo_text(ax, len(df) - 1.5, val + 0.04 * maxabs, f"n = {n}",
                      fs=FS_ANNOT, color="0.1", ha="right", va="bottom",
                      transform=ax.transData)
    label_panel(ax, letter)


def _trend_bars(ax, tr):
    """(d) PEP A_e trend (Theil–Sen) after removing top-1/3/5 accumulation events.

    Uncertainty is conveyed by the Mann–Kendall p (printed under each bar) —
    NOT by error bars: the slope is Theil–Sen and the p is rank-based, so a
    parametric CI cannot be recovered from `Ae_p`. Bars share a common baseline
    of zero so the (small) between-bar differences are not exaggerated.
    """
    from matplotlib.transforms import blended_transform_factory
    bars = tr["Ae_trend"].values
    pvals = tr["Ae_p"].values
    xpos = np.arange(len(tr))
    full = float(bars[0])
    colors = ["0.6"] + [C_PER] * (len(tr) - 1)

    ax.bar(xpos, bars, color=colors, width=0.62, zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.axhline(full, color="0.35", lw=0.9, ls="--", zorder=2)
    for i, v in enumerate(bars):
        ax.text(i, v + 0.22, f"{v:.1f}", ha="center", fontsize=FS_ANNOT)

    ax.set_xticks(xpos)
    # plan Fig.S8d: p shown as "p=…" (values untouched)
    ax.set_xticklabels(
        [f"{lab}\n($p$={_pfmt(p)})" for lab, p
         in zip(["full", "−1", "−3", "−5"], pvals)],
        fontsize=FS_TICK)
    ax.set_ylabel("PEP $A_e$ trend (mm dec$^{-1}$)", fontsize=FS_AXIS)
    ax.set_title("Trend robustness", fontsize=FS_TITLE, pad=3)
    ax.tick_params(labelsize=FS_TICK)


    ax.set_ylim(bottom=8.0, top=float(np.max(bars)) + 1.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    # tag the full-sample reference line just ABOVE its right end, INSIDE the

    # the panel); white halo keeps it legible where it overlaps the −5 bar
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    halo_text(ax, 0.99, full + 0.10, "full", fs=FS_ANNOT, color="0.35",
              ha="right", va="bottom", transform=trans)
    ax.text(0.98, 0.97, "Theil–Sen slope\n$p$: Mann–Kendall",
            transform=ax.transAxes, fontsize=FS_ANNOT, color="0.3",
            ha="right", va="top")
    label_panel(ax, "d")
    return ax


def _maps(fig, cells, sp, align_to):
    """(e) full PEP rain contribution; (f) footprint of the single top-
    accumulation event (= full minus leave-that-event-out), on its own fine
    scale with its own colourbar.

    Shown as two same-field maps, (e) and (f) look near-identical — removing
    one event barely moves the field (99.1% of PEP grid cells change <1%).
    So (f) plots the *difference* instead: where the extreme event actually
    deposited rain, up to 3.6% of local rainfall in any one cell. This is both
    more legible and a stronger robustness statement than two twin maps.
    """
    proj = ccrs.PlateCarree()
    lat = sp["lat"]; lon = sp["lon"]
    LON, LAT = np.meshgrid(lon, lat)
    cf = sp["contrib_full"]; cl = sp["contrib_loo"]
    diff = cf - cl                              # >= 0 everywhere: top-event contrib

    iy = np.where((lat >= EXTENT[2]) & (lat <= EXTENT[3]))[0]
    ix = np.where((lon >= EXTENT[0]) & (lon <= EXTENT[1]))[0]
    region = lambda a: a[iy[:, None], ix[None, :]]
    mean_full = float(np.nanmean(region(cf)[region(cf) > 0]))
    max_diff = float(np.nanmax(region(diff)))
    frac_lt1 = float((np.abs(region(diff))[region(cf) > 0] < 1.0).mean())

    fields = [cf, diff]


    titles = ["PEP rain contribution (all events)",
              "Top-accumulation event footprint"]
    # plan Fig.S8f: OrRd low end lifted 10% so the faint swath reads on white;

    from matplotlib.colors import LinearSegmentedColormap
    orr_lite = LinearSegmentedColormap.from_list(
        "OrRd_lite", plt.get_cmap("OrRd")(np.linspace(0.10, 1.0, 256)))
    cmaps = ["YlGnBu", orr_lite]
    # (e) full PEP contribution: bulk is 10-30%, p98≈51, tail reaches 100% (PEP-
    # only cells). vmax=30 emphasises the 10-30% body and high-contribution core;
    # the >30% tail (coastal hotspots, PEP-only cells) saturates.
    # (f) tight range: footprint is mostly <1%; vmax=0.2 brings out the swath,
    # cells above 0.2% (the TC core) saturate at the top.
    vmaxs = [30.0, 0.2]
    cblbl = ["PEP rain contrib. (%)", r"$\Delta$ contrib. (%)"]
    notes = [f"mean {mean_full:.1f}%",
             f"max {max_diff:.1f}%\n{100 * frac_lt1:.1f}% grid $|\\Delta|<1$%"]
    letters = ["e", "f"]

    map_axes = []
    caxes = []
    cbs = []
    for k in range(2):
        ax = fig.add_subplot(cells[k], projection=proj)
        # top-anchor: map box hangs from the top of its cell => (e)(f) panel
        # labels (label_panel, just above the box) stay clear of the (c)(d)
        # row; the freed height collects below (tight-cropped).
        # NO set_box_aspect here: combined with the colourbar divider below it
        # flips the axes to adjustable="datalim", which silently STRETCHES the
        # drawn lat range past set_extent ([12.7, 53.3] instead of [16, 50],

        # the GeoAxes settles at the data aspect and honours EXTENT exactly.
        ax.set_anchor("N")

        ax.title.set_visible(False); ax._right_title.set_visible(False)
        field = np.ma.masked_where(fields[k] <= 0, fields[k])
        im = ax.pcolormesh(LON, LAT, field, cmap=cmaps[k], vmin=0,
                           vmax=vmaxs[k], shading="auto", transform=proj, zorder=2)
        add_china(ax); ax.set_extent(VIEW, crs=proj)   # VIEW: display frame; stats use EXTENT
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
        gl.top_labels = False; gl.right_labels = False
        gl.left_labels = True           # both maps carry their own lat/lon labels
        gl.bottom_labels = True
        gl.xlocator = mticker.MultipleLocator(15)
        gl.xlabel_styles = gl.ylabel_styles = {"fontsize": FS_GRID}
        ax.set_title(titles[k], fontsize=FS_TITLE, pad=2, y=1.0)   # y=1.0: cartopy 0.25 + mpl 3.11 autotitlepos workaround

        halo_text(ax, 0.03, 0.95, notes[k], fs=FS_ANNOT, color="0.1",
                  va="top", ha="left")
        label_panel(ax, letters[k])
        _scs_inset(ax)


        # Fig3/Fig4. The cax is a PLAIN independent axes pinned below, NOT a
        # divider child: divider/locator positions are not stable across the
        # measure+render passes of bbox_inches="tight", which double-rendered

        cax = fig.add_axes([0.0, 0.0, 0.01, 0.1], label=f"cbar_{letters[k]}")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=2)
        cb.set_label(cblbl[k], fontsize=FS_CBAR_LAB, labelpad=3)
        cb.locator = mticker.MaxNLocator(nbins=5)
        cb.update_ticks()
        map_axes.append(ax)
        caxes.append(cax)
        cbs.append(cb)


    # via labelpad so both maps shrink by the same amount and stay equal in
    # width; the GeoAxes re-settles its height from the new width at the next

    # nothing re-derives it in the tight-bbox passes.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = 0.05 / fig.get_size_inches()[0]
    foots = [inv.transform((cb.ax.yaxis.label.get_window_extent(r).x1, 0.0))[0]
             - cax.get_position().x1 for cax, cb in zip(caxes, cbs)]
    fmax = max(foots)
    for cb, f_ in zip(cbs, foots):                 # top up the shorter label's pad
        if fmax - f_ > 1e-4:
            axy = cb.ax.yaxis
            axy.labelpad = axy.labelpad + (fmax - f_) * 72.0 * fig.get_size_inches()[0]
    over = max(ax.get_position().x1 + pad + CBAR_W * ax.get_position().width + fmax
               - ref.get_position().x1
               for ax, ref in zip(map_axes, align_to))
    over /= 1.0 + CBAR_W    # feedback: the cbar shrinks WITH the map (CBAR_W×new
                            # width), so W₀−over overshoots the needed trim by

    for ax, cax in zip(map_axes, caxes):
        mf = ax.get_position()
        ax.set_position([mf.x0, mf.y0, mf.width - over, mf.height])
    fig.canvas.draw()                              # re-settle aspect boxes
    for ax, cax in zip(map_axes, caxes):
        mf = ax.get_position()                     # settled GeoAxes box
        cax.set_position([mf.x1 + pad, mf.y0, CBAR_W * mf.width, mf.height])
    return map_axes


def main():
    loo = pd.read_csv(FD / "figS8_loo.csv")
    tr = pd.read_csv(FD / "figS8_trend.csv")
    sp = np.load(FD / "figS8_spatial.npz", allow_pickle=True)

    fig = plt.figure(figsize=(6.6, 8.6))                     # was: (8.6, 5.7)

    # the c/d row and the e/f maps): two stacked blocks instead of one 4-row grid,
    # so the strips↔maps gap (outer hspace) is tunable independently of the a↔c
    # row gap (inner, unchanged 0.17). The maps hang at their block's TOP (anchor
    # N), so the outer gap must house d's two-line tick stack + e/f's title
    # stacks; slack left over in the maps block is tight-cropped away. Per-block
    # wspace: the strips rows' gutter hosts b/d's rotated ylabels (0.24), the map
    # row's hosts e's full colourbar stack + f's lat labels (0.27). left

    # the freed width offsets the wider gutters, so the maps stay ~64 mm wide.


    # it only has to house d's two-line tick stack + e/f's title/letter stacks;
    # slack left at the maps-block bottom is tight-cropped away.
    gs_all = fig.add_gridspec(2, 1, left=0.10, right=0.97, top=0.95, bottom=0.06,
                              height_ratios=[2, 1.75], hspace=0.14)
    gs_top = gs_all[0].subgridspec(2, 2, wspace=0.20, hspace=0.17)
    gs_map = gs_all[1].subgridspec(1, 2, wspace=0.20)

    fullRT = float(loo.medianRT_PER.median())
    fullRD = float(loo.corr_RTD.median())
    fullRA = float(loo.corr_RTA.median())

    # (a-c) LOO deviation strips
    _loo_strip(fig.add_subplot(gs_top[0, 0]), loo, "medianRT_PER", fullRT,
               r"$\Delta$ PEP median $RT_{local}$ (h)", "a", " h")
    _loo_strip(fig.add_subplot(gs_top[0, 1]), loo, "corr_RTD", fullRD,
               r"$\Delta$ corr($RT_{local}$, $D$) ($\times 10^{-3}$)", "b", "",
               scale=1e3)
    axc = fig.add_subplot(gs_top[1, 0])
    _loo_strip(axc, loo, "corr_RTA", fullRA,
               r"$\Delta$ corr($RT_{local}$, $A$)", "c", "")

    # (d) trend robustness
    axd = fig.add_subplot(gs_top[1, 1])
    _trend_bars(axd, tr)

    # (e-f) maps in the bottom block (full block height each); each map+cbar
    # stack is sized to end at its column's top-panel right edge (c / d)
    axe = _maps(fig, [gs_map[0, 0], gs_map[0, 1]], sp, align_to=[axc, axd])

    return natural_save(fig, "FigS8_leave_one_out")


if __name__ == "__main__":
    main()
