"""Shared publication style for all figures.

Okabe-Ito palette plus semantic colours, a uniform font-size ladder, a layered
cartopy China basemap (add_china, nine-dotted line drawn last) that keeps data
fields visible under the land patch, and save()/natural_save() which enforce
journal column widths (Nature double column, 183 mm, by default). All style
parameters are set in this module; export helpers in _figure_export.py; China
shapefiles in assets/shapefiles.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

CJK = "Noto Sans CJK SC"

# Base publication rcParams (Nature-family conventions): sans-serif, minimal
# chart junk, editable vector text, Okabe-Ito colour cycle.
_BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "figure.titlesize": 8.0,
    "axes.prop_cycle": mpl.cycler("color", [
        "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7"]),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,
    "figure.figsize": [3.5, 2.6],
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def set_style():
    """Apply the base publication rcParams; enforce CJK + editable vector text.

    geometry normalized to a global table (axis 0.7 pt, outward ticks 2.8 pt)
    and the sans-serif chain gains Liberation Sans — the metric-compatible
    Arial substitute — ahead of the CJK fallback so Latin text resolves to an
    Arial-look face on hosts without real Arial.
    """
    mpl.rcParams.update(_BASE_RC)
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans",
                                       CJK, "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.dpi"] = 600
    # Global line table: axes 0.6-0.8 pt, outward ticks 2.5-3 pt uniform.
    # Base sizes are overridden so DEFAULT text also follows the ladder
    # (explicit FS_* call sites win over rcParams as before).
    mpl.rcParams["font.size"] = 8.0
    mpl.rcParams["axes.labelsize"] = 9.0
    mpl.rcParams["axes.titlesize"] = 9.5
    mpl.rcParams["xtick.labelsize"] = 8.0
    mpl.rcParams["ytick.labelsize"] = 8.0
    mpl.rcParams["legend.fontsize"] = 8.0
    mpl.rcParams["figure.titlesize"] = 10.5
    mpl.rcParams["axes.linewidth"] = 0.7
    for _side in ("xtick", "ytick"):
        mpl.rcParams[f"{_side}.direction"] = "out"
        mpl.rcParams[f"{_side}.major.width"] = 0.7
        mpl.rcParams[f"{_side}.major.size"] = 2.8
        mpl.rcParams[f"{_side}.minor.width"] = 0.5
        mpl.rcParams[f"{_side}.minor.size"] = 1.6


set_style()

# ---- Okabe-Ito colour-blind-safe palette ----
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]


C_PER   = "#D55E00"   # vermilion — PER events
C_NPER  = "#888888"   # grey      — non-PER
C_NORMAL = "#56B4E9"  # blue      — normal TC
C_RT    = "#0072B2"   # blue      — residence time
C_DYN   = "#0072B2"   # dynamic pathway
C_THE   = "#D55E00"   # thermodynamic pathway
C_INTER = "#009E73"
C_ACCENT = "#E69F00"

SEQ_CMAP = "magma"          # colour-blind-safe sequential (rainfall)
DIVERGE_CMAP = "PuOr"       # divergence / anomaly


SMOOTH_SIGMA = 1.8   # sigma in grid cells; display only, never feeds statistics


def smooth_field(z, sigma=SMOOTH_SIGMA):
    """NaN-aware Gaussian smoothing for DISPLAY ONLY.

    scipy.ndimage.gaussian_filter propagates NaNs, so: fill NaN cells with the
    field's nanmean, filter, then re-mask the original NaN cells. Returns a new
    array; the input is untouched. Vector/quiver fields are never smoothed.
    """
    from scipy.ndimage import gaussian_filter
    z = np.asarray(z, float)
    fin = np.isfinite(z)
    if not fin.any():
        return z.copy()
    filled = np.where(fin, z, np.nanmean(z))
    sm = gaussian_filter(filled, sigma=sigma, mode="nearest")
    return np.where(fin, sm, np.nan)


# ---- unified font sizes (single source of truth across all 14 figures) ----

# FINAL placement size (183 mm figures placed at 183 mm; factor 1.0):
#   axis titles 9 pt / ticks·legend·cbar·annotations 8 pt / panel titles
#   9–10 pt / panel letters 10–11 pt. Fig4 & FigS7 (tall figures designed for
#   ~134/131 mm final width) apply FS_TALL = 1.07 on top of these values.
FS_PANEL     = 10.5  # (a)(b) panel letter
FS_TITLE     = 9.5   # descriptive panel title
FS_AXIS      = 9.0   # axis xlabel / ylabel
FS_TICK      = 8.0   # tick labels
FS_ANNOT     = 8.0   # in-axes annotations (slope / p / n / r)
FS_LEGEND    = 8.0   # legend text
FS_CBAR_LAB  = 8.0   # colorbar title/label
FS_CBAR_TICK = 8.0   # colorbar tick labels
FS_GRID      = 8.0   # cartopy gridline lat/lon labels
FS_TALL      = 1.07
HALO_LW      = 2.4   # white-halo stroke width for annotations over busy backgrounds

# ---- shapefiles (project-local copy; China-map compliance) ----
SHP_DIR = str(ROOT / "assets" / "shapefiles")
SHP_CHINA = os.path.join(SHP_DIR, "china_country.shp")
SHP_PROVINCE = os.path.join(SHP_DIR, "china.shp")
SHP_NINE_DOTTED = os.path.join(SHP_DIR, "china_nine_dotted_line.shp")

# Provisional venue (verified spec; resize on user decision)
TARGET_JOURNAL = "nature"
COLUMN_DOUBLE = "double"
COLUMN_SINGLE = "single"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
PDF_DIR = FIG_DIR / "pdf"
PDF_DIR.mkdir(exist_ok=True)


def label_panel(ax, letter, x=-0.14, y=1.02, weight="bold", fs=FS_PANEL):
    """Bold (a)(b)(c) panel label, top-left of axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fs, fontweight=weight, va="bottom", ha="left")


def panel_title(ax, letter, text, *, title_fs=FS_TITLE, label_fs=FS_PANEL,
                pad=4, loc="left"):
    """Bold (letter) panel marker + short descriptive title in one call (unified style)."""
    label_panel(ax, letter, fs=label_fs)
    ax.set_title(text, fontsize=title_fs, pad=pad, loc=loc)


def halo_text(ax, x, y, s, *, fs=FS_ANNOT, lw=HALO_LW, color="0.1",
              halo="white", transform=None, **kw):
    """ax.text with a white halo (path-effects) — legible over busy map/scatter backgrounds.

    x, y default to axes-fraction (transAxes); pass transform=ax.transData for data coords.
    """
    import matplotlib.patheffects as pe
    if transform is None:
        transform = ax.transAxes
    t = ax.text(x, y, s, transform=transform, fontsize=fs, color=color, **kw)
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground=halo)])
    return t


def add_colorbar(mappable, label, *, cax=None, ax=None, nticks=5, integer=False,
                 diverging=False, pad=0.05, fraction=0.046, label_fs=FS_CBAR_LAB,
                 tick_fs=FS_CBAR_TICK, labelpad=1.0, tick_length=2):
    """Uniform colorbar with controlled tick density.

    cax : pre-placed colorbar axes (Fig.3/4 style via make_axes_locatable)
          -> fig.colorbar(mappable, cax=cax)
    ax  : steal space from this axes (Fig.1 style)
          -> fig.colorbar(mappable, ax=ax, pad=pad, fraction=fraction)

    nticks/integer -> MaxNLocator(nbins=nticks, integer=integer), killing the per-panel
    tick-density drift where default locators gave uneven ticks across a row.
    diverging -> even ticks spanning the full clim (clean for TwoSlopeNorm).
    """
    from matplotlib.ticker import MaxNLocator
    fig = (cax if cax is not None else ax).figure
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax)
    else:
        cb = fig.colorbar(mappable, ax=ax, pad=pad, fraction=fraction)
    cb.set_label(label, fontsize=label_fs, labelpad=labelpad)
    cb.ax.tick_params(labelsize=tick_fs, length=tick_length)
    try:
        if diverging:
            vmin, vmax = cb.norm.vmin, cb.norm.vmax
            cb.set_ticks(np.linspace(vmin, vmax, nticks))
        else:
            cb.locator = MaxNLocator(nbins=nticks, integer=integer)
            cb.update_ticks()
    except Exception:
        pass
    return cb


# Cache shapefile geometries at module level so add_china does not re-read and
# re-parse the .shp on every call (matters for batch renders of 1000+ panels).
_SHP_GEOM_CACHE = {}


def _cached_geoms(path):
    g = _SHP_GEOM_CACHE.get(path)
    if g is None:
        from cartopy.io.shapereader import Reader
        g = list(Reader(path).geometries())
        _SHP_GEOM_CACHE[path] = g
    return g


def add_china(ax, facecolor=None, edgecolor=None, lw=0.5,
              ocean="#D6E8F5", land="#F4F1EA", edge="#595959",
              land_alpha=0.55, provinces=True, nine_dotted=True):
    """Layered China basemap on a cartopy GeoAxes.

    Transparency scheme so map AND data field are both visible:
        ocean face          zorder 0   (background_patch)
        land fill (faint)   zorder 0.5 (BELOW data)
        --- data layer (caller, default zorder >=1) ---
        coastline outline   zorder 4   (ABOVE data — crisp)
        province borders    zorder 4.1
        nine-dotted line    zorder 6   (topmost)

    (z4.1), coastline a step darker (#595959, 0.5 pt) — fine enough not to
    smother data texture.
    """
    from cartopy.io.shapereader import Reader
    import cartopy.crs as ccrs
    if facecolor:
        land = facecolor
    if edgecolor:
        edge = edgecolor
    crs = ccrs.PlateCarree()
    try:
        ax.background_patch.set_facecolor(ocean)
    except Exception:
        pass
    geoms = _cached_geoms(SHP_CHINA)
    ax.add_geometries(geoms, crs=crs, facecolor=land, edgecolor="none",
                      alpha=land_alpha, zorder=0.5)
    ax.add_geometries(geoms, crs=crs, facecolor="none", edgecolor=edge,
                      linewidth=lw, zorder=4.0)
    if provinces:
        ax.add_geometries(_cached_geoms(SHP_PROVINCE), crs=crs,
                          facecolor="none", edgecolor="#9aa0a6", linewidth=0.35, zorder=4.1)
    if nine_dotted:
        ax.add_geometries(_cached_geoms(SHP_NINE_DOTTED), crs=crs,
                          facecolor="none", edgecolor=edge, linewidth=lw * 0.9, zorder=6)


def save(fig, basename, column=COLUMN_DOUBLE, journal=TARGET_JOURNAL, checks=True):
    """Save figure via save_for_journal (locks physical mm) + run compliance checks.

    figures/pdf/ — text/axes/boxes stay vector; embedded rasters keep native
    data resolution. Returns info dict.
    """
    from _figure_export import (save_for_journal, save_publication_figure,
                               check_figure_size, check_export_compliance)
    base = str(FIG_DIR / basename)
    # These figures already have explicit, reviewed axes positions. The generic
    # exporter enables constrained_layout, which replaces those positions and
    # can clip manually placed titles and colorbars. Lock width only here.
    from _figure_export import JOURNAL_SPECS
    width_mm = JOURNAL_SPECS[journal][f"{column}_mm"]
    old_w, old_h = fig.get_size_inches()
    fig.set_layout_engine(None)
    fig.set_size_inches(width_mm / 25.4, width_mm / 25.4 * old_h / old_w)
    paths = save_publication_figure(fig, base, formats=("png",),
                                     dpi=600, bbox_inches=None)
    info = {"width_mm": width_mm, "dpi": 600}
    # fig is now sized to the locked column width -> exact-width vector copy.
    save_publication_figure(fig, str(PDF_DIR / basename),
                            formats=("pdf",), dpi=600, bbox_inches=None)
    if checks:
        cs = check_figure_size(fig, journal=journal, column=column,
                               measured=True, path=paths[0])
        info["size_check"] = cs
        for p in paths:
            ce = check_export_compliance(p, journal=journal, column=column)
            info.setdefault("compliance", []).append((os.path.basename(p), ce))
    print(f"[save] {basename}  column={column}  -> {[os.path.basename(p) for p in paths]}")
    if info.get("font", {}).get("fallback"):
        print(f"  ⚠ font fallback: {info['font']}")
    return info


def natural_save(fig, basename, *, pad_inches=0.04, tight=True):
    """Natural-width save: no journal column lock, so the content sets the width.

    tight=True (default) uses bbox_inches='tight' to strip surrounding whitespace.
    Works for regular subplots / gridspec / cartopy panels. For ImageGrid figures
    pass tight=False — bbox tight is incompatible with ImageGrid, so fill the figure
    via the grid's `rect` and save standard instead.
    """
    out = FIG_DIR / f"{basename}.png"
    pdf = PDF_DIR / f"{basename}.pdf"
    if tight:
        fig.savefig(out, bbox_inches="tight", pad_inches=pad_inches, dpi=600)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=pad_inches, dpi=600)
    else:
        with mpl.rc_context({"savefig.bbox": None}):
            fig.savefig(out, dpi=600)
            fig.savefig(pdf, dpi=600)
    print(f"[natural_save] {basename} -> {out.name} (+ pdf/{pdf.name})")
    return {"path": str(out)}


def kde_1d(x, xlim, n=200, bw=None):
    """Return (grid, density) for a 1-D Gaussian KDE of x within xlim."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return None, None
    try:
        k = gaussian_kde(x, bw_method=bw)
    except Exception:
        return None, None
    g = np.linspace(xlim[0], xlim[1], n)
    return g, k(g)
