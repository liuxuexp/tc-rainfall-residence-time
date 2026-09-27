#!/usr/bin/env python3
"""Fig. 1 - What is TC-PEP?

(a) yearly rainfall amount and event count; (b) duration-accumulation space
coloured by residence time; (c) case storms Saomai 2006 (non-PEP) vs In-fa
2021 (PEP) on a shared China frame; (d) schematic of TC motion organising
rainfall events (embedded raster fig1-d-body.png).

Panels are drawn independently on a 2x2 grid of equal block widths and
merged; run with --panels to dump each panel to figures/_panels/.

Data: data/tc_events_full.csv, tc_event_day_map.csv, tc_prec_yearly/*.nc,
tc_track_table.csv, tc_footprint_yearly/*.npz.
Output: figures/Fig1_PEP_definition.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from _style import (save, add_china, C_PER, C_NORMAL, FIG_DIR,
                    FS_TITLE, FS_AXIS, FS_ANNOT, FS_LEGEND, FS_CBAR_LAB,
                    FS_CBAR_TICK, FS_GRID)
from config import DATA_DIR


# "(a)" aligns with the column's leftmost ink (the y-axis labels). Left column


TITLE_KW = dict(fontsize=FS_TITLE, pad=6, loc="left", x=-0.158)


def panel_title(ax, letter, text, **over):
    """Panel label + title merged onto ONE line: bold '(a)' then the title text,
    left-aligned. Replaces the old separate label_panel + set_title pair."""
    kw = dict(TITLE_KW); kw.update(over)
    ax.set_title(fr"$\bf{{({letter})}}$  {text}", **kw)


# cases: normal vs PEP (clean persistence contrast)
CASES = [
    dict(name="Saomai", year=2006, role="Fast"),  # D=3, A=125, RT=66 h, vt_min=17 (intense but short, non-PEP)
    dict(name="In-fa",  year=2021, role="PEP"),   # D=9, A=229, RT=192 h (Zhengzhou)
]


# empty 18-22°N ocean band so the case storms fill the frame).
CASE_DOMAIN = [105.0, 125.0, 22.0, 47.0]
MAP_ASPECT = (CASE_DOMAIN[3] - CASE_DOMAIN[2]) / (CASE_DOMAIN[1] - CASE_DOMAIN[0])  # 25/20 = 1.25
LON_TICKS = [110, 120]  # readable at the narrow case-map width
LAT_TICKS = [22, 30, 40, 47]

# rainfall cmap (YlGnBu, masked <1 mm so map bg shows through)
from matplotlib.colors import LinearSegmentedColormap, to_rgb
_base = plt.get_cmap("YlGnBu")
_bg = np.array(to_rgb("#EDF1F5"))
_N, _t0 = 512, 0.12
_p = np.linspace(0, 1, _N)
_low = _bg + (np.array(_base(0.0)[:3]) - _bg) * (_p / _t0)[:, None]
_hi = np.array([_base(v)[:3] for v in (_p - _t0) / (1 - _t0)])
RAIN_CMAP = LinearSegmentedColormap.from_list("rain_ygb",
            np.where((_p < _t0)[:, None], _low, _hi), N=_N)


# ===========================================================================
#  data helpers
# ===========================================================================
def load_events():
    df = pd.read_csv(DATA_DIR / "tc_events_full.csv")
    return df.dropna(subset=["RT_e_hours", "accumulation_Ae_mm", "duration_days"])


_FOOT_CACHE = {}   # year -> {"{tc_id}_{YYYYMMDD}": bool 2D mask}


def _tc_masks(year):
    """Per-TC-day footprint masks for one year (cached npz)."""
    m = _FOOT_CACHE.get(year)
    if m is None:
        fp = DATA_DIR / "tc_footprint_yearly" / f"tc_footprint_{year}.npz"
        if not fp.exists():
            m = {}
        else:
            z = np.load(fp, allow_pickle=True)
            m = {k: z[k] for k in z.files}
        _FOOT_CACHE[year] = m
    return m


def storm_total(tc_id, start, end):
    """Sum THIS TC's daily rainfall over its event dates (field2d, lat, lon).

    Uses the per-TC footprint masks (data/tc_footprint_yearly), NOT the daily
    tc_prec_yearly union field. The union field blends all coexisting TCs' 500 km
    circles into one field per day, so summing it would attribute a co-TC's rain
    to this TC (e.g. In-fa 2021 picks up ~2.7% of Cempaka's rain on the overlap
    days). Masking the field by this TC's own footprint keeps only its rain.
    """
    dm = pd.read_csv(DATA_DIR / "tc_event_day_map.csv", dtype={"date": str})
    dates = dm[(dm.tc_id == tc_id)].date.tolist() if "tc_id" in dm.columns else None
    if not dates:
        return None, None, None
    dates = [str(int(float(d))) for d in dates]
    acc = None; lat = lon = None
    for d in dates:
        y = int(d[:4])
        ncfp = DATA_DIR / "tc_prec_yearly" / f"tc_prec_{y}.nc"
        if not ncfp.exists():
            continue
        mask = _tc_masks(y).get(f"{int(tc_id)}_{d}")
        if mask is None:
            continue   # this TC had no footprint that day -> contributes nothing
        ds = xr.open_dataset(ncfp)
        if lat is None:
            lat = ds["lat"].values; lon = ds["lon"].values
        f = ds["tc_prec"].sel(time=pd.Timestamp(d)).values.astype(float)
        ds.close()
        f = np.where(mask & np.isfinite(f), f, 0.0)
        acc = f if acc is None else acc + f
    if acc is None:
        return None, None, None
    return acc, lat, lon


def load_track(tc_id):
    tt = pd.read_csv(DATA_DIR / "tc_track_table.csv", dtype={"date": str})
    g = tt[tt.tc_id == tc_id].copy()
    g["utc"] = pd.to_datetime(g["datetime_utc"], errors="coerce")
    g = g.dropna(subset=["utc"]).sort_values("utc").reset_index(drop=True)
    t0 = g["utc"].iloc[0]
    g["hours"] = (g["utc"] - t0).dt.total_seconds() / 3600.0
    g["day8"] = g["date"].str.slice(0, 8).astype("int64", errors="ignore")
    return g


def event_mask(g, start, end):
    s, e = int(str(start)[:8]), int(str(end)[:8])
    return (g["day8"] >= s) & (g["day8"] <= e)


# ===========================================================================


#  distortion/letterbox, and (c)'s cropped case maps (lat 22-47, aspect 0.80)

# ===========================================================================
G_FIG = (7.2, 6.37)

                                         #  keeps its physical size, the white is cropped)
_G = dict(left=0.07, right=0.074, top=0.94, bottom=0.063, gx=0.068,
          gy_l=0.095, gy_r=0.096)

                                         #  ~2.2 mm both columns, calibrated on the render)
_COL_W = (1 - _G["left"] - _G["right"] - _G["gx"]) / 2.0          # 0.394
_COL_L = [_G["left"], _G["left"] + _COL_W + _G["gx"]]            # [0.07, 0.532]

# title headroom. Row gaps are tuned PER COLUMN (a–c tighter than b–d), so the
# two top-row blocks share their top edge (_TOP_T, the title baseline grid) but
# not their bottom edge; the freed canvas space stays in the a/b panels
# (free-aspect axes). _BOT_H is shared by c/d; the (c) maps' exact aspect is
# enforced inside c_subrects (pinned to the block top), so _BOT_H is now a

# bottom, below the c bar's own tick/title labels.
_TOP_T = 0.957
_BOT_H = 0.312
_BOT_B = _G["bottom"]                                           # 0.063 (~1.5 mm under the c cb title ink)
_BOT_T = _BOT_B + _BOT_H                                        # 0.375

A_BLOCK = [_COL_L[0], _BOT_T + _G["gy_l"], _COL_W, _TOP_T - (_BOT_T + _G["gy_l"])]
B_RECT  = [_COL_L[1], _BOT_T + _G["gy_r"], _COL_W, _TOP_T - (_BOT_T + _G["gy_r"])]
C_BLOCK = [_COL_L[0], _BOT_B, _COL_W, _BOT_H]   # bottom-left (c/d same height)
D_RECT  = [_COL_L[1], _BOT_B, _COL_W, _BOT_H]   # bottom-right


# ===========================================================================

# ===========================================================================
def a_subrects(block):
    """Split the (a) block [l,b,w,h] into the two stacked sub-axes (amount top, count bottom)."""
    l, b, w, h = block
    gap = 0.012                      # vertical gap between the two sub-panels

                                      #   itself shrank by the same 0.013 via gy_l, so the

    sh = (h - gap) / 2.0
    return [l, b + sh + gap, w, sh], [l, b, w, sh]


def c_subrects(block, fig_size=G_FIG):
    """Split the (c) block [l,b,w,h] into (c1 map, c2 map, colorbar).

    Maps are pinned to the block TOP at their EXACT geographic aspect
    (CASE_DOMAIN 20°x25° → h/w = 1.25 in inches). The old fill-the-block rect
    was ~13% too tall, so cartopy letterboxed ~62 px of white above AND below
    the title–map and map–colorbar air). The colourbar hugs the lon tick
    labels (lon_lab below the maps) instead of sitting at the block bottom.
    """
    l, b, w, h = block
    cb_h = 0.018
    lon_lab = 0.028
                                      #  (0.028 at fh 6.37: ~14 px clear below the labels)
    gap2 = 0.020
    map_w = (w - gap2) / 2.0
    fw, fh = fig_size
    map_h = MAP_ASPECT * map_w * fw / fh          # exact aspect — no letterbox
    map_h = min(map_h, h - cb_h - lon_lab)        # (guard: never overflow the block)
    map_bot = (b + h) - map_h                     # pinned to the block top edge
    c1 = [l,                map_bot, map_w, map_h]
    c2 = [l + map_w + gap2, map_bot, map_w, map_h]
    cb = [l + 0.005, map_bot - lon_lab - cb_h, w - 0.010, cb_h]
    return c1, c2, cb


# ===========================================================================

# ===========================================================================
def draw_a(fig, ax_a1, ax_a2, df):
    """(a) twin stacked: rainfall AMOUNT ΣV_e (top) + EVENT COUNT (bottom)."""
    yearly = df.groupby("year")["PER_flag"].agg(["size", "sum"])
    years = np.arange(int(df.year.min()), int(df.year.max()) + 1)
    n_all = yearly["size"].reindex(years, fill_value=0).values
    n_per = yearly["sum"].reindex(years, fill_value=0).values
    n_per_tot = int(n_per.sum()); n_nper_tot = int(n_all.sum()) - n_per_tot
    df["V"] = pd.to_numeric(df["volume_Ve_km3"], errors="coerce").fillna(0.0)
    v_all = (df.groupby("year")["V"].sum().reindex(years, fill_value=0.0).values)
    v_per = (df[df.PER_flag == 1].groupby("year")["V"].sum()
             .reindex(years, fill_value=0.0).values)
    v_nper = v_all - v_per
    share_v = 100 * float(v_per.sum()) / max(float(v_all.sum()), 1e-9)
    share_n = 100 * n_per_tot / len(df)

    # top sub-panel: rainfall AMOUNT (ΣV_e km³/yr), stacked PEP/non-PEP + legend

    #  non-PEP stack reads against white while orange stays dominant)
    ax_a1.bar(years, v_nper, color=C_NORMAL, width=0.8, alpha=0.72, label="non-PEP")
    ax_a1.bar(years, v_per, bottom=v_nper, color=C_PER, width=0.8, label="PEP")
    ax_a1.set_ylabel(r"$\Sigma V_e$ (km$^3$/yr)", fontsize=FS_AXIS)
    ax_a1.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax_a1.set_ylim(0, v_all.max() * 1.22)
    ax_a1.tick_params(labelbottom=False)
    ax_a1.text(0.98, 0.95, f"PEP = {share_v:.1f}% of TC rainfall volume",
               transform=ax_a1.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
               color=C_PER)
    ax_a1.legend(loc="upper left", fontsize=FS_LEGEND, handlelength=1.1, framealpha=0.9)
    panel_title(ax_a1, "a", "Yearly rainfall amount and event count")

    # bottom sub-panel: EVENT COUNT (/yr), stacked
    ax_a2.bar(years, n_all - n_per, color=C_NORMAL, width=0.8, alpha=0.72)
    ax_a2.bar(years, n_per, bottom=n_all - n_per, color=C_PER, width=0.8)
    ax_a2.set_xlabel("Year"); ax_a2.set_ylabel("Events / yr", fontsize=FS_AXIS)
    ax_a2.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax_a2.set_ylim(0, n_all.max() * 1.22)
    ax_a2.text(0.98, 0.95,
               f"non-PEP {n_nper_tot} ({100-share_n:.1f}%)\n"
               f"PEP {n_per_tot} ({share_n:.1f}%)",
               transform=ax_a2.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
               color="#333")


def draw_b(ax_b, df):
    """(b) Duration–Accumulation scatter, RT-coloured; PEP = joint upper tail."""
    per = df["PER_flag"] == 1
    d_thr = float(df["duration_days"].quantile(.90))
    a_thr = float(df["accumulation_Ae_mm"].quantile(.90))
    rt = df["RT_e_hours"].clip(upper=240)

    # orange rings stay at 1 pt so they never fuse into a solid blob
    sc = ax_b.scatter(df.loc[~per, "duration_days"], df.loc[~per, "accumulation_Ae_mm"],
                      c=rt[~per], cmap="cividis", s=10, alpha=0.65, vmin=0, vmax=200,
                      edgecolor="none")
    ax_b.scatter(df.loc[per, "duration_days"], df.loc[per, "accumulation_Ae_mm"],
                 facecolor="none", edgecolor=C_PER, s=40, lw=1.0, zorder=5,
                 label=f"PEP ({per.sum()})")
    ax_b.axvline(d_thr, color="0.25", ls="--", lw=0.7)
    ax_b.axhline(a_thr, color="0.25", ls=":", lw=0.7)
    # plan Fig.1b: threshold labels offset clear of the dashed lines at 8 pt
    ax_b.text(d_thr + 0.15, df["accumulation_Ae_mm"].max() * 0.84,
              f"P90(D)={d_thr:.0f}d", fontsize=FS_ANNOT, color="0.25", va="top")
    ax_b.text(df["duration_days"].max(), a_thr + 8, f"P90(A)={a_thr:.0f}mm",
              fontsize=FS_ANNOT, color="0.25", ha="right", va="bottom")
    ax_b.set_xlabel("Duration $D_e$ (days)")
    ax_b.set_ylabel(r"$A_e=P_{95}(\Sigma P)$ (mm)")
    ax_b.axvspan(d_thr, df["duration_days"].max() + 2, color=C_PER, alpha=0.05, zorder=0)
    cb = ax_b.figure.colorbar(sc, ax=ax_b, pad=0.03, fraction=0.046)
    cb.set_ticks([0, 50, 100, 150, 200])
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)
    # label on the RIGHT of the bar, rotated 90° (mpl default for vertical

    # after two position rounds). Room is ample: the bar ends at x≈0.910, the
    # label strip lands ≈0.913–0.929, and nothing else sits on the canvas
    # right of b (b's title ink ends at x≈0.892, the d image stays below y 0.387).
    cb.set_label("$RT_e$ (h)", fontsize=FS_CBAR_LAB)
    ax_b.legend(loc="lower right", fontsize=FS_LEGEND,
                frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    panel_title(ax_b, "b", "Duration–rainfall accumulation phase space", x=-0.180)


def _rain_map(ax, ev_row, vmax, extent, is_per, name, left_lat=True, bot_lon=True):
    """Draw one case map (storm-total rain + track + stats box) onto a GeoAxes.

    `name` (storm + year + PEP tag) is folded into the stats box as its first line so
    the panel carries a single left-aligned title while each storm stays labelled.
    """
    field, lat, lon = storm_total(int(ev_row.tc_id),
                                  str(int(ev_row.start_date)), str(int(ev_row.end_date)))
    g = load_track(int(ev_row.tc_id))
    em = event_mask(g, ev_row.start_date, ev_row.end_date)
    track_color = C_PER if is_per else C_NORMAL

    # provinces one step lighter (add_china defaults)
    add_china(ax, facecolor="#EDF1F5")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, np.ma.masked_where(field < 1, field),
                  transform=ccrs.PlateCarree(), cmap=RAIN_CMAP,
                  vmin=0, vmax=vmax, zorder=4, shading="auto", rasterized=True)

    ax.plot(g["lon"], g["lat"], color="white", lw=1.6, alpha=0.5, zorder=5,
            transform=ccrs.PlateCarree(), solid_capstyle="round")
    ax.plot(g["lon"], g["lat"], color="#555", lw=0.7, alpha=0.5, zorder=5.2,
            transform=ccrs.PlateCarree())
    # event-window segment highlighted
    ax.plot(g.loc[em, "lon"], g.loc[em, "lat"],
            color=track_color, lw=2.0, alpha=0.5, zorder=6,
            transform=ccrs.PlateCarree(), solid_capstyle="round")

    em_pts = g.loc[em].iloc[::4]
    ax.scatter(em_pts["lon"], em_pts["lat"], s=15, color=track_color,
               edgecolor="white", linewidth=0.4, alpha=0.5, zorder=6.6,
               transform=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False; gl.right_labels = False
    gl.bottom_labels = bot_lon
    gl.left_labels = left_lat
    gl.xlocator = mticker.FixedLocator(LON_TICKS)
    gl.ylocator = mticker.FixedLocator(LAT_TICKS)
    gl.xlabel_styles = {"fontsize": FS_GRID}; gl.ylabel_styles = {"fontsize": FS_GRID}


    stat = (f"{name}\nRT {ev_row.RT_e_hours:.0f} h\n{ev_row.duration_days:.0f}-day rain\n"
            f"A$_e$={ev_row.accumulation_Ae_mm:.0f} mm\nvt$_{{min}}$ {ev_row.vt_min_kmh:.0f} km/h")
    ax.text(0.03, 0.97, stat, transform=ax.transAxes, fontsize=FS_ANNOT, va="top", ha="left",
            zorder=12, linespacing=1.25)


def _case_rows(df):
    rows = []
    for c in CASES:
        m = df[(df.tc_name.str.contains(c["name"], case=False, na=False)) &
               (df.year == c["year"])]
        rows.append(m.iloc[0])
    return rows


def draw_c(fig, ax_c1, ax_c2, cax, df):
    """(c) two case maps (non-PEP | PEP) on a shared China-TC frame + shared colorbar."""
    rows = _case_rows(df)
    both = []
    for r in rows:
        f, _, _ = storm_total(int(r.tc_id), str(int(r.start_date)), str(int(r.end_date)))
        if f is not None:
            both.append(f[f > 1])
    both = np.concatenate(both) if both else np.array([1.0])
    vmax = float(np.percentile(both, 98))
    extent = CASE_DOMAIN

    name0 = f"{rows[0].tc_name} {int(rows[0].year)} · non-PEP"
    name1 = f"{rows[1].tc_name} {int(rows[1].year)} · PEP"
    _rain_map(ax_c1, rows[0], vmax, extent, is_per=False, name=name0)
    _rain_map(ax_c2, rows[1], vmax, extent, is_per=True, name=name1, left_lat=False)
    # explicit y disables mpl's auto title positioning, which gridliner labels
    # on GeoAxes leave non-finite (title silently never drawn with cartopy

    # at their exact aspect (no letterbox), so y=1.0 + pad 6 now equals the
    # auto-positioned d title baseline exactly (the old y=1.055 was the fudge

    # a fat 101-px air gap under the title). x=-0.333: c1 is only HALF


    panel_title(ax_c1, "c", "Case studies: non-PEP vs. PEP", y=1.0, x=-0.333)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=RAIN_CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label("Storm-total rainfall (mm)", fontsize=FS_CBAR_LAB)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)


def draw_d(ax_d):
    """(d) research-gap diagram body (title stripped; rendered as mpl text for
    consistency with a/b/c). The diagram is drawn on a larger axes pinned to
    the block's TOP-LEFT: enlarged left into the column gutter, right into the
    empty canvas margin and downward past the block bottom for a more prominent
    again, then shrink 5% twice). The box is built at the image's NATIVE aspect → zero distortion/
    letterbox; the top edge stays at the block top so the title (on ax_d,
    pad 6) keeps its position."""
    body_png = Path(__file__).resolve().parents[1] / "fig1-d-body.png"
    img = plt.imread(str(body_png))
    aspect = img.shape[1] / img.shape[0]
    l, b, w, h = ax_d.get_position().bounds
    fw, fh = ax_d.figure.get_size_inches()
    overflow_l = 0.058
    overflow_r = 0.069

    # on-canvas guards (master values are in-bounds; the narrower --panels
    # canvas clamps here instead of clipping)
    overflow_l = min(overflow_l, l - 0.005)
    overflow_r = min(overflow_r, 0.995 - (l + w))
    span_full = w + overflow_l + overflow_r   # full-bleed span (previous round: 0.5375)
    span = 0.9025 * span_full
                                              #  diagram 5% TWICE (0.95²); box re-centred on
                                              #  the full-bleed box, top edge still pinned
    cx = (l - overflow_l) + span_full / 2.0
    left = max(cx - span / 2.0, 0.005)
    right = min(left + span, 0.995)
    box_w = right - left
    box_h = box_w * fw / aspect / fh       # exact native aspect
    top = b + h
    max_h = top - 0.005                    # keep the image on-canvas (height guard)
    if box_h > max_h:
        box_h = max_h
        box_w = box_h * fh * aspect / fw
    ax_img = ax_d.figure.add_axes([left, top - box_h, box_w, box_h])
    ax_img.imshow(img, aspect="equal")
    ax_img.set_axis_off()
    ax_d.set_axis_off()
    # x=-0.1663 (shallower than b's -0.180): b's title anchor sits on the


    # the (d) title ink on (b)'s anchor so the two right-column titles align

    # the d title 8.25 px left of b's at 150 dpi under the old -0.1857).
    panel_title(ax_d, "d", "From TC motion to rainfall-event organization", pad=6, x=-0.1663)


# ===========================================================================
#  master composer — 2×2 equal blocks
# ===========================================================================
def compose(df):
    fig = plt.figure(figsize=G_FIG)
    proj = ccrs.PlateCarree()

    # (a) twin stacked
    a1r, a2r = a_subrects(A_BLOCK)
    draw_a(fig, fig.add_axes(a1r), fig.add_axes(a2r), df)

    # (b) D-A scatter
    draw_b(fig.add_axes(B_RECT), df)

    # (c) two case maps + colorbar.


    # behind the colorbar's white background at the tight lon_lab gap).
    c1r, c2r, cbr = c_subrects(C_BLOCK)
    cax = fig.add_axes(cbr)
    ax_c1 = fig.add_axes(c1r, projection=proj)
    ax_c2 = fig.add_axes(c2r, projection=proj)
    draw_c(fig, ax_c1, ax_c2, cax, df)

    # (d) research-gap diagram
    draw_d(fig.add_axes(D_RECT))

    info = save(fig, "Fig1_PEP_definition", column="double", checks=True)
    plt.close(fig)
    return info


# ===========================================================================

#  output as PNG (600 dpi) for individual adjustment
# ===========================================================================
PANEL_DIR = FIG_DIR / "_panels"
_PANEL_FIG_AB = (3.6, 3.12)
_PANEL_FIG_CD = (3.6, 2.385)   # aspect 1.510 == bottom-row block aspect (c, d)


def _save_panel(fig, name, d, save_kw):
    png = d / f"{name}.png"
    fig.savefig(png, **save_kw)
    plt.close(fig)
    return {"png": str(png)}


def render_standalone(df):
    """Render each of a/b/c/d to its own PNG on a canvas matching its row's
    block aspect, so the four panels can be adjusted individually and the bottom
    row (c, d) tile at the shorter 1.51 aspect."""
    PANEL_DIR.mkdir(exist_ok=True)
    proj = ccrs.PlateCarree()
    SAVE = dict(dpi=600, facecolor="white")
    out = {}

    with mpl.rc_context({"savefig.bbox": "standard"}):   # full canvas, no tight-crop
        # (a) — twin stacked
        fig = plt.figure(figsize=_PANEL_FIG_AB)
        a1r, a2r = a_subrects([0.16, 0.12, 0.80, 0.78])
        draw_a(fig, fig.add_axes(a1r), fig.add_axes(a2r), df)
        out["a"] = _save_panel(fig, "fig1_a", PANEL_DIR, SAVE)

        # (b) — D-A scatter
        fig = plt.figure(figsize=_PANEL_FIG_AB)
        draw_b(fig.add_axes([0.16, 0.14, 0.78, 0.74]), df)
        out["b"] = _save_panel(fig, "fig1_b", PANEL_DIR, SAVE)


        fig = plt.figure(figsize=_PANEL_FIG_CD)
        c1r, c2r, cbr = c_subrects([0.10, 0.06, 0.84, 0.86], fig_size=_PANEL_FIG_CD)
        cax = fig.add_axes(cbr)
        ax_c1 = fig.add_axes(c1r, projection=proj)
        ax_c2 = fig.add_axes(c2r, projection=proj)
        draw_c(fig, ax_c1, ax_c2, cax, df)
        out["c"] = _save_panel(fig, "fig1_c", PANEL_DIR, SAVE)

        # (d) — research-gap diagram
        fig = plt.figure(figsize=_PANEL_FIG_CD)
        draw_d(fig.add_axes([0.04, 0.06, 0.92, 0.86]))
        out["d"] = _save_panel(fig, "fig1_d", PANEL_DIR, SAVE)

    return out


def main():
    df = load_events()
    info = compose(df)
    paths = render_standalone(df)                      # always emit single-panel PNG
    print(f"[panels] standalone panels -> {paths}")
    return info


if __name__ == "__main__":
    main()
