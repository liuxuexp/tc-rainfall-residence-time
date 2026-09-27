#!/usr/bin/env python3
"""Fig. 5 - trends and dynamic-thermodynamic attribution (a-f).

Reads data/tc_events_full.csv, yearly_core_tcwv_ivt.csv, trend_post1979.csv
and per_event_environment.csv. Output: figures/Fig5_trends_attribution.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import theilslopes, norm

from _style import (save, label_panel, halo_text, OKABE_ITO, C_PER,
                    FS_ANNOT, FS_TITLE, FS_LEGEND)
from config import DATA_DIR

# plan Fig.5e: the three decomposition colours deepened one step (same hue
# family, ~×0.82) for line contrast on white; shared by (e) lines and (f) bars
DEEP = dict(dyn="#005E93", the="#B04D00", inter="#008160")


def window_tag(ax, text):
    """Small grey time-window tag, top-right of the axes (the figure mixes 1960–2024
    and post-1979 windows; keeping that visible per-panel matters)."""
    halo_text(ax, 0.985, 0.97, text, fs=FS_ANNOT, color="0.4", ha="right", va="top")


def mk_p(x):
    """Mann–Kendall two-sided p (tie-corrected); ported from pipeline/10b."""
    x = np.asarray(x, float); n = len(x); s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))
    uq, c = np.unique(x, return_counts=True); ties = c[c > 1]
    var = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    z = (s - 1) / np.sqrt(var) if s > 0 else ((s + 1) / np.sqrt(var) if s < 0 else 0.0)
    return 2 * (1 - norm.cdf(abs(z)))


def sig_star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def ts_dec(values, years):
    """Theil–Sen slope per decade + MK p over finite pairs."""
    v = np.asarray(values, float); yr = np.asarray(years, float)
    m = np.isfinite(v)
    v, yr = v[m], yr[m]
    if len(v) < 5:
        return np.nan, np.nan
    return float(theilslopes(v, yr).slope) * 10, float(mk_p(v))


def main():
    ef = pd.read_csv(DATA_DIR / "tc_events_full.csv")
    y = pd.read_csv(DATA_DIR / "yearly_core_tcwv_ivt.csv")
    tr = pd.read_csv(DATA_DIR / "trend_post1979.csv").set_index("series")
    env = pd.read_csv(DATA_DIR / "per_event_environment.csv")

    yrs = sorted(ef["year"].unique())
    per_cnt = np.array([((ef["year"] == yy) & (ef["PER_flag"] == 1)).sum() for yy in yrs], float)

    # (a) carries twin-axis ticks + a rotated label on its right, so its full


    # copy of the cell via fixed axes-locators. make_axes_locatable was tried
    # and abandoned here: twinx() re-adds an axes on (a)'s subplotspec, which
    # silently cancels the divider's shrink ((a) then renders full width while
    # its twin furniture drifts right). A_FRAC = 88% widens (a) again (user
    # request) while still freeing a strip for the twin ticks + tick labels +
    # rotated ylabel (~0.33 in), which keeps ~0.2 in clear before (b)'s ylabel.
    A_FRAC = 0.88

    _cell = None                              # (a)'s full cell rect, captured once

    def _narrow_a(ax_, renderer):
        return Bbox.from_bounds(_cell.x0, _cell.y0, _cell.width * A_FRAC, _cell.height)

    fig = plt.figure(figsize=(7.2, 8.5))
    gs = fig.add_gridspec(3, 2)

    # Ink overhangs the axes box by left 0.448 (tick+ylabels) / top 0.178 (titles
    # + letters) / bottom 0.312 (xticks+xlabel) / right ~0 in, so the fractions


    # 183 mm), leaving ~2 mm air in each gutter after the label stacks.
    gs.update(left=0.08, right=0.983, top=0.965, bottom=0.051,
              wspace=0.17, hspace=0.26)
    ax = gs.subplots()
    _cell = ax[0, 0].get_position()
    ax[0, 0].set_axes_locator(_narrow_a)
    panels = ax.ravel()
    TREND = "#1a1a1a"

    # ---- (a) PEP annual counts (bars, left axis) + cumulative + mean-rate (twin axis) ----
    # Adapted from figlib 057 (cyclone counts + cumulative, twin-axis): the raw sparse
    # yearly counts (35/64 yr are zero) are shown honestly as bars, with the cumulative
    # trajectory + mean-rate reference on a colour-coded twin axis. Cumulative is the


    # (MK p=0.99, n.s.). Bars replace the earlier event "comb" (a bar already marks each
    # event year). Axes are colour-coded to their series.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    a = panels[0]

    # colour-coded twin axes still read in print; line types distinguish the
    # cumulative (solid) from the mean-rate reference (dashed)
    a.bar(yrs, per_cnt, color=C_PER, alpha=0.72, width=1.0, zorder=2)
    a.set_ylabel("PEP events yr$^{-1}$", color=C_PER)
    a.tick_params(axis="y", colors=C_PER)
    a.set_ylim(0, max(per_cnt.max() * 1.3, 4))
    a2 = a.twinx()
    a2.set_axes_locator(_narrow_a)          # twin follows the narrowed (a)
    a2.spines["right"].set_visible(True)
    cum = np.cumsum(per_cnt)
    rate = per_cnt.sum() / len(yrs)                                   # ~0.59 events/yr
    a2.plot(yrs, cum, "-", color="0.15", lw=1.6, zorder=3)
    a2.plot(yrs, rate * (np.array(yrs) - yrs[0]), "--", color="0.5", lw=1.0, zorder=3)
    a2.set_ylabel("Cumulative PEP events", color="0.15", labelpad=2)
    a2.tick_params(axis="y", colors="0.15")
    a2.set_ylim(0, cum.max() * 1.08)
    pa = mk_p(per_cnt)
    a.legend([Patch(facecolor=C_PER, alpha=0.72),
              Line2D([0], [0], color="0.15", lw=1.6),
              Line2D([0], [0], color="0.5", lw=1.0, ls="--")],
             ["PEP events yr$^{-1}$", "cumulative",
              f"mean rate {rate:.2f} yr$^{{-1}}$"],
             loc="upper left", fontsize=FS_LEGEND, frameon=False, labelspacing=0.3)
    halo_text(a, 0.97, 0.05, f"n={int(per_cnt.sum())} events\nMK p={pa:.2f} (n.s.)",
              ha="right", va="bottom")
    a.set_xlabel("Year", labelpad=2)
    label_panel(a, "a"); a.set_title("PEP occurrence", fontsize=FS_TITLE, pad=2)

    # ---- (b) annual-mean PEP A_e + Theil–Sen trend (PEP years only) ----
    a = panels[1]
    py = sorted(ef.loc[ef["PER_flag"] == 1, "year"].unique())
    ae = np.array([ef.loc[(ef["year"] == yy) & (ef["PER_flag"] == 1),
                          "accumulation_Ae_mm"].mean() for yy in py], float)
    # plan Fig.5b: data line ~1.2 pt, markers kept small
    a.plot(py, ae, "-o", color=OKABE_ITO[1], ms=2.5, lw=1.2)
    ts_b = theilslopes(ae, py)
    sb, pb = ts_b.slope * 10, mk_p(ae)
    a.plot(py, ts_b.intercept + ts_b.slope * np.array(py),
           "--", color=TREND, lw=1.0, alpha=0.8)
    halo_text(a, 0.03, 0.97, f"Theil–Sen {sb:+.1f} mm/dec {sig_star(pb)}\n(n={len(py)} PEP-yrs)",
              ha="left", va="top")
    a.set_xlabel("Year", labelpad=2); a.set_ylabel("PEP mean $A_e$ (mm)", labelpad=2)
    label_panel(a, "b"); a.set_title("PEP event accumulation", fontsize=FS_TITLE, pad=2)

    # ---- (c) mean RT_e: all events vs PEP events ----
    a = panels[2]
    rt_all = np.array([ef.loc[ef["year"] == yy, "RT_e_hours"].mean() for yy in yrs], float)
    rt_per = np.array([ef.loc[(ef["year"] == yy) & (ef["PER_flag"] == 1),
                              "RT_e_hours"].mean() if ((ef["year"] == yy) & (ef["PER_flag"] == 1)).any()
                       else np.nan for yy in yrs], float)
    a.plot(yrs, rt_all, "-", color=OKABE_ITO[0], lw=1.2, label="all events")
    pm = ~np.isnan(rt_per)
    # plan Fig.5c: PEP squares a touch smaller so near years don't fuse
    a.plot(np.array(yrs)[pm], rt_per[pm], "s", color=C_PER, ms=2.6, label="PEP events")
    sa, pa2 = ts_dec(rt_all, yrs)
    sp, pp2 = ts_dec(rt_per, yrs)
    halo_text(a, 0.97, 0.75, f"all {sa:+.1f}/dec {sig_star(pa2)}\nPEP {sp:+.1f}/dec {sig_star(pp2)}",
              ha="right", va="top")
    a.set_xlabel("Year", labelpad=2); a.set_ylabel("mean $RT_e$ (h)"); a.legend(loc="upper left", fontsize=FS_LEGEND)
    # widen the y-range: headroom (0.20) so the upper-left legend & top-right annotation
    # clear the data, plus footroom (0.12); mirrors the (e) ylim expansion.
    _ylo, _yhi = a.get_ylim()
    a.set_ylim(_ylo - 0.12 * (_yhi - _ylo), _yhi + 0.20 * (_yhi - _ylo))
    label_panel(a, "c"); a.set_title("Residence time", fontsize=FS_TITLE, pad=2)

    # ---- (d) PEP-event environment TCWV & |IVT| (standardised, post-1979) ----
    a = panels[3]
    e79 = env[env["year"] >= 1979]
    ey = sorted(e79["year"].unique())
    tcwv = np.array([e79.loc[e79["year"] == yy, "TCWV_env"].mean() for yy in ey], float)
    ivt = np.array([e79.loc[e79["year"] == yy, "IVT_env"].mean() for yy in ey], float)
    tz = (tcwv - tcwv.mean()) / tcwv.std(ddof=0)
    iz = (ivt - ivt.mean()) / ivt.std(ddof=0)
    dt, pt = ts_dec(tz, ey); di, pi = ts_dec(iz, ey)
    a.plot(ey, tz, "o", color=OKABE_ITO[5], ms=3.5, label="TCWV")
    a.plot(ey, iz, "s", color=OKABE_ITO[1], ms=3.5, label="|IVT|")
    # plan Fig.5d: trend lines deeper (amber one step darker than the yellow
    # markers so they read on white; blue likewise), zero line thin grey
    ts_t = theilslopes(tz, ey)
    a.plot(ey, ts_t.intercept + ts_t.slope * np.array(ey), "-",
           color=OKABE_ITO[5], lw=1.2, alpha=0.75)
    ts_i = theilslopes(iz, ey)
    a.plot(ey, ts_i.intercept + ts_i.slope * np.array(ey), "-",
           color="#B47C00", lw=1.2, alpha=0.9)
    a.axhline(0, color="0.6", lw=0.5)

    # (|z| <= 2.6) and both trend lines sit well inside
    a.set_ylim(-3, 3)
    halo_text(a, 0.97, 0.05, f"TCWV {dt:+.2f}/dec {sig_star(pt)}\nIVT {di:+.2f}/dec {sig_star(pi)}\n"
                             f"n={len(ey)} yrs/{len(e79)} events",
              ha="right", va="bottom")
    a.set_xlabel("Year", labelpad=2); a.set_ylabel("standardised anomaly")
    a.legend(loc="upper left", fontsize=FS_LEGEND)
    label_panel(a, "d"); a.set_title("PEP-event moisture env.", fontsize=FS_TITLE, pad=2)
    window_tag(a, "1979–2024")

    # ---- (e) dyn / the / interaction standardised time series (post-1979) ----
    # Slopes + MK p are recomputed in-panel from the raw series and shared with (f), so the
    # number annotated on each line here is the same one (f) plots as a bar (+ 95% CI).
    a = panels[4]
    y79 = y[y["year"] >= 1979].copy()
    yrs79 = y79["year"].values

    def z(s):
        v = y79[s].values
        return (v - v.mean()) / v.std(ddof=0)

    decomp = {}
    for key, lab, col, ls in [
            ("dyn", r"dynamic $\bar q\,\Delta\vec V$", DEEP["dyn"], "-"),
            ("the", r"thermo $\Delta q\,\bar{\vec V}$", DEEP["the"], "-"),
            ("interaction", "interaction", DEEP["inter"], (0, (4, 2)))]:
        v = y79[key].values
        ts = theilslopes(v, yrs79)
        decomp[key] = dict(label=lab, color=col, slope=ts.slope * 10,
                           lo=ts.low_slope * 10, hi=ts.high_slope * 10, p=mk_p(v))
        # plan Fig.5e: equal 1.2 pt lines; interaction dashed to separate it
        # from the two main-pathway curves
        a.plot(yrs79, z(key), linestyle=ls, color=col, lw=1.2, label=lab)
    # regression guard: in-panel slopes must reproduce data/trend_post1979.csv
    for key, tk in [("dyn", "dyn_1979"), ("the", "the_1979"), ("interaction", "interaction_1979")]:
        assert abs(decomp[key]["slope"] - float(tr.loc[tk, "slope"])) < 1e-6, \
            f"{key}: in-panel {decomp[key]['slope']} != trend_post1979 {tr.loc[tk, 'slope']}"
    a.axhline(0, color="0.6", lw=0.5)
    _ylo, _yhi = a.get_ylim(); a.set_ylim(_ylo - 0.38 * (_yhi - _ylo), _yhi + 0.25 * (_yhi - _ylo))
    halo_text(a, 0.97, 0.05,
              "\n".join(f"{k}: {decomp[k]['slope']:+.2f}{sig_star(decomp[k]['p'])}/dec"
                        for k in ["dyn", "the", "interaction"]),
              ha="right", va="bottom")
    a.set_xlabel("Year", labelpad=2); a.set_ylabel("standardised anomaly")
    a.legend(loc="upper left", fontsize=FS_LEGEND)
    label_panel(a, "e"); a.set_title("Decomposition terms", fontsize=FS_TITLE, pad=2)
    window_tag(a, "1979–2024")

    # ---- (f) decomposition Theil–Sen slopes + 95% CI + significance (post-1979) ----
    a = panels[5]
    keys = ["dyn", "the", "interaction"]
    vals = [decomp[k]["slope"] for k in keys]
    err = [[decomp[k]["slope"] - decomp[k]["lo"] for k in keys],
           [decomp[k]["hi"] - decomp[k]["slope"] for k in keys]]
    xpos = np.arange(3)
    # plan Fig.5f: uniform caps, stars at a fixed offset above the whisker top;
    # fixed symmetric ylim keeps the dynamic term's long upper CI fully inside the frame
    a.bar(xpos, vals, color=[decomp[k]["color"] for k in keys], width=0.62,
          yerr=err, capsize=2.5, error_kw=dict(lw=0.9, color="0.25"))
    a.set_ylim(-3.1, 3.1)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(xpos); a.set_xticklabels(["dynamic", "thermo.", "inter."])
    a.set_ylabel("Theil–Sen slope (decade$^{-1}$)")
    for i, k in enumerate(keys):
        a.text(i, decomp[k]["hi"] + 0.03, sig_star(decomp[k]["p"]),
               ha="center", va="bottom", fontsize=8)
    label_panel(a, "f"); a.set_title("Attribution", fontsize=FS_TITLE, pad=2)
    window_tag(a, "1979–2024")

    info = save(fig, "Fig5_trends_attribution", column="double", checks=True)
    plt.close(fig)
    return info


if __name__ == "__main__":
    main()
