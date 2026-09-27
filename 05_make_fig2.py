#!/usr/bin/env python3
"""Fig. 2 - controls on event residence time (panels a-f).

Reads data/tc_events_full.csv. Output: figures/Fig2_RT_controls.png (183 mm).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from _style import (save, label_panel, C_PER, C_NPER, C_NORMAL, OKABE_ITO,
                    halo_text, FS_ANNOT, FS_LEGEND, FS_AXIS, FS_TICK, FS_PANEL)
from config import DATA_DIR


# x1.2, then x1.4 with the subplot grid re-tuned to keep the larger labels
# clear of neighbouring panels). Fig2 only: the house FS_* ladder in _style
# and every other figure keep their sizes. (label_panel/halo_text defaults
# bind in _style's scope, so their call sites below pass fs= explicitly.)
TXT = 1.4
FS_ANNOT, FS_LEGEND = FS_ANNOT * TXT, FS_LEGEND * TXT
FS_PANEL, FS_AXIS, FS_TICK = FS_PANEL * TXT, FS_AXIS * TXT, FS_TICK * TXT
mpl.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "axes.labelsize": FS_AXIS, "legend.fontsize": FS_LEGEND})


def load():
    df = pd.read_csv(DATA_DIR / "tc_events_full.csv")
    df = df.dropna(subset=["RT_e_hours", "duration_days",
                           "accumulation_Ae_mm", "peak_daily_mm"])
    return df


def knn_prob_ci(rt, pf, grid, k=120, nboot=500, seed=0):
    """k-nearest-neighbour empirical P(PEP|RT) + seeded bootstrap 95% CI on a grid.

    At each grid RT, take the k events nearest in RT; P = mean(PEP flag);
    CI = bootstrap resampling those k events. Non-parametric smooth estimate
    (replaces the jumpy equal-count decile step)."""
    rng = np.random.default_rng(seed)
    rt = np.asarray(rt, float); pf = np.asarray(pf, float)
    xs, pm, lo, hi = [], [], [], []
    for g in grid:
        idx = np.argsort(np.abs(rt - g))[:k]
        pp = pf[idx]
        draws = rng.choice(pp, size=(nboot, k), replace=True).mean(axis=1)
        xs.append(g); pm.append(pp.mean())
        lo.append(np.percentile(draws, 2.5)); hi.append(np.percentile(draws, 97.5))
    return np.array(xs), np.array(pm), np.array(lo), np.array(hi)


def ols_ci(x, y, grid, alpha=0.05):
    """OLS linear fit + 95% CI band of the mean response on `grid`."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return None
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, grid)
    resid = y - np.polyval(p, x)
    s2 = np.sum(resid ** 2) / (x.size - 2)
    xm = x.mean(); sxx = np.sum((x - xm) ** 2)
    if sxx == 0:
        return None
    se = np.sqrt(s2 * (1.0 / x.size + (grid - xm) ** 2 / sxx))
    z = 1.96
    return yhat, yhat - z * se, yhat + z * se


def mean_ci(v, nboot=500, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, np.nan, np.nan
    boots = np.array([rng.choice(v, size=v.size, replace=True).mean()
                      for _ in range(nboot)])
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main():
    df = load()
    rt = df["RT_e_hours"].values
    D = df["duration_days"].values
    A = df["accumulation_Ae_mm"].values
    Pk = df["peak_daily_mm"].values
    pf = df["PER_flag"].values.astype(int)
    rt_l = df["RT_local_hours"].values

    fig, ax = plt.subplots(3, 2, figsize=(7.2, 8.4))
    # x1.4 text retune: wider column gap (right-column ytick+ylabel), taller
    # row gap (xlabel below vs bold panel letter above), roomier left margin.
    fig.subplots_adjust(left=0.115, right=0.975, top=0.95, bottom=0.085,
                        wspace=0.30, hspace=0.36)

    TREND = "#1a1a1a"

    # ---- (a) RT distribution: violin + inner box + median (005 / 070) ----
    a0 = ax[0, 0]
    _data = [rt[pf == 0], rt[pf == 1]]
    _pos = [1, 2]
    _cols = [C_NPER, C_PER]
    _labs = [f"non-PEP\n({int((pf == 0).sum())})", f"PEP\n({int((pf == 1).sum())})"]
    vp = a0.violinplot(_data, positions=_pos, widths=0.75,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, col in zip(vp["bodies"], _cols):
        # plan Fig.2a: less saturated fill, crisper outline
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(0.30); body.set_linewidth(0.9)
    bp = a0.boxplot(_data, positions=_pos, widths=0.13, patch_artist=True,
                    showfliers=False, zorder=3)
    for patch, med, col in zip(bp["boxes"], bp["medians"], _cols):
        patch.set_facecolor("white"); patch.set_edgecolor(col); patch.set_alpha(0.95)
        med.set_color(col); med.set_linewidth(1.8)
    for w, cap, col in zip(bp["whiskers"], bp["caps"],
                           [c for c in _cols for _ in (0, 1)]):
        w.set_color(col); cap.set_color(col)
    med_n = float(np.median(_data[0])); med_p = float(np.median(_data[1]))
    _trans = a0.get_xaxis_transform()
    for p, mv, col in [(1, med_n, C_NPER), (2, med_p, C_PER)]:
        halo_text(a0, p, 0.97, f"median {mv:.0f} h", fs=FS_ANNOT, transform=_trans,
                  ha="center", va="top", color=col)
    a0.set_xticks(_pos); a0.set_xticklabels(_labs)
    a0.set_xlim(0.4, 2.9); a0.set_ylabel("$RT_e$ (h)")
    label_panel(a0, "a", fs=FS_PANEL)

    # ---- (b) P(PEP|RT): k-NN sliding-window empirical proportion + bootstrap 95% CI ----
    #   Gentle (smooth) version: at each grid RT, P = mean PEP flag over the k=120 nearest

    #   Replaces the jumpy decile step (D1-9≈0%, D10≈30%); reveals onset ~100 h, plateau ~31%.
    bgrid = np.arange(0, 301, 10)
    xs, pm, lo, hi = knn_prob_ci(rt, pf, bgrid, k=120)
    a1 = ax[0, 1]
    base = pf.mean() * 100
    pm_pct = pm * 100
    # base-rate reference: dashed line + halo label (mirrors a's median tag);
    # plan Fig.2b: label nudged clear of the dashed line, value untouched
    a1.axhline(base, color="0.55", ls="--", lw=0.8)
    halo_text(a1, 292, base + 1.0, f"  base rate {base:.1f}%", fs=FS_ANNOT, transform=a1.transData,
              ha="right", va="bottom", color="0.35")
    # 95% bootstrap CI ribbon + smooth proportion curve (plan: band ~0.15 alpha)
    a1.fill_between(xs, lo * 100, hi * 100, color=C_PER, alpha=0.15, zorder=2)
    a1.plot(xs, pm_pct, "-", color=C_PER, lw=1.5, zorder=4)
    # saturation: hollow ring + halo label where curve first reaches 90% of its max
    isat = int(np.argmax(pm >= 0.9 * pm.max()))
    a1.plot(xs[isat], pm_pct[isat], "o", mfc="white", mec=C_PER, mew=1.4, ms=5.0, zorder=6)
    halo_text(a1, 0.5, 0.97, f"plateau ≈ {pm_pct[isat]:.0f}% @ $RT_e$ ≈ {xs[isat]:.0f} h", fs=FS_ANNOT,
              ha="center", va="top", color=C_PER)
    a1.set_xlim(0, 300)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a1.set_xlabel("$RT_e$ (h)"); a1.set_ylabel("P(PEP | RT)")
    label_panel(a1, "b", fs=FS_PANEL)

    # ---- (c)(d)(e) RT_local vs D / A / peak: scatter + OLS + 95% CI band (020 / 042) ----
    #   Event-centric RT_local, not RT_e: r(RT_e,D)=0.98 is near-mechanical (RT_e≈24·D),

    XL = float(np.nanmax(rt_l))
    xg = np.linspace(0, XL, 200)
    specs = [(2, D, "Duration $D_e$ (days)", "c"),
             (3, A, "$A_e$ (mm)",            "d"),
             (4, Pk, "Peak daily rain (mm)", "e")]
    # plan Fig.2(c-e): sky blue one step deeper for contrast; fit line <=1.5 pt;

    C_SCAT = "#45A2DE"
    for pos, yv, ylab, lab in specs:
        a = ax.flat[pos]
        a.scatter(rt_l, yv, c=C_SCAT, s=9, alpha=0.32, edgecolor="none", rasterized=True)
        ci = ols_ci(rt_l, yv, xg)
        if ci is not None:
            yhat, loy, hiy = ci
            a.fill_between(xg, loy, hiy, color=TREND, alpha=0.12, zorder=3)
            a.plot(xg, yhat, "-", color=TREND, lw=1.5, zorder=4)
        r = corr(rt_l, yv)
        weak = "  (weak)" if abs(r) < 0.40 else ""
        a.set_xlabel("$RT_{local}$ (h)"); a.set_ylabel(ylab)
        a.set_xlim(0, XL)
        a.text(0.03, 0.95, f"r = {r:.2f}{weak}", transform=a.transAxes,
               va="top", fontsize=FS_ANNOT,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
        label_panel(a, lab, fs=FS_PANEL)

    # ---- (f) dose-response by RT_e quartile + CI + Duration Δ arrow (001 / 010) ----
    #   RT_e quartiles: σ ranges D 2.4 / A 1.6 / Pk 1.6 match text L49 dose-response.

    #    not plotted here. c/d/e use RT_local to match text r; (f) uses RT_e to match σ.)
    qlabels = ["Short", "Moderate", "Long", "Extreme"]
    q = pd.qcut(rt, 4, labels=qlabels)

    def z(x):
        return (x - x.mean()) / x.std(ddof=0)

    dz = [mean_ci(z(D)[q == lab],  seed=i)      for i, lab in enumerate(qlabels)]
    az = [mean_ci(z(A)[q == lab],  seed=10 + i) for i, lab in enumerate(qlabels)]
    pz = [mean_ci(z(Pk)[q == lab], seed=20 + i) for i, lab in enumerate(qlabels)]

    def series(triplets):
        m = np.array([t[0] for t in triplets])
        lo = np.array([t[1] for t in triplets])
        hi = np.array([t[2] for t in triplets])
        return m, (m - lo, hi - m)

    bx = np.arange(4); w = 0.24
    af = ax[2, 1]
    dm, derr = series(dz)
    am, aerr = series(az)
    pm_, perr = series(pz)
    af.bar(bx - w, dm, w, label="Duration", color=OKABE_ITO[5],
           yerr=derr, capsize=2, ecolor="0.35")
    af.bar(bx,     am, w, label="Accum.",  color=OKABE_ITO[6],
           yerr=aerr, capsize=2, ecolor="0.35")
    af.bar(bx + w, pm_, w, label="Peak",    color=OKABE_ITO[3],
           yerr=perr, capsize=2, ecolor="0.35")

    y_top = max(dm.max(), am.max(), pm_.max())
    af.text(1.5 - w, y_top - 0.15, f"Duration +{dm[3] - dm[0]:.2f}σ",
            color=OKABE_ITO[5], ha="center", fontsize=FS_ANNOT)
    af.set_xticks(bx); af.set_xticklabels(["Short", "Mod", "Long", "Extr"])
    af.set_ylabel("Standardized anomaly"); af.set_ylim(None, y_top + 0.34)
    af.axhline(0, color="k", lw=0.5)
    af.grid(axis="y", lw=0.3, alpha=0.4)
    # x1.4 pass: in-axes lower-right legend now rides over the Extr negative

    # no xlabel; the row bottom-aligns with (e)'s "RT_local (h)" label).
    af.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=FS_LEGEND,
              columnspacing=1.2, handlelength=1.2, handletextpad=0.4)
    label_panel(af, "f", fs=FS_PANEL)

    info = save(fig, "Fig2_RT_controls", column="double", checks=True)
    plt.close(fig)
    return info


if __name__ == "__main__":
    main()
