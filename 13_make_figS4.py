#!/usr/bin/env python3
"""Fig. S4 - best-track dataset sensitivity (CMA vs IBTrACS).

Reads data/si_ibtracs.csv. Output: figures/FigS4_track_sensitivity.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from _style import (natural_save, label_panel, halo_text, C_PER, C_NPER, OKABE_ITO,
                    FS_PANEL, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND)
from config import DATA_DIR

# plan Fig.S4(a,d): light-grey points deepened one step (still the grey class)
GREY_DEEP = "#707070"


def knn(rt, flag, grid, k=120, seed=0):
    rng = np.random.default_rng(seed)
    rt = np.asarray(rt, float); flag = np.asarray(flag, float)
    xs, pm, lo, hi = [], [], [], []
    for g in grid:
        idx = np.argsort(np.abs(rt - g))[:k]
        pp = flag[idx]
        draws = rng.choice(pp, size=(200, k), replace=True).mean(axis=1)
        xs.append(g); pm.append(pp.mean())
        lo.append(np.percentile(draws, 2.5)); hi.append(np.percentile(draws, 97.5))
    return np.array(xs), np.array(pm), np.array(lo), np.array(hi)


def _style_ax(ax):
    """Main-figure panel style: drop top/right spines, faint y grid, tick size."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=FS_TICK)


def main():
    df = pd.read_csv(DATA_DIR / "si_ibtracs.csv")
    m = df.dropna(subset=["RT_CMA", "RT_IB"]).copy()
    pf = m.PER_flag.values.astype(int)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.955, bottom=0.085,
                        wspace=0.15, hspace=0.17)

    # (a) RT scatter
    a = ax[0, 0]
    for f, col, lab in [(0, GREY_DEEP, f"non-PEP ({(pf == 0).sum()})"),
                        (1, C_PER, f"PEP ({(pf == 1).sum()})")]:
        s = m[pf == f]
        a.scatter(s.RT_CMA, s.RT_IB, s=12, c=col, alpha=0.65, edgecolor="none",
                  rasterized=True, label=lab)
    r = float(np.corrcoef(m.RT_CMA, m.RT_IB)[0, 1])
    mx = float(np.nanmax([m.RT_CMA.max(), m.RT_IB.max()]))
    a.plot([0, mx], [0, mx], "k--", lw=0.6, alpha=0.5)
    halo_text(a, 0.03, 0.95, f"r = {r:.2f}  (n = {len(m)})", fs=FS_ANNOT,
              color="0.1", va="top", ha="left")
    a.set_xlabel("$RT_e$ CMA (h)", fontsize=FS_AXIS)
    a.set_ylabel("$RT_e$ IBTrACS (h)", fontsize=FS_AXIS)
    a.legend(fontsize=FS_LEGEND, frameon=False, loc="lower right")
    _style_ax(a)
    label_panel(a, "a")

    # (b) RT violin by dataset
    b = ax[0, 1]
    data = [m.RT_CMA[pf == 0], m.RT_CMA[pf == 1], m.RT_IB[pf == 0], m.RT_IB[pf == 1]]
    pos = [1, 2, 4, 5]; cols = [C_NPER, C_PER, C_NPER, C_PER]
    vp = b.violinplot(data, positions=pos, widths=0.8, showmedians=False, showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        # plan Fig.S4b: one brightness level for grey/orange fills, crisp outline
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(0.30); body.set_linewidth(0.9)
    for p, dd, col in zip(pos, data, cols):
        b.hlines(float(np.median(dd)), p - 0.34, p + 0.34, colors=col, lw=1.7, zorder=4)
    b.axvline(3.0, color="0.75", lw=0.4)   # plan: divider thinner + lighter
    b.set_xticks([1.5, 4.5]); b.set_xticklabels(["CMA", "IBTrACS"], fontsize=FS_TICK)
    b.set_ylabel("$RT_e$ (h)", fontsize=FS_AXIS)
    _style_ax(b)
    label_panel(b, "b")


    #     line, % axis, plateau ring). Both datasets agree: onset ~100 h, plateau ~30%.
    c = ax[1, 0]
    cgrid = np.arange(0, 301, 10)
    base = pf.mean() * 100
    c.axhline(base, color="0.55", ls="--", lw=0.8)
    halo_text(c, 292, base + 0.8, f"  base rate {base:.1f}%", transform=c.transData,
              ha="right", va="bottom", color="0.35")
    plateau_xs = []
    # plan Fig.S4c: IBTrACS dashed so the two datasets differ by line style too;
    # bands lighter (0.13) with a thin own-colour edge so overlaps stay readable
    for rt, col, lab, sd, ls in [(m.RT_CMA, C_PER, "CMA", 1, "-"),
                                 (m.RT_IB, OKABE_ITO[2], "IBTrACS", 2, "--")]:
        xs, pm, lo, hi = knn(rt, pf, cgrid, seed=sd)
        c.fill_between(xs, lo * 100, hi * 100, color=col, alpha=0.13, zorder=2)
        c.plot(xs, lo * 100, "-", color=col, lw=0.5, alpha=0.55, zorder=2.5)
        c.plot(xs, hi * 100, "-", color=col, lw=0.5, alpha=0.55, zorder=2.5)
        c.plot(xs, pm * 100, ls, color=col, lw=1.5, label=lab, zorder=4)
        isat = int(np.argmax(pm >= 0.9 * pm.max()))
        c.plot(xs[isat], pm[isat] * 100, "o", mfc="white", mec=col, mew=1.4,
               ms=5.0, zorder=6)
        plateau_xs.append(xs[isat])
    halo_text(c, 0.98, 0.28,
              f"plateau ≈ 30%\n$RT_e$ ≈ {min(plateau_xs):.0f}–{max(plateau_xs):.0f} h",
              ha="right", va="bottom", color="0.25")
    c.set_xlim(0, 300)
    c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    c.set_xlabel("$RT_e$ (h)", fontsize=FS_AXIS)
    c.set_ylabel("P(PEP | RT)", fontsize=FS_AXIS)
    c.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    _style_ax(c)
    label_panel(c, "c")

    # (d) translation-speed scatter
    d = ax[1, 1]
    mv = m.dropna(subset=["vt_CMA", "vt_IB"])
    for f, col in [(0, GREY_DEEP), (1, C_PER)]:
        s = mv[mv.PER_flag == f]
        d.scatter(s.vt_CMA, s.vt_IB, s=12, c=col, alpha=0.60, edgecolor="none", rasterized=True)
    rv = float(np.corrcoef(mv.vt_CMA, mv.vt_IB)[0, 1])
    mxv = float(np.nanmax([mv.vt_CMA.max(), mv.vt_IB.max()]))
    d.plot([0, mxv], [0, mxv], "k--", lw=0.6, alpha=0.5)
    halo_text(d, 0.03, 0.95, f"r = {rv:.2f}", fs=FS_ANNOT, color="0.1", va="top", ha="left")
    d.set_xlabel("translation speed CMA (km h$^{-1}$)", fontsize=FS_AXIS)
    d.set_ylabel("translation speed IBTrACS (km h$^{-1}$)", fontsize=FS_AXIS)
    _style_ax(d)
    label_panel(d, "d")

    return natural_save(fig, "FigS4_track_sensitivity")


if __name__ == "__main__":
    main()
