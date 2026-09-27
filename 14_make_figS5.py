#!/usr/bin/env python3
"""Fig. S5 - precipitation-product sensitivity (CHM_PRE vs IMERG, 2001-2024).

Reads data/si_imerg.csv and figure_data/figS5_hourly.npz.
Output: figures/FigS5_precip_sensitivity.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (natural_save, label_panel, C_PER, C_NPER, OKABE_ITO, FS_LEGEND,
                    FS_ANNOT, FS_TITLE, FS_TICK)
from config import DATA_DIR
FD = Path(__file__).resolve().parents[1] / "figure_data"


def knn(rt, flag, grid, k=80, seed=0):
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


def main():
    m = pd.read_csv(DATA_DIR / "si_imerg.csv")
    both = m.dropna(subset=["chm_PER_flag"]).copy()
    both["chm_PER_flag"] = both.chm_PER_flag.astype(int)
    both["PER_imerg"] = both.PER_imerg.astype(int)

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 2, left=0.09, right=0.975, top=0.955, bottom=0.065,
                          wspace=0.17, hspace=0.25)

    # (a) accumulation scatter
    a = fig.add_subplot(gs[0, 0])
    sub = both.dropna(subset=["A_e", "chm_accumulation_Ae_mm"])
    # plan Fig.S5(a,b): points more opaque; 1:1 reference a thin grey dash
    a.scatter(sub.chm_accumulation_Ae_mm, sub.A_e, c=OKABE_ITO[2], s=10, alpha=0.60,
              edgecolor="none", rasterized=True)
    mx = float(np.nanmax([sub.chm_accumulation_Ae_mm.max(), sub.A_e.max()]))
    a.plot([0, mx], [0, mx], ls="--", color="0.55", lw=0.6, alpha=0.9)
    r = float(np.corrcoef(sub.chm_accumulation_Ae_mm, sub.A_e)[0, 1])
    a.text(0.03, 0.95, f"r = {r:.2f} (n={len(sub)})", transform=a.transAxes, va="top",
           fontsize=FS_ANNOT, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    a.set_xlabel("$A_e$ CHM_PRE (mm)"); a.set_ylabel("$A_e$ IMERG (mm)")
    label_panel(a, "a")

    # (b) duration scatter
    b = fig.add_subplot(gs[0, 1])
    sub2 = both.dropna(subset=["duration", "chm_duration_days"])
    b.scatter(sub2.chm_duration_days, sub2.duration, c=OKABE_ITO[3], s=12, alpha=0.60,
              edgecolor="none", rasterized=True)
    mx = float(np.nanmax([sub2.chm_duration_days.max(), sub2.duration.max()]))
    b.plot([0, mx], [0, mx], ls="--", color="0.55", lw=0.6, alpha=0.9)
    rd = float(np.corrcoef(sub2.chm_duration_days, sub2.duration)[0, 1])
    b.text(0.03, 0.95, f"r = {rd:.2f}", transform=b.transAxes, va="top", fontsize=FS_ANNOT,
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    b.set_xlabel("Duration CHM_PRE (days)"); b.set_ylabel("Duration IMERG (days)")
    label_panel(b, "b")

    # (c) confusion matrix
    c = fig.add_subplot(gs[1, 0])
    cm = np.zeros((2, 2), dtype=int)
    for im, ch in zip(both.PER_imerg, both.chm_PER_flag):
        cm[im, ch] += 1
    im_obj = c.imshow(cm, cmap="Blues", origin="lower")
    for i in range(2):
        for j in range(2):
            # plan Fig.S5c: counts enlarged, exactly centred, dark/light by cell
            c.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                   color="white" if cm[i, j] > cm.max() * 0.6 else "0.1")
    c.set_xticks([0, 1]); c.set_yticks([0, 1])
    c.set_xticklabels(["non-PEP", "PEP"], fontsize=FS_TICK)
    c.set_yticklabels(["non-PEP", "PEP"], fontsize=FS_TICK)
    c.set_xlabel("CHM_PRE classification"); c.set_ylabel("IMERG classification")
    c.set_title("PEP overlap", fontsize=FS_TITLE)   # plan: same class as panel titles
    prec = cm[1, 1] / max(cm[1, :].sum(), 1)
    rec = cm[1, 1] / max(cm[:, 1].sum(), 1)
    c.text(0.5, -0.32, f"precision {prec*100:.0f}% · recall {rec*100:.0f}%",
           transform=c.transAxes, ha="center", fontsize=FS_ANNOT)

    # (d) P(PEP|RT)
    d = fig.add_subplot(gs[1, 1])
    rt = both.chm_RT_local_hours.dropna()
    flag_chm = both.chm_PER_flag.reindex(rt.index).values
    flag_img = both.PER_imerg.reindex(rt.index).values
    grid = np.arange(0, 301, 15)
    for fl, col, lab, sd in [(flag_chm, C_PER, "CHM_PRE", 1), (flag_img, OKABE_ITO[2], "IMERG", 2)]:
        xs, pm, lo, hi = knn(rt.values, fl, grid, seed=sd)
        # plan Fig.S5d: lighter bands, stronger main lines (more contrast)
        d.fill_between(xs, lo * 100, hi * 100, color=col, alpha=0.12)
        d.plot(xs, pm * 100, color=col, lw=1.6, label=lab)
    d.set_xlabel("$RT_{local}$ (h)"); d.set_ylabel("P(PEP | RT) (%)")
    d.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left", labelspacing=0.3)
    label_panel(d, "d")

    # (e) hourly lifecycle
    e = fig.add_subplot(gs[2, :])
    try:
        hz = np.load(FD / "figS5_hourly.npz", allow_pickle=True)
        dt = float(hz["dt_hours"]); pre = int(hz["pre_days"])
        onset = int(pre * 24 / dt)
        rs = np.random.RandomState(0)
        for key, col, lab in [("PER", C_PER, "PEP"), ("nonPER", C_NPER, "non-PEP")]:
            if key not in hz.files:
                continue
            tr = hz[key]                 # (n_trace, n_halfhour)
            mean = np.nanmean(tr, axis=0)
            draws = tr[rs.randint(0, tr.shape[0], size=(300, tr.shape[0]))]
            lo, hi = np.nanpercentile(np.nanmean(draws, axis=1), [2.5, 97.5], axis=0)
            xh = (np.arange(tr.shape[1]) - onset) * dt / 24.0   # days from onset
            # plan Fig.S5e: ~1.2 pt main lines, lighter CI, thin onset dash;
            # half-hourly detail and the grey line's original stop untouched
            e.fill_between(xh, lo, hi, color=col, alpha=0.12, lw=0)
            e.plot(xh, mean, color=col, lw=1.2, label=f"{lab} (n={tr.shape[0]})")
        e.axvline(0, color="0.5", lw=0.5, ls="--")
        e.set_xlim(left=-2)
        e.set_xlabel("days from onset (IMERG half-hourly)"); e.set_ylabel("Area-mean precip (mm h$^{-1}$)")
        e.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
        e.grid(axis="y", lw=0.3, alpha=0.4)
        for sp in ("top", "right"):
            e.spines[sp].set_visible(False)
    except Exception as ex:
        e.text(0.5, 0.5, f"hourly data unavailable ({ex})", transform=e.transAxes, ha="center")

    # Align (c) and (e) panel markers to (a)'s left edge. imshow() forces equal
    # aspect, which shrinks panel c's axes (and shifts it right) inside its cell;
    # panel e spans the full width. In both cases the default axes-fraction x=-0.14
    # lands at a different figure-x than (a), so the three markers are not flush.
    # Pin them to (a)'s label figure-x so they read as one left-aligned column.
    fig.canvas.draw()
    # shrink (c) confusion matrix to 80% about its centre (post-draw: equal-aspect
    # box is already realised; 0.8x0.8 keeps it square so apply_aspect preserves it)
    bbc = c.get_position()
    c.set_position([bbc.x0 + 0.1 * bbc.width, bbc.y0 + 0.1 * bbc.height,
                    0.8 * bbc.width, 0.8 * bbc.height])
    xa = a.get_position().x0 + (-0.14) * a.get_position().width
    for ax, letter in ((c, "c"), (e, "e")):
        bb = ax.get_position()
        label_panel(ax, letter, x=(xa - bb.x0) / bb.width)

    return natural_save(fig, "FigS5_precip_sensitivity")


if __name__ == "__main__":
    main()
