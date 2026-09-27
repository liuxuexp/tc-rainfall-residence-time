#!/usr/bin/env python3
"""Fig. S6 - residence-time definition and its link to event metrics.

Reads data/tc_events_full.csv, figure_data/figS6_curves.csv and
figure_data/figS6_corr.csv. Output: figures/FigS6_RT_definition.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from _style import (natural_save, label_panel, halo_text, C_PER, C_NPER, OKABE_ITO,
                    FS_ANNOT, FS_LEGEND, FS_TICK)
from config import DATA_DIR
FD = Path(__file__).resolve().parents[1] / "figure_data"
GREY_DEEP = "#707070"   # plan Fig.S6a: grey class deepened one step (as S4)


def main():
    df = pd.read_csv(DATA_DIR / "tc_events_full.csv").dropna(
        subset=["RT_e_hours", "RT_local_hours", "duration_days",
                "accumulation_Ae_mm", "peak_daily_mm"])
    pf = df["PER_flag"].values.astype(int)
    curves = pd.read_csv(FD / "figS6_curves.csv")
    corr = pd.read_csv(FD / "figS6_corr.csv")

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.95, bottom=0.085,
                        wspace=0.15, hspace=0.17)

    # (a) scatter RT_e vs RT_local
    a = ax[0, 0]
    for f, col, lab in [(0, GREY_DEEP, f"non-PEP ({(pf==0).sum()})"),
                        (1, C_PER, f"PEP ({(pf==1).sum()})")]:
        s = df[pf == f]
        # plan Fig.S6a: uniform size/alpha per class, grey deepened, dash light
        a.scatter(s.RT_e_hours, s.RT_local_hours, s=10, c=col, alpha=0.60,
                  edgecolor="none", rasterized=True, label=lab)
    r = float(np.corrcoef(df.RT_e_hours, df.RT_local_hours)[0, 1])
    a.text(0.03, 0.95, f"r = {r:.2f}", transform=a.transAxes, va="top", fontsize=FS_ANNOT,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
    a.plot([0, df.RT_e_hours.max()], [0, df.RT_e_hours.max()], ls="--",
           color="0.55", lw=0.6, alpha=0.9)
    a.set_xlabel("$RT_e$ (h)"); a.set_ylabel("$RT_{local}$ (h)")
    a.legend(fontsize=FS_LEGEND, frameon=False, loc="lower right")
    label_panel(a, "a")

    # (b) two-panel violin
    b = ax[0, 1]
    data = [df.RT_e_hours[pf == 0], df.RT_e_hours[pf == 1],
            df.RT_local_hours[pf == 0], df.RT_local_hours[pf == 1]]
    pos = [1, 2, 4, 5]; cols = [C_NPER, C_PER, C_NPER, C_PER]
    vp = b.violinplot(data, positions=pos, widths=0.8, showmedians=False,
                      showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        # plan Fig.S6b: soft fill + crisp outline, one brightness level
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(0.30); body.set_linewidth(0.9)
    # manual per-violin median lines
    from scipy.stats import gaussian_kde
    for p, d, col in zip(pos, data, cols):
        med = float(np.median(d))
        b.hlines(med, p - 0.34, p + 0.34, colors=col, lw=1.8, zorder=4)
    b.axvline(3.0, color="0.75", lw=0.4)     # plan: centre divider lighter/thinner
    for p, d in zip(pos, data):
        halo_text(b, p, 0.97, f"{np.median(d):.0f} h", transform=b.get_xaxis_transform(),
                  ha="center", va="top")
    b.set_xticks([1.5, 4.5]); b.set_xticklabels(["$RT_e$", "$RT_{local}$"],
                                                fontsize=FS_ANNOT)
    b.set_ylabel("residence time (h)")
    label_panel(b, "b")


    #   base-rate dashed line + halo tag, % y-ticks, 90%-of-max plateau rings per
    #   metric (colour-coded to the legend). The two rings co-saturate (~30%,
    #   150–165 h), the visual point of the sensitivity panel.
    c = ax[1, 0]
    g = curves.rt_grid.values
    base = pf.mean() * 100
    # plan Fig.S6c: RT_local dashed for style separation; bands lighter (0.12)
    for col, lo, hi, pm, lab, ls in [
            (C_PER,        "lo_e", "hi_e", "p_per_RT_e",     "$RT_e$",        "-"),
            (OKABE_ITO[2], "lo_l", "hi_l", "p_per_RT_local", "$RT_{local}$", "--")]:
        pm_v = curves[pm].values
        c.fill_between(g, curves[lo] * 100, curves[hi] * 100,
                       color=col, alpha=0.12, zorder=2)
        c.plot(g, pm_v * 100, ls, color=col, lw=1.5, label=lab, zorder=4)
        isat = int(np.argmax(pm_v >= 0.9 * pm_v.max()))
        c.plot(g[isat], pm_v[isat] * 100, "o", mfc="white", mec=col,
               mew=1.4, ms=5.0, zorder=6)
    c.axhline(base, color="0.55", ls="--", lw=0.8)
    halo_text(c, 290, base + 0.8, f"base rate {base:.1f}%", transform=c.transData,
              ha="right", va="bottom", color="0.35")
    c.set_xlim(0, 300)
    c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    c.set_xlabel("residence time (h)"); c.set_ylabel("P(PEP | RT)")
    c.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left",
             bbox_to_anchor=(0.02, 0.98))
    label_panel(c, "c")

    # (d) grouped corr bars
    d = ax[1, 1]
    metrics = corr["metric"].tolist()[:3]            # duration, accumulation, peak
    x = np.arange(len(metrics)); w = 0.36
    d.bar(x - w / 2, corr.RT_e.values[:3], w, color=C_PER, label="$RT_e$")
    d.bar(x + w / 2, corr.RT_local.values[:3], w, color=OKABE_ITO[2], label="$RT_{local}$")
    d.set_xticks(x); d.set_xticklabels(["Duration", "Accum.", "Peak"], fontsize=FS_TICK)
    d.set_ylabel("Pearson $r$"); d.set_ylim(0, 1.10)
    d.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    for i, (re, rl) in enumerate(zip(corr.RT_e.values[:3], corr.RT_local.values[:3])):
        d.text(i - w / 2, re + 0.02, f"{re:.2f}", ha="center", fontsize=FS_ANNOT)
        d.text(i + w / 2, rl + 0.02, f"{rl:.2f}", ha="center", fontsize=FS_ANNOT)
    label_panel(d, "d")

    return natural_save(fig, "FigS6_RT_definition")


if __name__ == "__main__":
    main()
