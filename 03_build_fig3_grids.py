#!/usr/bin/env python3
"""Build the Fig. 3 spatial grids and cache them to figure_data/fig3_grids.npz.

Keys: per_grid/all_grid (event counts), per_rain/all_rain (cumulative TC
rainfall), per_rain_yearly with per-grid OLS trend_slope/trend_pval/trend_nyr,
rt_mean/rt_count (residence-time-weighted means), years, lat, lon.

Rainfall attribution: for each event, per-cell rainfall on its event days
within its footprint union; tc_prec is already TC-only, so all_rain sums all
days without masking.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

from config import DATA_DIR

FD = ROOT / "figure_data"
FD.mkdir(parents=True, exist_ok=True)
NPZ = FD / "fig3_grids.npz"
YEARS = list(range(1960, 2025))          # 1960–2024 (65 yr; matches tc_prec_yearly)
BASE_KEYS = ("per_grid", "all_grid", "lat", "lon")
HEAVY_KEYS = ("per_rain", "all_rain", "per_rain_yearly", "rt_mean", "rt_count", "years")
TREND_KEYS = ("trend_slope", "trend_pval", "trend_nyr")   # OLS per-grid trend (cached)


def _compute_trends(pry, years):
    """OLS slope (mm/dec) + p-value + PEP-year count, per cell. Theil–Sen is degenerate on
    Cells with <5 PEP-years stay NaN/p=1. Cached so the drawer never recomputes."""
    slope = np.full(pry.shape[1:], np.nan, np.float32)
    pval = np.full(pry.shape[1:], 1.0, np.float32)
    nyr = (pry > 0).sum(axis=0).astype(np.int32)
    ii, jj = np.where(nyr >= 5)
    for i, j in zip(ii, jj):
        res = linregress(years, pry[:, i, j])
        slope[i, j] = res.slope * 10.0
        pval[i, j] = res.pvalue
    return slope, pval, nyr


def _cache_state():
    """'hit' = full cache · 'augment' = heavy cached, only trend keys missing (cheap add)
    · 'rebuild' = nothing usable."""
    if not NPZ.exists():
        return "rebuild"
    files = set(np.load(NPZ, allow_pickle=True).files)
    if all(k in files for k in BASE_KEYS + HEAVY_KEYS + TREND_KEYS):
        return "hit"
    if all(k in files for k in HEAVY_KEYS):
        return "augment"
    return "rebuild"


def build(force=False):
    state = "rebuild" if force else _cache_state()
    if state == "hit":
        print(f"[fig3_grids] cache hit ({NPZ}); use --force to rebuild")
        return
    if state == "augment":
        z = np.load(NPZ, allow_pickle=True)
        slope, pval, nyr = _compute_trends(z["per_rain_yearly"], z["years"])
        data = {k: z[k] for k in z.files}
        data.update(trend_slope=slope, trend_pval=pval, trend_nyr=nyr)
        np.savez_compressed(NPZ, **data)
        print(f"[fig3_grids] augmented {NPZ} with OLS trend keys "
              f"({int((nyr >= 5).sum())} ≥5 PEP-yr cells)")
        return

    pe = pd.read_csv(DATA_DIR / "per_event_table.csv").set_index("event_id")
    ev_dates = (pd.read_csv(DATA_DIR / "tc_event_day_map.csv", dtype={"date": str})
                .groupby("event_id")["date"].apply(lambda s: list(s)).to_dict())

    # grid shape + coords from a sample precip file
    smp = sorted((DATA_DIR / "tc_prec_yearly").glob("tc_prec_*.nc"))[0]
    lat = xr.open_dataset(smp)["lat"].values
    lon = xr.open_dataset(smp)["lon"].values
    shape = (len(lat), len(lon))
    print(f"[fig3_grids] grid {shape}, {len(YEARS)} years → building (force={force})")

    per_grid = np.zeros(shape, np.int32)
    all_grid = np.zeros(shape, np.int32)
    all_rain = np.zeros(shape, np.float32)
    per_rain = np.zeros(shape, np.float32)
    rt_sum = np.zeros(shape, np.float32)
    rt_count = np.zeros(shape, np.int32)
    pry = np.zeros((len(YEARS),) + shape, np.float32)

    for yi, y in enumerate(YEARS):
        fp_prec = DATA_DIR / "tc_prec_yearly" / f"tc_prec_{y}.nc"
        fp_mask = DATA_DIR / "tc_footprint_yearly" / f"tc_footprint_{y}.npz"
        if not (fp_prec.exists() and fp_mask.exists()):
            print(f"  {y}: missing inputs, skip")
            continue
        ds = xr.open_dataset(fp_prec)
        prec = ds["tc_prec"].values.astype(np.float32)        # (ndays, lat, lon)
        date_idx = {pd.Timestamp(t).strftime("%Y%m%d"): i
                    for i, t in enumerate(ds["time"].values)}
        ds.close()
        all_rain += prec.sum(axis=0)
        mdict = {k: v for k, v in np.load(fp_mask, allow_pickle=True).items()}

        ey = pe[pe["year"] == y]
        n_ev = n_per = 0
        for eid, row in ey.iterrows():
            dates = ev_dates.get(eid)
            if not dates:
                continue
            tcid = int(row["tc_id"])
            union = np.zeros(shape, bool)
            for d in dates:
                m = mdict.get(f"{tcid}_{d}")
                if m is not None:
                    union |= m
            if not union.any():
                continue
            all_grid[union] += 1
            n_ev += 1
            rt_e = row["RT_e_hours"]
            if np.isfinite(rt_e):
                rt_sum[union] += rt_e
                rt_count[union] += 1
            if int(row["PER_flag"]) == 1:
                per_grid[union] += 1
                di = [date_idx[d] for d in dates if d in date_idx]
                if di:
                    cell = prec[di].sum(axis=0)               # event-day rainfall per cell
                    per_rain[union] += cell[union]
                    pry[yi][union] += cell[union]
                n_per += 1
        print(f"  {y}: events={n_ev} PEP={n_per}")

    rt_mean = np.where(rt_count > 0, rt_sum / np.maximum(rt_count, 1), np.nan).astype(np.float32)
    trend_slope, trend_pval, trend_nyr = _compute_trends(pry, np.array(YEARS, np.int32))
    np.savez_compressed(
        NPZ,
        per_grid=per_grid, all_grid=all_grid,
        per_rain=per_rain, all_rain=all_rain,
        per_rain_yearly=pry, years=np.array(YEARS, np.int32),
        rt_mean=rt_mean, rt_count=rt_count, lat=lat, lon=lon,
        trend_slope=trend_slope, trend_pval=trend_pval, trend_nyr=trend_nyr,
    )
    print(f"[fig3_grids] wrote {NPZ}")
    print(f"  per_rain max={per_rain.max():.0f} mm  all_rain max={all_rain.max():.0f} mm"
          f"  PEP-years active={int((pry.sum(axis=(1,2))>0).sum())}/{len(YEARS)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build(force=ap.parse_args().force)
