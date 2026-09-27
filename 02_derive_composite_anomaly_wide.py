#!/usr/bin/env python3
"""Derive the PEP-minus-non-PEP anomaly composites on the wide Fig. 4 frame.

Reads the group composites written by 01_rebuild_composites_wide.py, subtracts
the same-calendar-day climatological mean of each event-day set, and writes
data/composite_anomaly_wide.nc (variable names mirror the narrow-domain file).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = ROOT / "data"
VARS = ["IVT_u", "IVT_v", "IVT", "TCWV", "MFC", "steer_u", "steer_v",
        "omega500", "hgt500"]          # as rebuild_composites_wide.py


def day_sets():
    """PEP / non-PEP event-day date sets (as in 01_rebuild_composites_wide)."""
    dm = pd.read_csv(DATA_DIR / "tc_event_day_map.csv", dtype={"date": str})
    pe = pd.read_csv(DATA_DIR / "per_event_table.csv")
    m = dm.merge(pe[["event_id", "PER_flag"]], on="event_id")
    per = set(m.loc[m["PER_flag"] == 1, "date"].astype(str))
    non = set(m.loc[m["PER_flag"] == 0, "date"].astype(str))
    return per, non


def group_clim(clim, doys, days_set, k):
    """Group climatology baseline for var k."""
    dy = sorted({pd.Timestamp(d).timetuple().tm_yday for d in days_set})
    da = xr.DataArray(clim[k], dims=("doy", "lat", "lon"), coords={"doy": doys})
    return da.sel(doy=dy).mean("doy").values


def derive(era_path):
    """per_anom_* / non_anom_* / diff_* from an era5_event_composite*.nc group-mean file."""
    era = xr.open_dataset(era_path)
    per_days, non_days = day_sets()
    doys = era["doy"].values
    clim = {k: era[f"clim_{k}"].values for k in VARS}
    per_anom = {k: era[f"per_{k}"].values - group_clim(clim, doys, per_days, k)
                for k in VARS}
    non_anom = {k: era[f"nonper_{k}"].values - group_clim(clim, doys, non_days, k)
                for k in VARS}
    lat, lon = era["lat"].values, era["lon"].values
    era.close()
    diff = {k: per_anom[k] - non_anom[k] for k in VARS}
    return per_anom, non_anom, diff, lat, lon


def main():

    per_a, non_a, diff_n, lat_n, lon_n = derive(DATA_DIR / "era5_event_composite.nc")
    ref = xr.open_dataset(DATA_DIR / "composite_anomaly.nc")
    worst = max(np.nanmax(np.abs(diff_n[k] - ref[f"diff_{k}"].values)) for k in VARS)
    ref.close()
    print(f"[check a] narrow round-trip max|formula − submitted diff| = {worst:.3g}")
    if worst != 0:
        sys.exit("ABORT: formula does not reproduce the submitted narrow diff_*")

    # wide derivation from the already-recomputed wide group means
    per_anom, non_anom, diff, lat, lon = derive(DATA_DIR / "era5_event_composite_wide.nc")
    assert lat[0] == -10.0 and lat[-1] == 70.0 and lon[0] == 60.0 and lon[-1] == 180.0

    ds = xr.Dataset(
        {f"per_anom_{k}": (("lat", "lon"), per_anom[k]) for k in VARS}
        | {f"non_anom_{k}": (("lat", "lon"), non_anom[k]) for k in VARS}
        | {f"diff_{k}": (("lat", "lon"), diff[k]) for k in VARS},
        coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    out = DATA_DIR / "composite_anomaly_wide.nc"
    ds.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"[write] {out.relative_to(ROOT)}")


    ola = np.where((lat >= lat_n[1]) & (lat <= lat_n[-2]))[0]
    olo = np.where((lon >= lon_n[1]) & (lon <= lon_n[-2]))[0]
    ref = xr.open_dataset(DATA_DIR / "composite_anomaly.nc")
    print("[check b] overlap interior 72.5-137.5E / 7.5-52.5N (minus 1-cell border):")
    for k in VARS:
        a = ref[f"diff_{k}"].values[1:-1, 1:-1]
        b = diff[k][np.ix_(ola, olo)]
        print(f"    diff_{k:9s}: max|wide-narrow| = {np.nanmax(np.abs(a - b)):.3g}")
    ref.close()


if __name__ == "__main__":
    main()
