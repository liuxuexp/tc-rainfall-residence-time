#!/usr/bin/env python3
"""Rebuild the Fig. 4 composite fields on the wide 60-180E / 10S-70N frame.

Reads daily NCEP/NCAR fields (override the default path with $NCAR_DAILY_DIR)
and the event-day sets in data/. Computes 1000-300 hPa IVT/TCWV, 850-300 hPa
mass-weighted steering flow, spherical-divergence MFC, omega500 and hgt500;
anomalies are de-seasonalised against the same-calendar-day climatology.

Writes:
  data/era5_event_composite_wide.nc    per_*/nonper_*/clim_* fields
  data/composite_RTquartile_wide.nc    ext_anom_*/sho_anom_*/diff_* fields
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = ROOT / "data"
import os
NCAR_DIR = Path(os.environ.get(
    "NCAR_DAILY_DIR",
    "/data-02/home/20091008/databank/NCAR/Data-NCAR/data/daily"))
YEARS = range(1960, 2025)


LAT0, LAT1 = -10.0, 70.0
LON0, LON1 = 60.0, 180.0

G_GRAV = 9.80665
R_EARTH_M = 6_371_000.0
LEV = np.array([1000, 925, 850, 700, 600, 500, 400, 300], dtype=float)
EDGES = np.concatenate([[LEV[0]], 0.5 * (LEV[:-1] + LEV[1:]), [LEV[-1]]])
DP_PA = np.array([EDGES[i] - EDGES[i + 1] for i in range(len(LEV))]) * 100.0
SLEV = np.array([850, 700, 600, 500, 400, 300], dtype=float)
SEDGES = np.concatenate([[SLEV[0]], 0.5 * (SLEV[:-1] + SLEV[1:]), [SLEV[-1]]])
SDP = np.array([SEDGES[i] - SEDGES[i + 1] for i in range(len(SLEV))]) * 100.0
VARS = ["IVT_u", "IVT_v", "IVT", "TCWV", "MFC", "steer_u", "steer_v",
        "omega500", "hgt500"]


def load_field(v, year, levels=None):
    """Load one sub-domain field slice for a given year."""
    ds = xr.open_dataset(NCAR_DIR / f"{v}.{year}.nc")
    var = v if v in ds.data_vars else [k for k in ds.data_vars][0]
    lat = ds["lat"].values
    ilat = np.where((lat >= LAT0) & (lat <= LAT1))[0]
    lon = ds["lon"].values
    ilon = np.where((lon >= LON0) & (lon <= LON1))[0]
    times = ds["time"].values
    if levels is not None and "level" in ds[var].dims:
        lev = ds["level"].values
        ilev = np.array([int(np.where(lev == L)[0][0]) for L in levels])
        arr = ds[var].isel(lat=ilat, lon=ilon, level=ilev).values
        arr = arr[:, :, ::-1, :] if lat[ilat[0]] > lat[ilat[-1]] else arr  # lat ascending
        arr = arr[:, np.argsort(levels), :, :]
    else:
        arr = ds[var].isel(lat=ilat, lon=ilon).values
        arr = arr[:, ::-1, :] if (arr.ndim == 3 and lat[ilat[0]] > lat[ilat[-1]]) else arr
    ds.close()
    return np.asarray(arr), np.sort(lat[ilat]), lon[ilon], times


def compute_mfc(IVT_u, IVT_v, lat, lon):
    """MFC = -div(IVT) on the sphere (mm/day)."""
    dphi = np.radians(2.5)
    dlam = np.radians(2.5)
    R = R_EARTH_M
    cosphi = np.cos(np.radians(lat))[:, None]
    du_dlam = np.gradient(IVT_u, dlam, axis=-1)
    dvcos_dphi = np.gradient(IVT_v * cosphi[None, :, :], dphi, axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        div = (1.0 / (R * cosphi[None, :, :])) * (du_dlam + dvcos_dphi)
    div = np.where(np.isfinite(div), div, 0.0)
    return -div * 86400.0


def compute_year_fields(year):
    q, lat, lon, t = load_field("shum", year, levels=LEV)     # [T,8,nlat,nlon] kg/kg
    u, _, _, _ = load_field("uwnd", year, levels=LEV)
    v, _, _, _ = load_field("vwnd", year, levels=LEV)
    dp = DP_PA.reshape(1, -1, 1, 1)
    IVT_u = np.sum(q * u * dp, axis=1) / G_GRAV
    IVT_v = np.sum(q * v * dp, axis=1) / G_GRAV
    TCWV = np.sum(q * dp, axis=1) / G_GRAV
    sidx = np.array([int(np.where(LEV == L)[0][0]) for L in SLEV])
    us, vs = u[:, sidx], v[:, sidx]
    sdp = SDP.reshape(1, -1, 1, 1)
    denom = float(np.sum(SDP))
    steer_u = np.sum(us * sdp, axis=1) / denom
    steer_v = np.sum(vs * sdp, axis=1) / denom
    om, _, _, _ = load_field("omega", year, levels=[500])
    hg, _, _, _ = load_field("hgt", year, levels=[500])
    return dict(IVT_u=IVT_u, IVT_v=IVT_v, IVT=np.hypot(IVT_u, IVT_v),
                TCWV=TCWV, MFC=compute_mfc(IVT_u, IVT_v, lat, lon),
                steer_u=steer_u, steer_v=steer_v,
                omega500=om[:, 0, :, :], hgt500=hg[:, 0, :, :]), lat, lon, t


def build_day_sets():
    dm = pd.read_csv(DATA_DIR / "tc_event_day_map.csv", dtype={"date": str})
    pe = pd.read_csv(DATA_DIR / "per_event_table.csv")
    rt = pd.read_csv(DATA_DIR / "residence_time_metrics.csv")
    m = dm.merge(pe[["event_id", "PER_flag"]], on="event_id")
    per = set(m.loc[m["PER_flag"] == 1, "date"].astype(str))
    non = set(m.loc[m["PER_flag"] == 0, "date"].astype(str))
    m2 = dm.merge(rt[["event_id", "RT_quartile"]], on="event_id")
    ext = set(m2.loc[m2["RT_quartile"] == "Extreme", "date"].astype(str))
    sho = set(m2.loc[m2["RT_quartile"] == "Short", "date"].astype(str))
    return per, non, ext, sho


def main():
    print(f"[wide] recompositing on {LON0}-{LON1}E / {LAT0}-{LAT1}N ...")
    per_days, non_days, ext_days, sho_days = build_day_sets()
    print(f"  PEP event-days {len(per_days)}; non-PEP {len(non_days)}; "
          f"Extreme-RT {len(ext_days)}; Short-RT {len(sho_days)}")
    per_sum = {k: None for k in VARS}
    non_sum = {k: None for k in VARS}
    ext_sum = {k: None for k in VARS}
    sho_sum = {k: None for k in VARS}
    clim_sum, clim_cnt = {}, {}
    lat = lon = None
    for y in YEARS:
        try:
            fields, lat, lon, times = compute_year_fields(y)
        except FileNotFoundError:
            print(f"  [skip] {y} (missing NCAR)")
            continue
        doy = pd.DatetimeIndex(times).dayofyear.values
        for ti, d in enumerate(doy):
            rec = clim_sum.setdefault(d, {})
            for k in VARS:
                rec[k] = rec.get(k, 0) + fields[k][ti]
            clim_cnt[d] = clim_cnt.get(d, 0) + 1
        dates = pd.DatetimeIndex(times).strftime("%Y%m%d")
        for di, d in enumerate(dates):

            # PEP and a non-PEP event-day feeds the PEP composite ONLY (82 such
            # dates 1960-2024). The first wide run fed BOTH sets and silently
            # inflated nonper_* by ~2.7 % of level (hgt500 +156 m); caught by the


            # BOTH composites (44 such dates).
            if d in per_days:
                for k in VARS:
                    per_sum[k] = (fields[k][di].astype(np.float64).copy() if per_sum[k] is None
                                  else per_sum[k] + fields[k][di])
            elif d in non_days:
                for k in VARS:
                    non_sum[k] = (fields[k][di].astype(np.float64).copy() if non_sum[k] is None
                                  else non_sum[k] + fields[k][di])
            in_ext, in_sho = d in ext_days, d in sho_days
            if in_ext or in_sho:
                for k in VARS:
                    fv = fields[k][di]
                    if in_ext:
                        ext_sum[k] = (fv.astype(np.float64).copy() if ext_sum[k] is None
                                      else ext_sum[k] + fv)
                    if in_sho:
                        sho_sum[k] = (fv.astype(np.float64).copy() if sho_sum[k] is None
                                      else sho_sum[k] + fv)
        print(f"  {y} done", flush=True)

    n_per, n_non = len(per_days), len(non_days)
    n_ext, n_sho = len(ext_days), len(sho_days)
    doys = sorted(clim_sum.keys())
    clim = {k: np.stack([clim_sum[d][k] / clim_cnt[d] for d in doys]) for k in VARS}

    # --- era5_event_composite_wide.nc (key layout as 08) ---
    ds1 = xr.Dataset(
        {f"per_{k}": (("lat", "lon"), per_sum[k] / n_per) for k in VARS}
        | {f"nonper_{k}": (("lat", "lon"), non_sum[k] / n_non) for k in VARS}
        | {f"clim_{k}": (("doy", "lat", "lon"), clim[k]) for k in VARS},
        coords={"lat": ("lat", lat), "lon": ("lon", lon), "doy": ("doy", doys)})
    ds1.to_netcdf(DATA_DIR / "era5_event_composite_wide.nc",
                  encoding={v: {"zlib": True, "complevel": 4} for v in ds1.data_vars})
    print(f"  wrote data/era5_event_composite_wide.nc")

    # --- composite_RTquartile_wide.nc (anomalies + diff, key layout as 09b) ---
    def cb(dates_set, k):
        dy = sorted({pd.Timestamp(d).timetuple().tm_yday for d in dates_set})
        da = xr.DataArray(clim[k], dims=("doy", "lat", "lon"), coords={"doy": doys})
        return da.sel(doy=dy).mean("doy").values
    ext_anom = {k: ext_sum[k] / n_ext - cb(ext_days, k) for k in VARS}
    sho_anom = {k: sho_sum[k] / n_sho - cb(sho_days, k) for k in VARS}
    diff = {k: ext_anom[k] - sho_anom[k] for k in VARS}
    ds2 = xr.Dataset(
        {f"ext_anom_{k}": (("lat", "lon"), ext_anom[k]) for k in VARS}
        | {f"sho_anom_{k}": (("lat", "lon"), sho_anom[k]) for k in VARS}
        | {f"diff_{k}": (("lat", "lon"), diff[k]) for k in VARS},
        coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    ds2.to_netcdf(DATA_DIR / "composite_RTquartile_wide.nc",
                  encoding={v: {"zlib": True, "complevel": 4} for v in ds2.data_vars})
    print(f"  wrote data/composite_RTquartile_wide.nc")


    old = xr.open_dataset(DATA_DIR / "composite_RTquartile.nc")
    ola = np.where((lat >= old.lat.values[1]) & (lat <= old.lat.values[-2]))[0]
    olo = np.where((lon >= old.lon.values[1]) & (lon <= old.lon.values[-2]))[0]
    print("\n  overlap 72.5-137.5E / 7.5-52.5N (interior of the OLD domain):")
    for k in VARS:
        a = old[f"diff_{k}"].values[1:-1, 1:-1]        # old grid, minus 1-cell border
        b = diff[k][np.ix_(ola, olo)]                  # same cells on the wide grid
        print(f"    diff_{k:9s}: max|wide-narrow| = {np.nanmax(np.abs(a - b)):.3g}")
    old.close()


    old08 = xr.open_dataset(DATA_DIR / "era5_event_composite.nc")
    for k in VARS:
        for pre in ("per_", "nonper_"):
            a = old08[f"{pre}{k}"].values[1:-1, 1:-1]
            b = ds1[f"{pre}{k}"].values[np.ix_(ola, olo)]
            print(f"    {pre}{k:9s}: max|wide-narrow| = {np.nanmax(np.abs(a - b)):.3g}")
    old08.close()


if __name__ == "__main__":
    main()
