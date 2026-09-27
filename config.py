"""Shared configuration: paths and frozen constants.

Set $TCPEP_DATABANK to point at a local databank mirror (CMA BST / CHM_PRE V2 /
NCEP-NCAR / IMERG / IBTrACS; see data/README.md). Nothing here is a tunable
parameter - fixed physical constants and frozen paths only.
"""
from pathlib import Path

# --- paths ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent          # project root
import os
DATABANK = Path(os.environ.get("TCPEP_DATABANK", "/data-02/home/20091008/databank"))

CMA_DIR      = DATABANK / "CMABSTdata"
CHMPRE_DIR   = DATABANK / "CHM_PRE_V2" / "daily"   # https://data.tpdc.ac.cn
NCAR_DIR     = DATABANK / "NCAR" / "Data-NCAR" / "data" / "daily"
IMERG_DF_DIR = DATABANK / "IMERG" / "3IMERGDF"
IBTRACS_NC   = DATABANK / "IBTrACS" / "IBTrACS.ALL.v04r01.nc"
SHAPE_DIR    = DATABANK / "china-shapefiles" / "shapefiles"
CHINA_SHP    = SHAPE_DIR / "china_country.shp"

DATA_DIR     = PROJECT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


R_EARTH_M   = 6_371_000.0          # mean Earth radius [m]
G_GRAV      = 9.80665              # gravity [m s-2]  (for IVT/MFC/W integrals)
BUFFER_KM   = 500
INFLUENCE_KM = 500
P_THRESHOLD_MM = 1.0


#   0 = sub-TD weak low / disturbance (mean wind ~5.7 m/s)
#   1..6 = TD, TS, STS, TY, STY, SuperTY
#   9 = extratropical (post-transition; mean lat ~41N, fast ~40 km/h, weak wind)
CMA_GRADE = {0: "LOW", 1: "TD", 2: "TS", 3: "STS", 4: "TY", 5: "STY",
             6: "SuperTY", 9: "EX"}

# Study years (CMA Best Track complete 1960-2024)
YEARS = range(1960, 2025)          # 1960..2024 inclusive -> 65 years
