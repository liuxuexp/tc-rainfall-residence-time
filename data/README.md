# data/

Two kinds of content live here:

1. **Packaged derived tables** (included in the repository, ~14 MB total) — the
   outputs of the event-analysis pipeline (event identification from CMA
   best-track + CHM_PRE, residence-time metrics, per-event reanalysis
   diagnostics) that the figure scripts read directly.
2. **Per-dataset pointer directories** (`CHM_PRE_V2/`, `CMABSTdata/`,
   `Data-NCAR/`, `ERA5/`, `IMERG/`, `IBTrACS/`) — each with a README and
   `download-url.txt` for the large public inputs, which are not
   redistributed.

## Packaged files

| File | What it is | Used by |
|---|---|---|
| `tc_events_full.csv` | master event table: per-event metrics, RT, PEP flag | Fig 1, 2, 5, S2, S3, S6, S8 |
| `per_event_table.csv` | event table with PEP flag (pipeline layout) | 01, 03 |
| `tc_event_day_map.csv` | event ↔ calendar-day map | 01, 03, Fig 1 |
| `residence_time_metrics.csv` | per-event residence time + RT quartile | 01 |
| `tc_track_table.csv` | 6-hourly CMA track positions | Fig 1 |
| `tc_footprint_yearly/` | yearly event footprint masks | Fig 1 |
| `composite_RTquartile.nc` / `_wide.nc` | Extreme/Short-RT anomaly + diff composites | Fig 4, S7 |
| `composite_anomaly.nc` / `_wide.nc` | PEP−non-PEP anomaly composites | 02, S7 |
| `event_moisture_budget.csv` / `.npz` | per-event moisture-budget terms | Fig 4, S7 |
| `per_event_environment.csv` | per-event environment conditions | Fig 5 |
| `yearly_core_tcwv_ivt.csv` | yearly South-China JJAS TCWV/IVT | Fig 5 |
| `trend_post1979.csv` | post-1979 trend table | Fig 5 |
| `si_radius.csv` | influence-radius sensitivity summary | S3 |
| `si_ibtracs.csv` | best-track dataset sensitivity summary | S4 |
| `si_imerg.csv` | precipitation-product sensitivity summary | S5 |

`figure_data/` is not included in the repository. `fig3_grids.npz` is rebuilt
by `03_build_fig3_grids.py` (needs `data/tc_prec_yearly/`); the `figS*_*.csv/.npz`
caches consumed by the supplementary figures are products of the upstream
sensitivity pipeline and are available from the corresponding author on
reasonable request.

## Not included (and how to get them back)

| Item | Size | How |
|---|---|---|
| `tc_prec_yearly/` | ~61 MB | regenerate from CMA best-track + CHM_PRE with the event-extraction pipeline; only needed to rebuild Fig. 1 caches / Fig. 3 grids from scratch |
| `era5_event_composite.nc`, `era5_event_composite_wide.nc` | 14 / 30 MB | rebuilt by `01_rebuild_composites_wide.py` from daily NCEP/NCAR fields (needed by Fig. 4) |
| raw CHM_PRE V2 / CMA / NCEP-NCAR / ERA5 / IMERG / IBTrACS | — | official sources listed in the pointer directories and the main README |

CHM_PRE: <https://data.tpdc.ac.cn>
