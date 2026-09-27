# Tropical cyclone rainfall residence time over China — analysis and figure code

Code for the paper:

> **Large-scale circulation regulates tropical cyclone-induced persistent extreme
> precipitation through residence time over China**

Tropical-cyclone residence time (RT) links large-scale circulation anomalies to
TC-induced persistent extreme precipitation (TC-PEP) over China. This
repository reproduces the paper's statistics and all 14 figures (6 main +
8 supplementary), and rebuilds the wide-domain circulation composites from
daily reanalysis.

## Pipeline (scripts numbered in execution order)

| Script | What it does |
|---|---|
| `00_make_all.py` | run the whole numbered pipeline in order |
| `01_rebuild_composites_wide.py` | rebuild wide-domain (60–180°E / 10°S–70°N) PEP/non-PEP and RT-quartile composites from daily NCEP/NCAR fields |
| `02_derive_composite_anomaly_wide.py` | derive the composite anomaly fields |
| `03_build_fig3_grids.py` | build the Fig. 3 spatial grids (cached to `figure_data/`) |
| `04`–`09_make_fig{1..6}.py` | main Figures 1–6 → `figures/` |
| `10`–`17_make_figS{1..8}.py` | supplementary Figures S1–S8 → `figures/` |

Supporting files: `config.py` (paths + frozen constants), `_style.py` (single
source of publication style) with `_figure_export.py` (journal width specs and
export checks), `MANIFEST.md` (per-figure outputs, layouts, data dependencies),
`assets/` (China shapefiles with the nine-dotted line, matplotlib style sheets).

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 00_make_all.py          # or run individual scripts in order
```

The small derived data files needed by the figure scripts are packaged in
`data/` (see `data/README.md`), so Figures 2, 5 and 6 render out of the box.
Other figures need more:

- `01_rebuild_composites_wide.py` reads daily NCEP/NCAR fields — point
  `$NCAR_DAILY_DIR` (or `$TCPEP_DATABANK`) at a local copy.
- Fig. 1 / Fig. 3 rebuild also needs the yearly TC precipitation stacks
  (`data/tc_prec_yearly/`, produced by the event-extraction pipeline from
  CMA best-track + CHM_PRE; not redistributed, ~61 MB).
- The supplementary figures read precomputed caches under `figure_data/`
  (not included; produced by the upstream sensitivity pipeline - see
  `data/README.md`).
- Fig. 1(d) embeds the schematic raster `fig1-d-body.png` (not included;
  available from the corresponding author on reasonable request).

## Data

| Dataset | Source |
|---|---|
| CHM_PRE / CHM_PRE V2 daily gridded precipitation | National Tibetan Plateau Data Center, <https://data.tpdc.ac.cn> |
| CMA best-track (6-hourly, WNP) | <https://tcdata.typhoon.org.cn> |
| NCEP/NCAR Reanalysis 1 daily | <https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html> |
| ERA5 reanalysis | Copernicus C3S, <https://cds.climate.copernicus.eu> |
| GPM IMERG daily | <https://gpm.nasa.gov/data/imerg> |
| IBTrACS v04 | <https://www.ncdc.noaa.gov/ibtracs> |

Each `data/<DATASET>/` subdirectory holds a README and `download-url.txt`
pointer. The derived event tables and composite fields included in `data/`
were produced with these inputs.

## Figure conventions

Maps go through `_style.py::add_china` — a layered China basemap including the
nine-dotted line (`assets/shapefiles/`), drawn consistently across all
figures. `save()` locks the npj (Nature Portfolio) double-column width (183 mm) for Fig. 1/2/5/6;
PNG export is 600 dpi.

## License

MIT — see `LICENSE`. Datasets remain under their providers' terms.
