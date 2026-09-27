# Figure manifest

The 14 deliverable figures (6 main + 8 supplementary), their scripts, panel
layouts, data dependencies and width class. All outputs land in `figures/`
as 600-dpi PNG; a vector PDF working copy of each goes to `figures/pdf/`.

| # | Output | Script | Layout | Data dependencies | Width |
|---|---|---|---|---|---|
| Fig 1 | `Fig1_PEP_definition.png` | `04_make_fig1.py` | 2×2: (a) yearly amount + event count; (b) duration–accumulation space; (c) two case-map panels + shared colorbar; (d) schematic (embedded `fig1-d-body.png`*) | `tc_events_full.csv`, `tc_event_day_map.csv`, `tc_prec_yearly/*.nc`*, `tc_track_table.csv`, `tc_footprint_yearly/*.npz` | 183 mm |
| Fig 2 | `Fig2_RT_controls.png` | `05_make_fig2.py` | 3×2 (a–f) | `tc_events_full.csv` | 183 mm |
| Fig 3 | `Fig3_PEP_spatial.png` | `06_make_fig3.py` | 3×2 maps + violins, South China Sea inset | `figure_data/fig3_grids.npz` (from `03_build_fig3_grids.py`) | natural |
| Fig 4 | `Fig4_mechanism.png` | `07_make_fig4.py` | 3×2 maps (a–f, shared frame 60–180°E/10°S–70°N) + g lifecycle / h moisture budget | `composite_RTquartile_wide.nc`, `era5_event_composite_wide.nc`*, `event_moisture_budget.csv`/`.npz` | natural |
| Fig 5 | `Fig5_trends_attribution.png` | `08_make_fig5.py` | 3×2 (a–f) | `tc_events_full.csv`, `yearly_core_tcwv_ivt.csv`, `trend_post1979.csv`, `per_event_environment.csv` | 183 mm |
| Fig 6 | `Fig6_conceptual_model.png` | `09_make_fig6.py` | single-panel conceptual schematic | none | 183 mm |
| S1 | `FigS1_threshold_sensitivity.png` | `10_make_figS1.py` | 2×2 ImageGrid of 3×3 matrices | `figure_data/figS1.csv` (= Table S1) | natural |
| S2 | `FigS2_severe_PEP.png` | `11_make_figS2.py` | 2×2 | `tc_events_full.csv`, `figure_data/figS2_spatial.npz` | natural |
| S3 | `FigS3_radius_sensitivity.png` | `12_make_figS3.py` | 3×2 | `si_radius.csv`, `figure_data/figS3_spatial.npz`, `figure_data/_si_radius_cache/`, `tc_events_full.csv` | natural |
| S4 | `FigS4_track_sensitivity.png` | `13_make_figS4.py` | 2×2 | `si_ibtracs.csv` | natural |
| S5 | `FigS5_precip_sensitivity.png` | `14_make_figS5.py` | 3×2, lifecycle spanning the bottom row | `si_imerg.csv`, `figure_data/figS5_hourly.npz` | natural |
| S6 | `FigS6_RT_definition.png` | `15_make_figS6.py` | 2×2 | `tc_events_full.csv`, `figure_data/figS6_curves.csv`, `figS6_corr.csv` | natural |
| S7 | `FigS7_composite_comparison.png` | `16_make_figS7.py` | 2-col × 4-row: maps (a–c PEP−non-PEP, d–f Extreme−Short RT) + g/h | `composite_anomaly_wide.nc`, `composite_RTquartile_wide.nc`, `event_moisture_budget.csv`/`.npz` | natural |
| S8 | `FigS8_leave_one_out.png` | `17_make_figS8.py` | 2-col × 4-row: LOO strips + trend robustness + maps | `figure_data/figS8_loo.csv` (= Table S4), `figS8_trend.csv`, `figS8_spatial.npz` | natural |

* not redistributed — see `data/README.md` (`data/tc_prec_yearly/`, the two
  wide composite nc files, `fig1-d-body.png`, and the `figure_data/` caches are
  runtime or upstream products, or available on request).

## Width conventions

- **183 mm locked** (Fig 1/2/5/6): `_style.save(column="double")` fixes the
  Nature double-column width and verifies it after export.
- **Natural width** (Fig 3/4 and all SI): `natural_save()` keeps the tuned
  figsize; S-figure tight outputs are kept ≤ 183 mm.

## Smoothing disclosure

Displayed contour fields in Fig 4 and Fig S7 are Gaussian-smoothed for legibility
only (σ = 1.8 grid cells, NaN-aware `_style.smooth_field`); vectors are never
smoothed and every statistic uses unsmoothed fields.
