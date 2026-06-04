# Forecast Pipeline

## Layout

```
forecast/
  config.py            # paths, split date, grid/model constants
  run.py               # data load, solar grid, FeatureBuilder, NBEATSx training
  feature_sweep.py     # compute_all_features() — engineered feature library (used by production)
  exp_humidity.py      # merge_humidity() — humidity/wind + directional-wind + RH-tercile features (used by production)
  run_final_model.py   # THE PAPER MODEL: lead-gated two-stage NBEATSx + Ridge
  traj_loss.py         # TrajHuberLoss — derivative-aware loss (lam=0 == stock Huber; used by production)
  build_final.py       # baselines + blend + external forecasts -> paper_results_final.csv
  calibrate_meteoblue.py # per-solar-bin MeteoBlue calibration table

  optimization/        # hyperparameter / data-volume sweeps (not in the critical path)
  analysis/            # one-off experiments, diagnostics, alternate models (wandering scripts)
```

`config.py`, `run.py`, `feature_sweep.py`, `exp_humidity.py`, and `traj_loss.py`
are imported by everything else (the last two are misleadingly named but are
production dependencies). Scripts in `optimization/` and `analysis/` add a
one-line `sys.path` shim so they still import these from `forecast/` after the move.

## Quick start (produce the paper results + figures)

```bash
# 1. Train the production model -> results/paper_results_diff.csv
uv run --no-sync python forecast/run_final_model.py

# 2. Baselines + Linear blend + Prophet/MeteoBlue -> results/paper_results_final.csv
uv run --no-sync python forecast/build_final.py

# 3. Regenerate all figures + Table 1
uv run --no-sync python src/plot.py
```

## What the core steps do

### `forecast/run_final_model.py` (production model)
- Two-stage: NBEATSx (direct-T, width 16, 700 steps, pre-2025 train) + per-lead Ridge.
- **Lead-gated features**: short leads (<5 h) use the paper feature set; long leads
  (>=5 h) add multi-day trend + RH terciles + directional-wind features that cut
  Winter/Spring morning RMSE (see `docs/winter_spring_plan.md`).
- Output: `results/paper_results_diff.csv` (model name `NBEATSx-Diff`).

### `forecast/build_final.py`
- Reads `paper_results_diff.csv`; trains Linear/RF/MLP baselines + Persistence.
- Blends NBEATSx + Linear (per-lead optimal alpha) -> `NBEATSx-Ridge`.
- Adds Prophet-BMA and MeteoBlue; writes `results/paper_results_final.csv`.

### `src/plot.py`
- Reads `paper_results_final.csv`; writes fig0/2/3/5/6/7/8 + Table 1 to `figures/`.

## optimization/  (sweeps)
- `sweep_maxsteps.py`, `sweep_mlp.py`, `sweep_samplesize.py` — NBEATSx sweeps.
- `plot_maxsteps.py`, `plot_mlp.py`, `plot_samplesize.py` — their plots.

## analysis/  (experiments & diagnostics — not required to build the paper)
- `exp_winter_spring.py` — Phase 1+2 (trend features, input_size sweep).
- `exp_humidity.py`* / `exp_hardday_clf.py` — humidity/wind features; hard-day classifier.
- `exp_capacity.py` — NBEATSx capacity/regularization & train-test gap study.
- `exp_traj_loss.py` — derivative-aware (slope) loss sweep.
- `exp_pca_sweep.py` — PCA decorrelation test.
- `exp_*_exog.py`, `exp_nwp_variant.py`, `exp_persist_*`, `exp_ridge_*` — feature/exog/blend experiments.
- `investigate.py`, `analyze_sweep_corr.py`, `eval_real_hours.py` — diagnostics.
- `ridge_full.py`, `ridge_twilight.py`, `run_96.py`, `run_nosolar.py`, `solar_grid.py` — alternate models / earlier variants.

(*`exp_humidity.py` lives in `forecast/` — production imports `merge_humidity` from it.)

## Experiment log

Each row: the question asked, the verdict, and whether it changed the production
model. Full write-ups in `docs/winter_spring_plan.md` (sections cited).

| Experiment (script) | Question | Verdict | In production? |
|---|---|---|---|
| Seasonal diagnosis | Why are Winter/Spring worse? | Day-to-day synoptic variability + trend-lag (r≈0.5–0.6), not cold fronts/`bad_weather`. ~15–19% of 9h error is reducible. (§2, §3.7) | — (diagnosis) |
| `exp_winter_spring.py` Phase 2 | Does a longer NBEATSx lookback help? | No — WS9 rises monotonically with input_size. Keep `input_size=48`. (§3.5) | No |
| `exp_winter_spring.py` Phase 1 | Multi-day trend features in Ridge? | Small win at long lead (~2% WS9), neutral at 3h. (§3.5) | **Yes** (long-lead) |
| `exp_humidity.py` | Humidity + wind as causal features? | Help at 9h (Winter −6.7%, Spring −5.5%), hurt at 3h → lead-gate. Coverage confound fixed via climatology imputation. (§3.6) | **Yes** (long-lead) |
| directional wind | Decompose wind on prevailing axis? | Wins Spring (9h −7.9%); loses Winter. (§3.8) | **Yes** |
| RH terciles | Nonlinear (binned) humidity? | Best variant: binned RH halves the 3h penalty AND improves 9h Winter (−8.0%). (§3.10) | **Yes** |
| wind tendency | Wind trend-lag (`dws_6h`, dir-variability)? | Real error correlate (Spring r=−0.42) but redundant with existing features; no gain. (§3.12) | No (available) |
| `exp_hardday_clf.py` | Can we classify hard days at issue time? | Yes, AUC 0.75/0.77 (Winter/Spring) from temperature volatility; humidity/wind don't add. (§3.11) | No (flag candidate) |
| `exp_pca_sweep.py` | PCA-decorrelate the collinear features? | No — Ridge L2 already handles collinearity; truncation deletes signal. (§3.9) | No |
| `exp_capacity.py` | Why does Ridge beat NBEATSx? | Overfitting: bigger nets lower train but raise test RMSE (gap up to +0.6°C @9h); dropout only recovers the small net. Target is near-linear at ~400 twilights. | No (keep w16/b1) |
| `exp_traj_loss.py` | Penalize trajectory slope (for dT/dt control)? | Gated by `LAMBDA` in `run_final_model.py` (default 0 = stock Huber). | Hook (lam=0) |
| `exp_nwp_variant.py` | MeteoBlue NWP as exog/feature? | Degrades or no gain unconditionally; de-scoped (MB data ends Aug-2025). | No |

## Key settings (forecast/config.py)
- `SOLAR_GRID_STEP = 1/48` (30-min cadence); `HALFDAY_LAG_STEPS = 24` (12 h differencing lag).
- `NBEATS_INPUT_SIZE = 48` (1 solar day); `NBEATS_HORIZON = 26`; `NBEATS_MAX_STEPS = 700`.
- `TEST_START_DATE = 2025-01-01`; forecast target alt_sun = -20 deg (astro. twilight).

## Notes
- Local runs use the `uv` env (`uv run --no-sync python ...`); FeatureBuilder
  (`twilight` package) is found via the candidate paths in `run.py`.
- `analysis/run_nosolar.py` and `analysis/investigate.py` hardcode a slacd path
  for `twilight` and only run on slacd.
