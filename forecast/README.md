# Forecast Pipeline

## Quick Start

```bash
cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper

# Step 1: Train NBEATSx-Diff and evaluate at alt_sun=-25 target
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/run.py

# Step 2: Build final results (baselines + blend + external forecasts + plots)
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/build_final.py
```

## What each step does

### `forecast/run.py`
- Loads data from `rubin-twilight-forecast/data/temp_history_jan2026_with_weather.csv`
- Builds solar grid (48 steps/day) via FeatureBuilder (sun_alt_midpoint=-25)
- Adds features: lags, trend, DTR, solar anchors
- Trains NBEATSx on D = T(tn) - T(tn-24 steps) with early stopping
- Evaluates at all lead times (0.5–12h)
- Outputs: `results/paper_results_diff.csv`
- Model cached at: `results/nbeats_diff_cache/NBEATSx_diff_v3/`

### `forecast/build_final.py`
- Reads `paper_results_diff.csv`
- Trains baselines (Linear/Ridge, MLP, RandomForest) on solar grid features
- Blends NBEATSx + Linear (per-lead optimal alpha)
- Adds Prophet-BMA and MeteoBlue external forecasts
- Writes: `results/paper_results_final.csv`
- Runs `src/plot.py` to regenerate all figures

### `src/plot.py`
- Reads `results/paper_results_final.csv`
- Generates: fig0, fig2, fig5, fig6, fig7, fig8, table1
- Output: `figures/`

## Other scripts

- `forecast/feature_sweep.py` — forward selection to find optimal feature set
- `forecast/investigate.py` — error analysis (persistence comparison, outliers)
- `forecast/eval_real_hours.py` — evaluate with real-time lead hours
- `forecast/ridge_full.py` — Ridge correction on all twilights

## Key settings (forecast/config.py)

- `SOLAR_GRID_STEP = 1/48` (30-min cadence)
- `HALFDAY_LAG_STEPS = 24` (12h differencing lag)
- `NBEATS_HORIZON = 26` (covers up to 12h lead)
- `NBEATS_INPUT_SIZE = 48` (1 solar day lookback)
- `NBEATS_MAX_STEPS = 500`
- Target: alt_sun = -25° (sun setting), ~1.8h after sunset

## To retrain from scratch

Delete the cache and run:
```bash
rm -rf results/nbeats_diff_cache/NBEATSx_diff_v3
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/run.py
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/build_final.py
```

## For feature experiments

Request a compute node (needed for long jobs):
```bash
srun -A rubin:developers -p roma -q normal --cpus-per-task=32 --mem=64G --time=02:00:00 --pty bash
```

Then run:
```bash
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/feature_sweep.py
```
