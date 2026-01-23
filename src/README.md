# Twilight Temperature Forecasting - Source Code

This directory contains the source code for reproducing the NBEATSx-Ridge twilight temperature forecasting paper.

## Required Data Files

The following data files must be placed in the `data/` directory:

| File | Size | Description | Source |
|------|------|-------------|--------|
| `temp_history_all_dec2025_sunrise_sunset.csv` | ~10 MB | Temperature history with twilight/sunrise markers | Rubin Observatory EFD |
| `results_hybrid_07d_30min.csv` | ~1.5 GB | Prophet hybrid forecasts (optional) | Prophet forecasting system |
| `meteo_blue_weather_station.csv` | ~5 MB | MeteoBlue commercial forecasts (optional) | MeteoBlue API |

**Note:** The Prophet and MeteoBlue files are required to generate `paper_results_v3.csv` (via `add_external_forecasts.py`). Without these files, you can still run the main pipeline but `plot.py` will use `paper_results_v2.csv` instead.

### Data Format

**temp_history_all_dec2025_sunrise_sunset.csv** columns:
- `ds`: Timestamp (15-min intervals)
- `y`: Temperature (°C)
- `twilight_temp`: Temperature at twilight (NaN for non-twilight rows)
- `sunrise_temp`: Temperature at sunrise (NaN for non-sunrise rows)

## Environment Setup

```bash
conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro
```

## Pipeline Overview

```
train.py → evaluate.py → plot.py
```

## Step-by-Step Reproduction

### 1. Train Models and Generate Base Results

```bash
python src/train.py
```

**Outputs:**
- `results/paper_results.csv` - Predictions from all models (Persistence, Linear, RF, MLP, NBEATSx)
- `results/twilight_offset_predictions.csv` - Detailed NBEATSx predictions for Ridge correction

**Cached models:**
- `results/paper_model_cache/sklearn_hourly_temps_with_trend/` - Linear, RF, MLP models
- `results/paper_nbeats_cache/` - NBEATSx models

### 2. Apply Ridge Correction and Evaluate

```bash
python src/evaluate.py
```

**Outputs:**
- `results/paper_results_v2.csv` - Results with NBEATSx-Ridge corrections (24-feature model)
- `results/slope_predictions_10am.csv` - Slope predictions for iterative correction (optional)

### 3. Add External Forecasts (Required for full comparison)

```bash
python src/add_external_forecasts.py
```

**Requires:**
- `data/results_hybrid_07d_30min.csv` (Prophet forecasts)
- `data/meteo_blue_weather_station.csv` (MeteoBlue forecasts)

**Outputs:**
- `results/paper_results_v3.csv` - Combined results including Prophet and MeteoBlue forecasts

**Note:** `plot.py` uses `paper_results_v3.csv` for figures that compare NBEATSx-Ridge with Prophet and MeteoBlue (fig2, fig5, fig8). If this file doesn't exist, you'll need to modify `plot.py` to use `paper_results_v2.csv`.

### 4. Generate Figures

```bash
python src/plot.py
```

**Outputs:**
- `figures/fig0_dataset_overview.pdf` - Dataset overview
- `figures/fig1_residuals_vs_time_to_tw.pdf` - Residual analysis
- `figures/fig2_rmse_heatmap.pdf` - RMSE by model and lead time
- `figures/fig3_rmse_vs_lead_time.pdf` - RMSE vs lead time
- `figures/fig4_error_histograms_3h.pdf` - Error distributions
- `figures/fig5_cdf_error.pdf` - CDF of absolute errors
- `figures/fig6_seasonal_trend_analysis.pdf` - Seasonal analysis
- `figures/fig7_rate_forecast.pdf` - Rate forecasting results
- `figures/fig8_comparison.pdf` - Model comparison

### 5. Generate Rate Forecasts (Optional)

```bash
python src/rate_forecast_delta.py
```

## Quick Reproduction

To regenerate all paper results:

```bash
cd /Users/esteves/Documents/github/RubinsOraclePaper
conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro

python src/train.py
python src/evaluate.py
python src/plot.py
```

## File Descriptions

| File | Description |
|------|-------------|
| `config.py` | Central configuration (paths, hyperparameters, feature lists) |
| `train.py` | Train baseline models and NBEATSx |
| `evaluate.py` | Apply 24-feature Ridge correction, compute metrics |
| `plot.py` | Generate all paper figures |
| `add_external_forecasts.py` | Add Prophet and MeteoBlue forecasts |
| `rate_forecast_delta.py` | Temperature rate forecasting |
| `evaluate_iterative.py` | Iterative slope correction (not needed with 24-feature model) |
| `data/loader.py` | Data loading utilities |
| `data/preprocessing.py` | Feature engineering |
| `models/nbeats.py` | NBEATSx model and prediction utilities |

## Key Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEST_START_DATE` | 2025-01-01 | Start of test period |
| `LEAD_TIMES_HOURS` | 0.5 to 12h | Forecast lead times |
| `NBEATS_HORIZON` | 48 steps | 12h at 15-min resolution |
| `NBEATS_INPUT_SIZE` | 96 steps | 24h lookback |
| `RIDGE_ALPHA` | 1.0 | Ridge regularization |

## Results Summary

At 3-hour lead time (dome opening):
- **NBEATSx-Ridge**: 0.36°C RMSE, 99% < 1°C
- **NBEATSx-Oracle**: 0.79°C RMSE, 77% < 1°C
- **Persistence**: 2.42°C RMSE, 17% < 1°C
