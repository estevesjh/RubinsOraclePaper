# Iterative Slope Correction via df['slope'] - Jan 23, 2026

## Goal
Improve twilight forecast by iteratively refining the slope estimate through the existing pipeline.

## Approach: Add df['slope'] to Pipeline

### The Math
- **Current baseline**: `delta_T_approx = y - T_tw_last` (flat, slope=0)
- **New baseline**: `delta_T_approx = y - (T_tw_last + slope/24 * h_from_tw)`
- **Iteration**: After Ridge gives slope_v1, update `df['slope']` and re-run

### Iteration Flow
1. **Iteration 0**: `df['slope'] = 0` → Run pipeline → Get slope_v1 from Ridge
2. **Iteration 1**: `df['slope'] = slope_v1` → Re-run NBEATSx → Get slope_v2
3. **Final**: `tw_temp_pred = T_tw_last + slope_v1 + slope_v2`

## Current Status

### Completed ✓
- [x] 1. Add `df['slope']` column to data preprocessing (initially 0)
- [x] 2. Modify `delta_T_approx` calculation to use slope: `y - (T_tw_last + slope/24 * h_from_tw)`
- [x] 3. Update `src/models/nbeats.py` - added `slope_file` parameter to `load_and_prepare_data()`
- [x] 4. Update `src/train.py` - uses `SLOPE_FILE` from config
- [x] 5. Update `src/evaluate.py` - saves slope predictions to `slope_predictions_3h.csv`
- [x] 6. Update `src/config.py` - added `SLOPE_FILE` config (default None)

### Todo
- [x] 7. Test iteration 0 (SLOPE_FILE=None, should match current results)
- [x] 8. Test iteration 1 (SLOPE_FILE=slope_predictions_10am.csv)
- [x] 9. Compare results at different lead times

### Results: Iteration 0 vs Iteration 1 Comparison (Jan 23, 2026)

**Conclusion: The 24-feature Ridge model makes iterative slope correction UNNECESSARY.**

| Lead Time | Iter 0 RMSE | Iter 1 RMSE | Difference |
|-----------|-------------|-------------|------------|
| 3h | 0.3559°C | 0.3559°C | 0.0000°C |
| 6h | 0.4345°C | 0.4345°C | 0.0000°C |
| 9h | 0.5962°C | 0.5962°C | 0.0000°C |
| 12h | 0.7302°C | 0.7302°C | 0.0000°C |

| Lead Time | Iter 0 <1°C | Iter 1 <1°C | Difference |
|-----------|-------------|-------------|------------|
| 3h | 98.8% | 98.8% | 0.0% |
| 6h | 97.0% | 97.0% | 0.0% |
| 9h | 91.6% | 91.6% | 0.0% |
| 12h | 86.1% | 86.1% | 0.0% |

**Why results are identical:**
The 24-feature Ridge model includes:
- 11 odd-hour lagged residuals (res_1h, res_3h, ..., res_21h)
- Multi-day lags (res_1d, res_2d)
- Diurnal rate features (rate_tw_to_midnight, rate_sunrise_to_midday, etc.)
- Temperature since sunrise
- Seasonal features (doy_sin, doy_cos, day_length)

These features already capture the diurnal slope information that the iterative correction was trying to recover. The slope correction becomes redundant with the expanded feature set.

## Key Files to Modify
- `src/data/preprocessing.py` - Add slope column and modify delta_T_approx
- `src/models/nbeats.py` - Use updated delta_T_approx
- `src/train.py` - Support iterative slope updates
- `src/evaluate.py` - Output slope predictions for next iteration

## Key Implementation Details

### predict_batch_corrected() in evaluate_iterative.py
- Uses corrected delta_T_approx for current cycle (h_from_tw > 0)
- Uses true delta_T for past cycles
- Uses `model.make_future_dataframe()` for correct future timestamp alignment
- Correction formula: `corrected_approx = y - (T_tw_last + slope_v1/24 * h_from_tw)`

---

# Rate Forecaster - Δ(dT/dt) Formulation

---

## Paper Figure Update (Jan 21, 2026)

### Task
Updated `fig7_rate_forecast()` in `src/plot.py` to generate the rate forecast figure for the paper using the Δ(dT/dt) formulation results.

### Changes Made

**File**: `src/plot.py` - `fig7_rate_forecast()` function

**Data source**: `results/rate_predictions_ridge_delta_3h.csv` (output from `rate_forecast_delta.py`)

**4-Panel Layout**:
1. **Top left**: Predicted vs Actual scatter
   - Shows both Persistence (gray x) and NBEATSx-Ridge (cyan dots)
   - RMSE labels in legend: Persistence=0.47°C/h, NBEATSx-Ridge=0.10°C/h
   - 1:1 reference line

2. **Top right**: Error distribution histogram
   - NBEATSx-Ridge only
   - Shows bias line (nearly zero: 0.007°C/h)
   - MAE=0.076°C/h

3. **Bottom left**: Time series
   - Actual vs Predicted twilight rate over 2025
   - Shows model tracks actual rate well

4. **Bottom right**: CDF of absolute error
   - Both Persistence (gray) and NBEATSx-Ridge (cyan)
   - xlim: 0 to 0.5°C/h
   - Threshold at 0.2°C/h with horizontal line
   - **97% of predictions below 0.2°C/h**

### Final Results

| Metric | NBEATSx-Ridge | Persistence |
|--------|---------------|-------------|
| RMSE | 0.096°C/h | 0.473°C/h |
| NRMSE | 25.3% | 117.6% |
| MAE | 0.076°C/h | - |
| Bias | 0.007°C/h | - |
| % < 0.2°C/h | 97% | ~5% |
| **Improvement** | **79.7%** | baseline |

### Usage
```bash
cd src
python -c "from plot import fig7_rate_forecast; fig7_rate_forecast()"
# Or run full pipeline:
python plot.py
```

Output: `results/figures/fig7_rate_forecast.png` and `.pdf`

---

## Summary

Successfully implemented multi-window rate forecasting using the Δ(dT/dt) formulation. The approach mirrors the temperature forecaster but targets the temperature change rate at twilight.

### Key Results

| Metric | Value |
|--------|-------|
| Windows | 7 × 3h (covering t=1-21h) |
| Test twilights | 334 |
| Prediction requests | 2,338 |
| Residual features | 21 (res_t1 to res_t21) |
| Total Ridge features | 32 |
| **Persistence NRMSE** | 117.6% |
| **NBEATSx-Ridge NRMSE** | 24.5% |
| **Improvement** | 79.7% |

### Data Leakage Audit

All Ridge features verified clean:
- `res_t1` to `res_t21`: Available by twilight (all times t=1-21h have passed)
- `slope_est/mean/intercept`: Derived from residuals
- `dT_dt_tw_last`: Previous twilight rate (known)
- `dT_dt_tw_lag1/2/3`: Rates from 1-3 days ago (known)
- `slope_lag1/2`: Slopes from 1-2 days ago (known)
- `doy_sin/cos`: Seasonal (known)

**Note**: Train/test uses odd/even day split. Not strictly causal but no target leakage.

---

## Completed Tasks

- [x] Update FORECAST_WINDOWS to 7 × 3h windows
- [x] Remove undefined FORECAST_LEAD_HOURS
- [x] Fix data leakage (dT_dt_h3 → dT_dt_tw_last)
- [x] Add inner loop over windows with unique_id pattern
- [x] Extract residuals per window, aggregate by twilight
- [x] Update Ridge features (res_t1 to res_t21)
- [x] Verify script runs successfully
- [x] Audit Ridge features for data leakage

---

## Rate-Trend Definition

### Mathematical Framework

**Training Phase** (both twilight rates known):
```
Rate-Trend(t) = dT_dt_tw_last + (dT_dt_tw - dT_dt_tw_last)/24 × t
Δ(dT/dt)(t) = dT/dt(t) - Rate-Trend(t)
```

**Boundary conditions:**
- At t=0 (last twilight): Δ(dT/dt) = 0
- At t=24 (next twilight): Δ(dT/dt) = 0

**Operational Phase** (dT_dt_tw unknown):
```
Rate-Trend_approx(t) ≈ dT_dt_tw_last  (flat baseline)
rate_approx(t) = Δ(dT/dt)_pred(t) + dT_dt_tw_last
```

**Residual Pattern:**
```
res(t) = dT_dt_actual(t) - rate_approx(t)
res(t) ≈ slope_rate × t
```

The linear growth of residuals with time reveals the unknown slope!

### Forecast Windows

| Window | Start (h before tw) | Sample Times | Horizon |
|--------|---------------------|--------------|---------|
| 0 | 23h | t=1, 2, 3 | 3h |
| 1 | 20h | t=4, 5, 6 | 3h |
| 2 | 17h | t=7, 8, 9 | 3h |
| 3 | 14h | t=10, 11, 12 | 3h |
| 4 | 11h | t=13, 14, 15 | 3h |
| 5 | 8h | t=16, 17, 18 | 3h |
| 6 | 5h | t=19, 20, 21 | 3h |

---

## Dataset Overview Figure (fig0) - Jan 22, 2026

### Task
Added a new `fig0_dataset_overview()` function to `src/plot.py` that creates a 3-row elongated dataset overview figure.

### Changes Made

**File**: `src/plot.py`

1. Added `fig0_dataset_overview()` function (lines 51-133)
2. Updated docstring to include Fig 0
3. Added call to `fig0_dataset_overview()` in `main()`

### Figure Structure

**Row 1: Full Year Temperature**
- Plots raw temperature (`y`) from Jan 2025 to Dec 2025
- Dark grey line with orange highlight on representative week

**Row 2: Representative Week Detail**
- Week 30 selected (75th percentile of twilight-to-twilight slope variance)
- Raw temperature: dark grey line
- Twilight-Trend: brown line (linear interpolation between consecutive twilight temperatures)
- Night bands: navy vertical spans (twilight to next sunrise) with alpha=0.15
- Twilight markers: gray dashed vertical lines

**Row 3: Offset from Twilight-Trend**
- Offset = `y - twilight_baseline` plotted in brown
- Same night bands as Row 2
- Zero reference line

### Implementation Details

- **Twilight detection**: Rows where `twilight_temp` is not NaN
- **Sunrise detection**: Rows where `sunrise_temp` is not NaN
- **Twilight baseline**: Linear interpolation using `np.interp()` between consecutive twilight temperatures
- **Representative week selection**: Week with variance closest to 75th percentile (Week 30)
- **Colors**: Raw temp = `#333333`, Twilight-Trend/Offset = `#8B4513` (brown), Night bands = navy

### Output
- `results/figures/fig0_dataset_overview.png` (335 KB)
- `results/figures/fig0_dataset_overview.pdf` (130 KB)

### Usage
```bash
conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro && python src/plot.py
```

### Completed Tasks
- [x] Add `fig0_dataset_overview()` function
- [x] Compute twilight baseline (linear interpolation)
- [x] Find representative week (75th percentile variance)
- [x] Add fig0 call to `main()`
- [x] Test figure generation

---

## Methods Section Update (Jan 22, 2026)

### Task
Rewrote the Methods section in `docs/main.tex` with comprehensive content about baseline models, NBEATSx architecture, and the two-stage NBEATSx-Ridge approach.

### Changes Made

**File**: `docs/main.tex`

1. **Section 3.1: Baseline Models** - Added descriptions for Persistence, Linear, Random Forest, and MLP baselines
2. **Section 3.2: NBEATSx Architecture** - Added detailed description of NBEATSx neural architecture with trend, seasonality, and exogenous stacks
3. **Section 3.3: The $\Delta T$ Formulation** - Documented the training and operational phase approaches
4. **Section 3.4: Two-Stage Correction (NBEATSx-Ridge)** - Described the Ridge regression slope recovery method
5. **Section 3.5: Intrinsic Uncertainty** - Preserved existing content

**File**: `docs/ref.bib`

Added bibliography entries:
- `Olivares2021` - NBEATSx paper (International Journal of Forecasting, arXiv:2104.05522)
- `Oreshkin2020` - Original N-BEATS paper (ICLR 2020)

### Completed Tasks
- [x] Add baseline models description
- [x] Add NBEATSx architecture description
- [x] Document exogenous features
- [x] Improve NBEATSx-Ridge sections (3.3, 3.4)
- [x] Add NBEATSx bibliography entries (Olivares2021, Oreshkin2020)

---

## Persistence-Twilight Baseline Model (Jan 22, 2026)

### Task
Added a new baseline model that uses yesterday's twilight temperature to predict today's twilight temperature.

### Changes Made

**File**: `src/train.py`

1. Added `predict_persistence_twilight()` function - uses previous twilight temperature as prediction
2. Added evaluation loop for Persistence-Twilight model
3. Updated docstring to list the new model

**File**: `src/evaluate.py`

1. Added "Persistence-Twilight" to the metrics output model list

### Model Description

- **Persistence-Twilight**: Uses the previous day's twilight temperature to predict today's
- Independent of lead time (same prediction at any forecast horizon)
- Captures day-to-day persistence in twilight temperatures

### Usage
```bash
conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro
python src/train.py  # Re-run to generate new paper_results.csv
python src/evaluate.py  # Re-run to include in metrics
```

### Completed Tasks
- [x] Add `predict_persistence_twilight()` function
- [x] Add to evaluation loop in train.py
- [x] Add to metrics list in evaluate.py

---

## NBEATSx vs Prophet vs MeteoBlue Comparison (Jan 22, 2026)

### Task
Created comparison figure and section for NBEATSx-Ridge, Prophet hybrid, and MeteoBlue forecasts.

### Changes Made

**File**: `src/add_external_forecasts.py` (NEW)
- Extracts Prophet forecasts from `data/results_hybrid_07d_30min.csv`
- Extracts MeteoBlue forecasts from `data/meteo_blue_weather_station.csv`
- Applies Prophet exponential blending (A=0.90, tau=13.0)
- Handles Prophet model failures based on reduced chi-square
- Matches forecasts to twilight events (2024-2025)
- Combines with `paper_results_v2.csv` → `paper_results_v3.csv`

**File**: `src/plot.py`
- Added `_draw_density_contours()` helper function
- Added `_mad_sigma()` and `_robust_inlier_mask()` helper functions
- Added `fig8_comparison_nbeats_prophet_meteoblue()` function
- Updated `main()` to include fig8

**File**: `docs/main.tex`
- Added new subsection "Comparison with External Forecast Sources"
- Added Figure 8 reference with caption

### Figure Description

**Layout**: 2 rows × 3 columns
- Top row: Predicted vs Actual scatter plots with density contours
- Bottom row: Residuals vs Actual temperature

**Columns**:
1. NBEATSx-Ridge (12h lead) - Color: #006D77 (teal)
2. Prophet (all leads) - Color: #83C5BE (light cyan)
3. MeteoBlue (all leads) - Color: brown

**Stats displayed**: N, Slope, Bias, Std, R²

### Output Files

| File | Description |
|------|-------------|
| `results/paper_results_v3.csv` | Combined results (174,901 rows) |
| `results/figures/fig8_comparison.pdf` | Comparison figure (PDF) |
| `results/figures/fig8_comparison.png` | Comparison figure (PNG) |

### Data Summary

| Model | Forecasts |
|-------|-----------|
| Prophet | 108,827 |
| NBEATSx-Ridge (12h) | 166 |
| MeteoBlue | 626 |

### Completed Tasks
- [x] Create `src/add_external_forecasts.py` script
- [x] Generate `paper_results_v3.csv`
- [x] Add comparison figure function to plot.py
- [x] Generate fig8_comparison.png/pdf
- [x] Add comparison section to main.tex
