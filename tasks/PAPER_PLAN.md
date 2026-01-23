# Twilight Temperature Forecasting Paper

## Summary

Two-stage forecasting system for predicting twilight temperature at Rubin Observatory using NBEATSx + Ridge correction. Achieves **0.42°C RMSE** at 3h lead time with **98% of predictions within 1°C**.

### Ridge Feature Optimization (Jan 2026)
Reduced from 29 to **20 features** while improving performance:
- RMSE: 0.427 → **0.420°C** (improved)
- P95: 0.82°C (same)
- Max|Bias|: 0.099 → **0.098°C** (improved)
- Features: 29 → **20** (31% reduction)

---

## Project Structure

```
src/
├── config.py                      # Central configuration (all constants)
├── train.py                       # Main training pipeline
├── evaluate.py                    # Ridge correction + evaluation
├── plot.py                        # Figure generation
├── data/                          # Data preparation & loading
│   ├── loader.py                  # Data loading
│   ├── preprocessing.py           # Feature engineering
│   ├── fill_gaps.py               # Gap filling with Prophet
│   ├── sun_utils.py               # Astropy sun calculations
│   └── compute_sunrise_sunset.py  # Compute sun events
└── models/                        # Model definitions
    └── nbeats.py                  # NBEATSx training/inference
```

---

## Data Pipeline

```
data/temp_history_all_dec2025.csv                    [RAW - 9.3MB]
    ↓ python -m data.fill_gaps
data/temp_history_all_dec2025_filled_gaps.csv        [8.2MB]
    ↓ python -m data.compute_sunrise_sunset
data/temp_history_all_dec2025_sunrise_sunset.csv     [10MB]
    ↓ python train.py
results/paper_results.csv + twilight_offset_predictions.csv
    ↓ python evaluate.py
results/paper_results_v2.csv
    ↓ python plot.py
results/figures/*.png
    ↓
docs/twilight_forecast_report.tex → PDF
```

---

## Configuration (config.py)

All constants centralized in `src/config.py`:

```python
# Prediction Schedule
PRED_TIMES = [(f"{h:.1f}h", h) for h in np.arange(0.5, 24.5, 0.5)]  # 48 times, 30-min intervals

# Lead times for evaluation
LEAD_TIMES_HOURS = list(np.arange(0.5, 12.5, 0.5))  # 0.5h to 12h

# NBEATSx hyperparameters
NBEATS_HORIZON = 48       # 12 hours at 15-min
NBEATS_INPUT_SIZE = 96    # 24 hours at 15-min
NBEATS_MAX_STEPS = 500
NBEATS_CONFIG = {
    "activation": "SELU",
    "scaler_type": "robust",
    "stack_types": ["trend", "seasonality", "exogenous"],
    "mlp_units": 3 * [[32, 32]],
    "n_blocks": [1, 1, 1],
    "learning_rate": 0.01,
}

# Features
NBEATS_HIST_EXOG = [
    "temp_raw", "temp_last_sunrise", "trend_2h", "temp_trend_3d",
    "rate_sunrise_to_midday", "rate_midday_to_twilight",
    "rate_twilight_to_midnight", "rate_midnight_to_sunrise"
]
NBEATS_FUTR_EXOG = ["twilight_cos"]
```

---

## Data Overview

- **Location**: Cerro Pachón, Chile (30°14'40" S, 70°44'58" W)
- **Period**: October 2023 – December 2025 (~771 days)
- **Resolution**: 15-minute intervals
- **Train/Test Split**: Even days (train) / Odd days (test) from Jan 1, 2025
- **Test twilights**: 384 events
- **Prediction frequency**: Every 30 minutes (48 predictions per twilight cycle)

---

## Mathematical Framework

### Training Phase (Full Information)

With both twilight temperatures known:
```
Twilight-Trend(h) = T_tw_last + (T_tw - T_tw_last) × h / 24
ΔT(h) = T(h) - Twilight-Trend(h)
```

**Boundary conditions:** ΔT = 0 at h=0 and h=24

### Operational Phase (Approximation)

Future twilight unknown:
```
Twilight-Trend_approx = T_tw_last
T_approx(h) = ΔT_pred(h) + T_tw_last
```

### Ridge Slope Recovery

The residual reveals the unknown slope:
```
residual(h) = T_actual(h) - T_approx(h) ≈ slope × h
slope = (T_tw - T_tw_last) / 24
T_tw_pred = slope_Ridge × 24 + T_tw_last
```

**Ridge Features (20 total, optimized Jan 2026):**

*Core features (9):*
- `res_tw` - current residual (correlation: +0.98 with target)
- `rate_twilight_to_midnight` - evening cooling rate (+0.59)
- `rate_sunrise_to_midday` - morning warming rate (+0.56)
- `rate_midnight_to_sunrise` - overnight rate (+0.65)
- `day_length` - hours from sunrise to twilight
- `res_tw_sq_temp` - res_tw² / (|temp_actual| + 5) - bias correction
- `twilight_cos` - cyclical encoding of twilight hour
- `doy_sin`, `doy_cos` - day-of-year seasonal encoding

*Lagged residuals (11 odd lags):*
- `res_tw_1, res_tw_3, res_tw_5, ..., res_tw_21` - capture diurnal autocorrelation

*Removed features (no longer needed):*
- `hour_to_tw`, `hour_to_tw_cos` - constant at 3h lead time
- `temp_actual`, `T_tw_last`, `temp_last_sunrise` - hurt RMSE
- `trend_temp_3d`, `trend_2h` - hurt RMSE
- `res_tw_23, res_tw_25, res_tw_27` - marginal benefit, increase bias

### Intrinsic Uncertainty Analysis

The intrinsic uncertainty was measured by combining in quadrature:
1. **Instrumental noise**: median(tempMax - tempMin) / d_n, where d_n ≈ 4.0 for n=30 samples (Gaussian)
2. **Temperature variation rate**: std of temperature change over 15-min intervals

| Lead Time | Intrinsic Uncertainty | Model RMSE |
|-----------|----------------------|------------|
| 0.5h | 0.35°C | 0.44°C |
| 1.0h | 0.41°C | 0.44°C |
| **3.0h** | **0.47°C** | **0.44°C** |
| 6.0h | 0.42°C | 0.49°C |

**Key insight**: At 3h lead time, the model RMSE (0.44°C) is below the intrinsic uncertainty (0.47°C), indicating near-optimal performance.

### Operational Cutoff at 3 Hours

Operations begin 3 hours before twilight, so the Ridge correction is applied at this lead time. The model achieves **0.44°C RMSE** at the operational cutoff, which is below the intrinsic uncertainty (0.47°C), indicating near-optimal performance when the forecast is operationally relevant.

---

## Results

### RMSE by Model and Lead Time (°C)

| Model | 0.5h | 1h | 2h | 3h | 4h | 5h | 6h |
|-------|------|-----|-----|-----|-----|-----|-----|
| Persistence | 0.76 | 1.26 | 2.09 | 2.56 | 2.78 | 2.79 | 2.63 |
| Linear | 0.77 | 0.77 | 0.99 | 1.16 | 1.21 | 1.30 | 1.44 |
| RandomForest | 0.77 | 0.77 | 0.96 | 1.12 | 1.19 | 1.29 | 1.39 |
| MLP | 0.76 | 0.76 | 1.09 | 1.02 | 1.30 | 1.31 | 1.41 |
| NBEATSx-Oracle | 0.74 | 0.85 | 0.82 | 0.70 | 0.78 | 0.82 | 0.82 |
| **NBEATSx-Ridge** | **0.42** | **0.42** | **0.42** | **0.42** | **0.49** | **0.53** | **0.55** |

*Correction applies up to beginning of operations: 0.42°C for all lead times ≤ 3h*

### Metrics at 3h Lead Time

| Model | RMSE | MAE | Bias | % < 1°C |
|-------|------|-----|------|---------|
| Persistence | 2.56 | 2.30 | −2.24 | 16% |
| NBEATSx-Oracle | 0.83 | 0.56 | −0.50 | 83% |
| **NBEATSx-Ridge** | **0.42** | **0.34** | **−0.02** | **98%** |

### Key Findings

1. **NBEATSx-Ridge achieves noise floor** (0.42°C) - theoretically optimal at 3h
2. **Correction up to ops** - RMSE stays at 0.42°C for all lead times 0.5h-3h
3. **Minimal bias** - Only −0.02°C (near zero), Max|Bias| = 0.098°C across seasons
4. **98% within 1°C** - exceeds operational requirements
5. **84% improvement** over Persistence baseline
6. **31% fewer features** - 20 features vs previous 29

---

## Commands

```bash
# Activate environment
conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro

# Check configuration
cd src
python config.py

# Data preparation (if needed)
python -m data.fill_gaps
python -m data.compute_sunrise_sunset

# Run full pipeline
python train.py      # Train models, generate predictions (slow)
python evaluate.py   # Ridge correction (fast)
python plot.py       # Generate figures

# Compile paper
cd ../docs && pdflatex twilight_forecast_report.tex
```

---

## Model Computation Details

### NBEATSx-Oracle (train.py)

Uses **actual twilight temperature** (future information) - for comparison only:

```python
# Oracle: uses actual twilight temp as baseline endpoint
pred_temp_oracle = delta_T_pred + tw_temp  # tw_temp is ACTUAL (future info!)
```

### NBEATSx (Operational) (train.py)

Uses **T_tw_last approximation** (no future information):

```python
# Operational: uses previous twilight temp as baseline
pred_temp = delta_T_pred + T_tw_last  # T_tw_last is KNOWN
```

### NBEATSx-Ridge (evaluate.py)

**Two-stage correction** - NBEATSx approximates current temperature (nowcast), Ridge corrects to predict twilight:

```python
# Stage 1: NBEATSx produces T_approx (nowcast at current_time)
# Note: pred_horizon = 0, so NBEATSx is approximating CURRENT temp, not forecasting
T_approx = delta_T_pred + T_tw_last  # NBEATSx approximation at current_time
res_tw = actual_temp - T_approx       # Residual: how wrong is NBEATSx now?

# Stage 2: Ridge predicts twilight slope from current residual pattern
# Target: tw_slope = tw_temp - T_tw_last
feature_cols = [
    "res_tw",                    # Current residual (correlation +0.98)
    "rate_twilight_to_midnight", # Evening cooling rate (+0.59)
    "rate_sunrise_to_midday",    # Morning warming rate (+0.56)
    "rate_midnight_to_sunrise",  # Overnight rate (+0.65)
    "day_length",                # Seasonal daylight
    "res_tw_sq_temp",            # res_tw² / (|temp| + 5) - bias correction
    "twilight_cos",              # Twilight hour encoding
    "doy_sin", "doy_cos",        # Seasonal encoding
] + odd_lags_1_to_21             # res_tw_1, res_tw_3, ..., res_tw_21

# Train Ridge
model = Ridge(alpha=RIDGE_ALPHA)
model.fit(X_train_s, y_train)  # y_train = tw_slope

# Predict twilight temperature
pred_slope = model.predict(X_test_s)
pred_tw_temp = T_tw_last + pred_slope
```

**Key insight**: NBEATSx is used as a **temperature approximator** (nowcast), NOT a forecaster. The residual `res_tw` captures how the approximation deviates from reality at the current moment. Ridge then uses this residual pattern (plus rates and seasonal features) to predict the slope to twilight. This design is intentional: NBEATSx predictions with pred_horizon > 0 are less accurate than nowcast + Ridge slope prediction.

---

## Temperature Rate (dT/dt) Forecasting

### Goal
Forecast the temperature change rate at twilight for HVAC pre-conditioning.

### Approach: Δ(dT/dt) Formulation (Jan 2026)

Mirrors the temperature forecaster approach but targets rate instead of temperature.

#### Rate-Trend Definition

**Training Phase** (both twilight rates known):
```
Rate-Trend(t) = dT_dt_tw_last + (dT_dt_tw - dT_dt_tw_last)/24 × t
Δ(dT/dt)(t) = dT/dt(t) - Rate-Trend(t)
```

**Boundary conditions:**
- At t=0 (last twilight): Δ(dT/dt) = 0 ✓
- At t=24 (next twilight): Δ(dT/dt) = 0 ✓

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

#### Multi-Window Forecasting

7 windows × 3h horizon covering t=1-21h after last twilight:

| Window | Start | Sample Times |
|--------|-------|--------------|
| 0 | 23h before tw | t=1, 2, 3 |
| 1 | 20h before tw | t=4, 5, 6 |
| 2 | 17h before tw | t=7, 8, 9 |
| 3 | 14h before tw | t=10, 11, 12 |
| 4 | 11h before tw | t=13, 14, 15 |
| 5 | 8h before tw | t=16, 17, 18 |
| 6 | 5h before tw | t=19, 20, 21 |

### Results (Jan 2026)

| Metric | Value |
|--------|-------|
| Test twilights | 334 |
| Prediction requests | 2,338 |
| Ridge features | 32 (21 residuals + 11 other) |
| **Persistence NRMSE** | 117.6% |
| **NBEATSx-Ridge NRMSE** | 24.5% |
| **Improvement** | **79.7%** |

**Ridge Features (32 total):**
- `res_t1` to `res_t21` - residuals at each hour (21 features)
- `slope_est`, `slope_mean`, `intercept` - slope estimates from linear fit
- `dT_dt_tw_last` - previous twilight rate (known)
- `dT_dt_tw_lag1/2/3` - lagged twilight rates
- `slope_lag1/2` - lagged slopes
- `doy_sin`, `doy_cos` - seasonal

**Data Leakage Audit:** All features verified clean - no future information used.

### Implementation Status
- [x] Create `src/rate_forecast_delta.py`
- [x] Multi-window forecasting (7 × 3h)
- [x] Ridge correction with 32 features
- [x] Data leakage audit
- [ ] Add to paper results

---

## Completed Phases

- [x] Data preparation (fill_gaps, compute_sunrise_sunset)
- [x] Baseline models (Persistence, Linear, RF, MLP)
- [x] NBEATSx ΔT model training
- [x] Ridge slope recovery
- [x] Paper writing (twilight_forecast_report.tex)
- [x] Code refactoring (config.py, modular structure)
- [x] 30-minute prediction intervals
- [ ] Temperature rate forecasting (NEXT)
