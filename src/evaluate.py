"""Generate paper_results_v2.csv with NBEATSx-Ridge predictions.

Two-stage correction approach:
1. NBEATSx approximates current temperature (nowcast, pred_horizon=0)
2. res = actual - NBEATSx_approx (residual at current time)
3. Ridge uses res (and other features) to predict tw_slope
4. Final prediction = T_tw_last + Ridge_slope

For each lead_time:
1. Filter: forecast_time <= current_time (prediction issued before decision)
2. Filter: target_time <= current_time (no future data leakage)
3. Select most recent prediction (max forecast_time, then max target_time)
4. Train Ridge on odd days, test on even days
5. Target: tw_slope = tw_temp - T_tw_last

Optimized feature set (20 features, Jan 2026):
- Core: res, rates (3), day_length, res_sq_temp, twilight_cos, doy_sin, doy_cos
- Lags: res_1, res_3, ..., res_21 (11 odd lags)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Configuration - imported from config.py
from config import (
    RESULTS_PATH,
    PAPER_RESULTS_FILE,
    TWILIGHT_OFFSET_FILE,
    PAPER_RESULTS_V2_FILE,
    METRICS_3H_FILE,
    LEAD_TIMES_HOURS,
    MAX_PREDICTION_HORIZON,
    RIDGE_ALPHA,
    DATA_PATH,
    SLOPE_FILE,
)

# Use config values - switch to iter1 files if using slope correction
PAPER_RESULTS = PAPER_RESULTS_FILE
if SLOPE_FILE is not None:
    TWILIGHT_OFFSET = RESULTS_PATH / "twilight_offset_predictions_iter1.csv"
    OUTPUT_PATH = RESULTS_PATH / "paper_results_v2_iter1.csv"
    print("*** ITERATION 1: Using slope-corrected predictions ***")
else:
    TWILIGHT_OFFSET = TWILIGHT_OFFSET_FILE
    OUTPUT_PATH = PAPER_RESULTS_V2_FILE
LEAD_TIMES = LEAD_TIMES_HOURS

print("=" * 70)
print("GENERATING paper_results_v2.csv WITH NBEATSx-Ridge")
print(
    f"Max prediction horizon: {MAX_PREDICTION_HORIZON}h (target_time - forecast_time)"
)
print("=" * 70)

# =============================================================================
# 1. Load data
# =============================================================================
print("\n[1] Loading data...")
paper_results = pd.read_csv(PAPER_RESULTS)
paper_results["twilight_time"] = pd.to_datetime(paper_results["twilight_time"])
paper_results["forecast_time"] = pd.to_datetime(paper_results["forecast_time"])
print(f"  paper_results rows: {len(paper_results)}")

df = pd.read_csv(TWILIGHT_OFFSET)
df["tw_date"] = pd.to_datetime(df["tw_date"])
df["tw_time"] = pd.to_datetime(df["tw_time"])
df["forecast_time"] = pd.to_datetime(df["forecast_time"])
df["target_time"] = pd.to_datetime(df["target_time"])

print(f"  twilight_offset rows: {len(df)}")
print(f"  Twilight events: {df['tw_time'].nunique()}")
print(f"  pred_time values: {df['pred_time'].unique()}")

# =============================================================================
# 2. Build features
# =============================================================================
print("\n[2] Building features...")

# Rename res_tw to res for cleaner notation
df = df.rename(columns={"res_tw": "res"})

# 11 odd-hour lagged residuals (original paper feature set)
# These capture the residual evolution pattern for slope prediction
ODD_HOUR_LAGS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# Add odd-hour lagged residuals
lag_cols = []
for lag_h in ODD_HOUR_LAGS:
    col_name = f"res_{lag_h}h"
    lag_steps = int(lag_h * 4)  # Convert hours to 15-min steps
    df[col_name] = df["res"].shift(lag_steps)
    lag_cols.append(col_name)
print(f"  Odd-hour lagged residuals: {len(lag_cols)} features (1h, 3h, ..., 21h)")

# Target: twilight slope
df["tw_slope"] = df["tw_temp"] - df["T_tw_last"]

# Train/test split: odd/even days
df["day_num"] = (df["tw_date"] - df["tw_date"].min()).dt.days

# -----------------------------------------------------------------------------
# Seasonal features
# -----------------------------------------------------------------------------
print("  Adding seasonal features...")

# Day-of-year cyclical encoding
df["doy"] = df["tw_time"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

# Twilight hour cyclical encoding
df["tw_hour"] = df["tw_time"].dt.hour + df["tw_time"].dt.minute / 60
df["twilight_cos"] = np.cos(2 * np.pi * df["tw_hour"] / 24)

# Day length (hours from sunrise to twilight)
df_sun = pd.read_csv(DATA_PATH)
df_sun["timestamp"] = pd.to_datetime(df_sun["timestamp"])
sunrise_times = df_sun[df_sun["sunrise_temp"].notna()][["timestamp"]].copy()
sunrise_times = sunrise_times.rename(columns={"timestamp": "sunrise_time"})

day_lengths = []
for tw_time in df["tw_time"].unique():
    # Find sunrise within 18h before twilight
    mask = (sunrise_times["sunrise_time"] < tw_time) & \
           (sunrise_times["sunrise_time"] >= tw_time - pd.Timedelta(hours=18))
    nearby_sunrise = sunrise_times[mask]
    if len(nearby_sunrise) > 0:
        sr_time = nearby_sunrise["sunrise_time"].max()
        day_len = (tw_time - sr_time).total_seconds() / 3600
        day_lengths.append({"tw_time": tw_time, "day_length": day_len})
    else:
        day_lengths.append({"tw_time": tw_time, "day_length": np.nan})

day_length_df = pd.DataFrame(day_lengths)
df = df.merge(day_length_df, on="tw_time", how="left")
print(f"  Seasonal: doy_sin, doy_cos, twilight_cos, day_length (mean={df['day_length'].mean():.1f}h)")

# -----------------------------------------------------------------------------
# Temperature trend feature
# -----------------------------------------------------------------------------
print("  Adding trend feature...")
print(f"  trend_temp_3d: 3-day temperature trend (mean={df['trend_temp_3d'].mean():.2f})")

# -----------------------------------------------------------------------------
# NEW FEATURES (Jan 2026 research)
# -----------------------------------------------------------------------------
print("  Adding new research features...")

# 1. Multi-day lagged residuals (r=0.90, r=0.89)
df["res_1d"] = df["res"].shift(96)   # 24h = 96 steps
df["res_2d"] = df["res"].shift(192)  # 48h = 192 steps
print(f"  res_1d, res_2d: multi-day lagged residuals")

# 2. Temperature since sunrise (r=0.60)
df["temp_since_sunrise"] = df["temp_actual"] - df["temp_last_sunrise"]
print(f"  temp_since_sunrise: warming since sunrise (mean={df['temp_since_sunrise'].mean():.2f}°C)")

# 3. rate_midday_to_twilight is already in data (r=0.40)
print(f"  rate_midday_to_twilight: afternoon rate (mean={df['rate_midday_to_twilight'].mean():.2f}°C/h)")

# =============================================================================
# 3. Define feature columns (20 features, optimized Jan 2026)
# =============================================================================
feature_cols = [
    # Core residual (correlation +0.98 with target)
    "res",
    # 11 odd-hour lagged residuals (original paper features)
    "res_1h", "res_3h", "res_5h", "res_7h", "res_9h", "res_11h",
    "res_13h", "res_15h", "res_17h", "res_19h", "res_21h",
    # Multi-day lagged residuals (NEW: r=0.90, r=0.89)
    "res_1d", "res_2d",
    # Diurnal rate features
    "rate_twilight_to_midnight",
    "rate_sunrise_to_midday",
    "rate_midnight_to_sunrise",
    "rate_midday_to_twilight",  # NEW: r=0.40
    # Temperature features
    "temp_since_sunrise",  # NEW: r=0.60
    # Seasonal features
    "day_length",
    "twilight_cos",
    "doy_sin",
    "doy_cos",
    # Temperature trend
    "trend_temp_3d",
]  # 24 features total

target_col = "tw_slope"
print(f"\n[3] Feature set: {len(feature_cols)} features")

# =============================================================================
# 4. Create NBEATSx-Ridge rows for paper_results_v2
# =============================================================================
# =============================================================================
# 4. Train Ridge at 3h lead time (operational cutoff)
# =============================================================================
print("\n[4] Training Ridge model at 3h lead time (operational cutoff)...")
OPERATIONAL_LEAD_TIME = 3.0

# Prepare data at 3h lead time
df["current_time"] = df["tw_time"] - pd.Timedelta(hours=OPERATIONAL_LEAD_TIME)
df_available = df[df["forecast_time"] <= df["current_time"]].copy()
df_no_future = df_available[df_available["target_time"] <= df_available["current_time"]].copy()
df_by_issue = df_no_future.loc[df_no_future.groupby("tw_time")["forecast_time"].idxmax()]
df_recent = df_no_future.merge(df_by_issue[["tw_time", "forecast_time"]], on=["tw_time", "forecast_time"])
df_recent = df_recent.loc[df_recent.groupby("tw_time")["target_time"].idxmax()]
df_recent["pred_horizon"] = (df_recent["target_time"] - df_recent["forecast_time"]).dt.total_seconds() / 3600
df_recent = df_recent[df_recent["pred_horizon"] <= MAX_PREDICTION_HORIZON].copy()

df_3h = df_recent.dropna(subset=feature_cols + [target_col, "tw_temp"])
train_3h = df_3h[df_3h["day_num"] % 2 == 1]
test_3h = df_3h[df_3h["day_num"] % 2 == 0]

print(f"  Training samples: {len(train_3h)}, Test samples: {len(test_3h)}")

# Train Ridge model once at 3h
scaler = StandardScaler()
X_train_3h = scaler.fit_transform(train_3h[feature_cols].values)
model = Ridge(alpha=RIDGE_ALPHA)
model.fit(X_train_3h, train_3h[target_col].values)

# Predict at 3h (this prediction is used for all lead times <= 3h)
X_test_3h = scaler.transform(test_3h[feature_cols].values)
pred_slope_3h = model.predict(X_test_3h)
pred_tw_temp_3h = test_3h["T_tw_last"].values + pred_slope_3h
error_3h = pred_tw_temp_3h - test_3h["tw_temp"].values
rmse_3h = np.sqrt((error_3h**2).mean())
print(f"  RMSE at 3h: {rmse_3h:.3f}°C")

# Store 3h predictions for reuse
predictions_3h = dict(zip(test_3h["tw_time"], pred_tw_temp_3h))

# Also predict slopes for train set (for next iteration)
pred_slope_train_3h = model.predict(X_train_3h)

# Save ALL slope predictions (train + test) for iterative refinement
slope_predictions = []
for tw_time, slope in zip(train_3h["tw_time"], pred_slope_train_3h):
    slope_predictions.append({"tw_time": tw_time, "slope": slope})
for tw_time, slope in zip(test_3h["tw_time"], pred_slope_3h):
    slope_predictions.append({"tw_time": tw_time, "slope": slope})

slope_df = pd.DataFrame(slope_predictions)
slope_file = RESULTS_PATH / "slope_predictions_3h.csv"
slope_df.to_csv(slope_file, index=False)
print(f"  Saved slope predictions to: {slope_file.name} ({len(slope_df)} twilights)")

# =============================================================================
# 5. Create NBEATSx-Ridge rows for all lead times
# =============================================================================
print("\n[5] Creating NBEATSx-Ridge rows...")
print("    Correction applies up to beginning of operations (3h before twilight)")
print("-" * 85)
print(
    f"{'Lead':>6} {'Strategy':>12} {'N_test':>8} {'RMSE':>10}"
)
print("-" * 85)

all_ridge_rows = []

for lead_time in LEAD_TIMES:
    # For lead times <= 3h, use the 3h prediction
    # Correction applies up to beginning of operations (3h before twilight)
    if lead_time <= OPERATIONAL_LEAD_TIME:
        # Use 3h predictions for all lead times <= 3h
        test_twilights = test_3h["tw_time"].values
        actual_tw_temp = test_3h["tw_temp"].values

        pred_tw_temp = pred_tw_temp_3h
        error_ridge = error_3h
        rmse_ridge = rmse_3h

        print(f"{lead_time:>5.1f}h {'ops_cutoff':>12} {len(test_3h):>8} {rmse_ridge:>9.3f}°C")

        for i, tw_time in enumerate(test_twilights):
            all_ridge_rows.append({
                "twilight_time": tw_time,
                "forecast_time": tw_time - pd.Timedelta(hours=OPERATIONAL_LEAD_TIME),
                "lead_time_hours": lead_time,
                "actual_temp": actual_tw_temp[i],
                "model": "NBEATSx-Ridge",
                "forecast_temp": pred_tw_temp[i],
                "error": error_ridge[i],
            })
    else:
        # For lead times > 3h, train a separate model with data available at that time
        df["current_time"] = df["tw_time"] - pd.Timedelta(hours=lead_time)
        df_available = df[df["forecast_time"] <= df["current_time"]].copy()

        if len(df_available) == 0:
            print(f"{lead_time:>5.1f}h: no predictions available")
            continue

        df_no_future = df_available[df_available["target_time"] <= df_available["current_time"]].copy()

        if len(df_no_future) == 0:
            print(f"{lead_time:>5.1f}h: no data available")
            continue

        df_by_issue = df_no_future.loc[df_no_future.groupby("tw_time")["forecast_time"].idxmax()]
        df_recent = df_no_future.merge(df_by_issue[["tw_time", "forecast_time"]], on=["tw_time", "forecast_time"])
        df_recent = df_recent.loc[df_recent.groupby("tw_time")["target_time"].idxmax()]
        df_recent["pred_horizon"] = (df_recent["target_time"] - df_recent["forecast_time"]).dt.total_seconds() / 3600
        df_recent = df_recent[df_recent["pred_horizon"] <= MAX_PREDICTION_HORIZON].copy()

        if len(df_recent) == 0:
            print(f"{lead_time:>5.1f}h: no data after horizon filter")
            continue

        df_lt = df_recent.dropna(subset=feature_cols + [target_col, "tw_temp"])
        train = df_lt[df_lt["day_num"] % 2 == 1]
        test = df_lt[df_lt["day_num"] % 2 == 0]

        if len(train) < 10 or len(test) < 10:
            print(f"{lead_time:>5.1f}h: skipping (train={len(train)}, test={len(test)})")
            continue

        # Train model for this lead time
        scaler_lt = StandardScaler()
        X_train_lt = scaler_lt.fit_transform(train[feature_cols].values)
        X_test_lt = scaler_lt.transform(test[feature_cols].values)

        model_lt = Ridge(alpha=RIDGE_ALPHA)
        model_lt.fit(X_train_lt, train[target_col].values)

        pred_slope = model_lt.predict(X_test_lt)
        pred_tw_temp = test["T_tw_last"].values + pred_slope
        actual_tw_temp = test["tw_temp"].values

        error_ridge = pred_tw_temp - actual_tw_temp
        rmse_ridge = np.sqrt((error_ridge**2).mean())

        print(f"{lead_time:>5.1f}h {'trained':>12} {len(test):>8} {rmse_ridge:>9.3f}°C")


        for i, (_, row) in enumerate(test.iterrows()):
            all_ridge_rows.append({
                "twilight_time": row["tw_time"],
                "forecast_time": row["target_time"],
                "lead_time_hours": lead_time,
                "actual_temp": row["tw_temp"],
                "model": "NBEATSx-Ridge",
                "forecast_temp": pred_tw_temp[i],
                "error": error_ridge[i],
            })

print("-" * 85)
ridge_df = pd.DataFrame(all_ridge_rows)
print(f"  Total NBEATSx-Ridge rows: {len(ridge_df)}")

# =============================================================================
# Save slopes at 10am LOCAL TIME (for iterative correction)
# =============================================================================
print("\n[*] Training Ridge at 10am local time...")

# Filter for 10am local (target_time hour = 10)
df["target_hour"] = df["target_time"].dt.hour
df_10am = df[(df["target_hour"] >= 9) & (df["target_hour"] <= 10)].copy()
df_10am = df_10am.groupby("tw_time").last().reset_index()
df_10am = df_10am.dropna(subset=feature_cols + [target_col, "tw_temp"])

train_10am = df_10am[df_10am["day_num"] % 2 == 1]
test_10am = df_10am[df_10am["day_num"] % 2 == 0]

if len(train_10am) >= 10 and len(test_10am) >= 10:
    scaler_10am = StandardScaler()
    X_train_10am = scaler_10am.fit_transform(train_10am[feature_cols].values)
    X_test_10am = scaler_10am.transform(test_10am[feature_cols].values)

    model_10am = Ridge(alpha=RIDGE_ALPHA)
    model_10am.fit(X_train_10am, train_10am[target_col].values)

    pred_slope_train_10am = model_10am.predict(X_train_10am)
    pred_slope_test_10am = model_10am.predict(X_test_10am)

    # Evaluate
    pred_tw_10am = test_10am["T_tw_last"].values + pred_slope_test_10am
    error_10am = pred_tw_10am - test_10am["tw_temp"].values
    rmse_10am = np.sqrt((error_10am**2).mean())
    mean_lead = (test_10am["tw_time"] - test_10am["target_time"]).dt.total_seconds().mean() / 3600

    print(f"  Samples: train={len(train_10am)}, test={len(test_10am)}")
    print(f"  Mean lead time: {mean_lead:.1f}h")
    print(f"  RMSE @ 10am local: {rmse_10am:.3f}°C, <1°C: {(np.abs(error_10am) < 1).mean() * 100:.1f}%")

    # Save ALL slopes (train + test)
    slope_predictions_10am = []
    for tw_time, slope in zip(train_10am["tw_time"], pred_slope_train_10am):
        slope_predictions_10am.append({"tw_time": tw_time, "slope": slope})
    for tw_time, slope in zip(test_10am["tw_time"], pred_slope_test_10am):
        slope_predictions_10am.append({"tw_time": tw_time, "slope": slope})

    slope_df_10am = pd.DataFrame(slope_predictions_10am)
    slope_file_10am = RESULTS_PATH / "slope_predictions_10am.csv"
    slope_df_10am.to_csv(slope_file_10am, index=False)
    print(f"  Saved slope predictions to: {slope_file_10am.name} ({len(slope_df_10am)} twilights)")
else:
    print(f"  Not enough data at 10am (train={len(train_10am)}, test={len(test_10am)})")

# =============================================================================
# 5. Combine and save
# =============================================================================
print("\n[5] Creating paper_results_v2.csv...")

paper_results_v2 = paper_results.copy()
paper_results_v2["model"] = paper_results_v2["model"].replace(
    "NBEATSx_Oracle", "NBEATSx"
)
paper_results_v2["model"] = paper_results_v2["model"].replace("NHITS_Oracle", "NHITS")

paper_results_v2 = pd.concat([paper_results_v2, ridge_df], ignore_index=True)
paper_results_v2.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved to: {OUTPUT_PATH}")

# =============================================================================
# 6. Summary metrics at 3h
# =============================================================================
print("\n[6] Metrics at 3h lead time:")
print("=" * 70)

at_3h = paper_results_v2[paper_results_v2["lead_time_hours"] == 3.0].copy()
at_3h["abs_error"] = np.abs(at_3h["error"])

metrics = []
for model in [
    "Persistence",
    "Persistence-Twilight",
    "Linear",
    "NHITS",
    "RandomForest",
    "MLP",
    "NBEATSx",
    "NBEATSx-Ridge",
]:
    data = at_3h[at_3h["model"] == model]
    if len(data) > 0:
        metrics.append(
            {
                "Model": model,
                "RMSE": np.sqrt(np.mean(data["error"] ** 2)),
                "MAE": np.mean(data["abs_error"]),
                "Bias": data["error"].mean(),
                "% < 1C": (data["abs_error"] < 1.0).mean() * 100,
                "N": len(data),
            }
        )

metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))

# Save metrics
metrics_df.to_csv(METRICS_3H_FILE, index=False)
print(f"\nMetrics saved to: {METRICS_3H_FILE}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
