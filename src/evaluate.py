"""Generate paper_results_final.csv with NBEATSx-Ridge + Prophet + MeteoBlue.

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

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Configuration - imported from config.py
from config import (
    RESULTS_PATH,
    PAPER_RESULTS_FILE,
    TWILIGHT_OFFSET_FILE,
    TWILIGHT_OFFSET_INSAMPLE_FILE,
    PAPER_RESULTS_V2_FILE,
    METRICS_3H_FILE,
    LEAD_TIMES_HOURS,
    MAX_PREDICTION_HORIZON,
    RIDGE_ALPHA,
    DATA_PATH,
    SLOPE_FILE,
    TRAINING_MODE,
)

# External forecast paths
PROPHET_FILE = Path(__file__).parent.parent / "data" / "results_hybrid_07d_30min.csv"
METEOBLUE_FILE = Path(__file__).parent.parent / "data" / "meteo_blue_weather_station.csv"


# =============================================================================
# External Forecast Functions (Prophet, MeteoBlue)
# =============================================================================


def apply_prophet_blending(df: pd.DataFrame, A: float = 0.90, tau: float = 13.0) -> pd.DataFrame:
    """Apply exponential blending to Prophet forecasts."""
    df = df.copy()
    lead = df["lead_time"].astype(float).to_numpy()
    f = A * np.exp(-lead / tau)
    df["y_hat"] = f * df["yhat_short"] + (1 - f) * df["yhat_long"]
    return df


def handle_prophet_failures(df: pd.DataFrame, xi2_tol: float = 2.0) -> pd.DataFrame:
    """Handle Prophet model failures based on reduced chi-square."""
    out = df.copy()
    xi2_long = out["xi2_long_model"].astype(float).to_numpy()
    xi2_short = out["xi2_short_model"].astype(float).to_numpy()
    fail_long = np.abs(xi2_long - 1.0) > xi2_tol
    fail_short = np.abs(xi2_short - 1.0) > xi2_tol

    yhat_new = out["y_hat"].to_numpy(dtype=float)
    only_long_fail = fail_long & ~fail_short
    yhat_new[only_long_fail] = out.loc[only_long_fail, "yhat_short"].to_numpy(dtype=float)
    only_short_fail = fail_short & ~fail_long
    yhat_new[only_short_fail] = out.loc[only_short_fail, "yhat_long"].to_numpy(dtype=float)

    both_fail = fail_long & fail_short
    if both_fail.any():
        d_long = np.abs(xi2_long - 1.0)
        d_short = np.abs(xi2_short - 1.0)
        choose_long = d_long <= d_short
        idx = np.where(both_fail & choose_long)[0]
        if idx.size:
            yhat_new[idx] = out.iloc[idx]["yhat_long"].to_numpy(dtype=float)
        idx = np.where(both_fail & ~choose_long)[0]
        if idx.size:
            yhat_new[idx] = out.iloc[idx]["yhat_short"].to_numpy(dtype=float)

    out["y_hat"] = yhat_new
    return out


def process_prophet_forecasts(twilight_events: pd.DataFrame) -> pd.DataFrame:
    """Extract Prophet-BMA forecasts matched to twilight events."""
    if not PROPHET_FILE.exists():
        print("  Prophet file not found, skipping")
        return pd.DataFrame()

    print("Processing Prophet-BMA forecasts...")
    TARGET_LEAD_TIMES = [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]

    df = pd.read_csv(PROPHET_FILE)
    df = df.rename(columns={"timestamp": "valid_time"})
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = apply_prophet_blending(df)
    df = handle_prophet_failures(df)

    twilight_events = twilight_events.copy()
    twilight_events["twilight_time"] = pd.to_datetime(twilight_events["twilight_time"])

    results = []
    for lt in TARGET_LEAD_TIMES:
        df_lt = df[np.abs(df["lead_time"] - lt) < 0.1].copy()
        df_lt = df_lt.sort_values("valid_time").reset_index(drop=True)

        for _, tw_row in twilight_events.iterrows():
            tw_time = tw_row["twilight_time"]
            time_diff = np.abs((df_lt["valid_time"] - tw_time).dt.total_seconds())
            within_window = time_diff < 3600

            if not within_window.any():
                continue

            closest_idx = time_diff[within_window].idxmin()
            match = df_lt.loc[closest_idx]
            actual_temp = match["y"]
            forecast_temp = match["y_hat"]
            error = forecast_temp - actual_temp

            if np.abs(error) > 10.0:
                continue

            results.append({
                "twilight_time": match["valid_time"],
                "forecast_time": match["valid_time"] - pd.Timedelta(hours=lt),
                "lead_time_hours": lt,
                "actual_temp": actual_temp,
                "model": "Prophet",
                "forecast_temp": forecast_temp,
                "error": error,
            })

    print(f"  Created {len(results)} Prophet forecasts")
    return pd.DataFrame(results)


def process_meteoblue_forecasts(twilight_events: pd.DataFrame) -> pd.DataFrame:
    """Extract MeteoBlue forecasts for each twilight event."""
    if not METEOBLUE_FILE.exists():
        print("  MeteoBlue file not found, skipping")
        return pd.DataFrame()

    print("Processing MeteoBlue forecasts...")
    df = pd.read_csv(METEOBLUE_FILE)
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.sort_values("valid_time").reset_index(drop=True)

    results = []
    for _, tw_row in twilight_events.iterrows():
        tw_time = tw_row["twilight_time"]
        actual_temp = tw_row["actual_temp"]

        tw_time_utc = pd.Timestamp(tw_time).tz_localize("UTC")
        time_diff = np.abs((df["valid_time"] - tw_time_utc).dt.total_seconds())
        within_window = time_diff < 7200

        if not within_window.any():
            continue

        closest_idx = time_diff[within_window].idxmin()
        match = df.loc[closest_idx]

        lead_time = (match["valid_time"] - match["issue_time"]).total_seconds() / 3600.0
        forecast_temp = match["temperature"]

        results.append({
            "twilight_time": tw_time,
            "forecast_time": match["issue_time"].tz_localize(None),
            "lead_time_hours": lead_time,
            "actual_temp": actual_temp,
            "model": "MeteoBlue",
            "forecast_temp": forecast_temp,
            "error": forecast_temp - actual_temp,
        })

    print(f"  Created {len(results)} MeteoBlue forecasts")
    return pd.DataFrame(results)

# Use config values - select files based on training mode and slope correction
PAPER_RESULTS = PAPER_RESULTS_FILE

if TRAINING_MODE == "operational":
    # Operational mode: train Ridge on in-sample NBEATSx predictions (pre-2025)
    TWILIGHT_OFFSET = TWILIGHT_OFFSET_INSAMPLE_FILE
    OUTPUT_PATH = PAPER_RESULTS_V2_FILE
    print("*** OPERATIONAL MODE: Training Ridge on in-sample predictions ***")
elif SLOPE_FILE is not None:
    # Paper mode with slope correction
    TWILIGHT_OFFSET = RESULTS_PATH / "twilight_offset_predictions_iter1.csv"
    OUTPUT_PATH = RESULTS_PATH / "paper_results_v2_iter1.csv"
    print("*** PAPER MODE (ITERATION 1): Using slope-corrected predictions ***")
else:
    # Paper mode (default): train Ridge on 2025 test data with odd/even split
    TWILIGHT_OFFSET = TWILIGHT_OFFSET_FILE
    OUTPUT_PATH = PAPER_RESULTS_V2_FILE
    print("*** PAPER MODE: Training Ridge on 2025 test data (odd/even split) ***")

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

if TRAINING_MODE == "operational":
    # Load in-sample predictions for training
    df_train = pd.read_csv(TWILIGHT_OFFSET_INSAMPLE_FILE)
    df_train["tw_date"] = pd.to_datetime(df_train["tw_date"])
    df_train["tw_time"] = pd.to_datetime(df_train["tw_time"])
    df_train["forecast_time"] = pd.to_datetime(df_train["forecast_time"])
    df_train["target_time"] = pd.to_datetime(df_train["target_time"])
    print(f"  Training data (in-sample): {len(df_train)} rows, {df_train['tw_time'].nunique()} twilights")

    # Load 2025 test predictions for testing
    df_test = pd.read_csv(TWILIGHT_OFFSET_FILE)
    df_test["tw_date"] = pd.to_datetime(df_test["tw_date"])
    df_test["tw_time"] = pd.to_datetime(df_test["tw_time"])
    df_test["forecast_time"] = pd.to_datetime(df_test["forecast_time"])
    df_test["target_time"] = pd.to_datetime(df_test["target_time"])
    print(f"  Test data (2025): {len(df_test)} rows, {df_test['tw_time'].nunique()} twilights")

    # Combine for feature engineering (but track source)
    df_train["_is_train"] = True
    df_test["_is_train"] = False
    df = pd.concat([df_train, df_test], ignore_index=True)
else:
    # Paper mode: load single file, split by odd/even days
    df = pd.read_csv(TWILIGHT_OFFSET)
    df["tw_date"] = pd.to_datetime(df["tw_date"])
    df["tw_time"] = pd.to_datetime(df["tw_time"])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    df["_is_train"] = None  # Will be determined by day_num later

print(f"  Total twilight_offset rows: {len(df)}")
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

# 4. Seasonal interaction features (reduce seasonal bias)
df["rate_tw_mid_x_doy_cos"] = df["rate_twilight_to_midnight"] * df["doy_cos"]
df["rate_mid_tw_x_doy_cos"] = df["rate_midday_to_twilight"] * df["doy_cos"]
print(f"  rate_tw_mid_x_doy_cos, rate_mid_tw_x_doy_cos: seasonal interaction features")

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
    # Seasonal interaction features (reduce seasonal bias)
    "rate_tw_mid_x_doy_cos",
    "rate_mid_tw_x_doy_cos",
    # Temperature trend
    "trend_temp_3d",
]  # 26 features total

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

# Train/test split based on training mode
if TRAINING_MODE == "operational":
    # Operational: train on in-sample data, test on 2025 data
    train_3h = df_3h[df_3h["_is_train"] == True]
    test_3h = df_3h[df_3h["_is_train"] == False]
else:
    # Paper mode: odd days for training, even days for testing
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

# Store 3h predictions for reuse
predictions_3h = dict(zip(test_3h["tw_time"], pred_tw_temp_3h))

# Seasonal bias analysis (interaction features reduce bias)
def get_season(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Fall"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"

test_3h = test_3h.copy()
test_3h["month"] = test_3h["tw_time"].dt.month
test_3h["season"] = test_3h["month"].apply(get_season)

seasonal_bias = {}
print("\n  Seasonal bias (3h lead):")
for season in ["Summer", "Fall", "Winter", "Spring"]:
    mask = test_3h["season"] == season
    if mask.any():
        bias = error_3h[mask.values].mean()
        seasonal_bias[season] = bias
        print(f"    {season}: {bias:+.3f}°C (n={mask.sum()})")

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

        # Train/test split based on training mode
        if TRAINING_MODE == "operational":
            train = df_lt[df_lt["_is_train"] == True]
            test = df_lt[df_lt["_is_train"] == False]
        else:
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

ridge_df["twilight_time"] = pd.to_datetime(ridge_df["twilight_time"])

# =============================================================================
# Save slopes at 10am LOCAL TIME (for iterative correction)
# =============================================================================
print("\n[*] Training Ridge at 10am local time...")

# Filter for 10am local (target_time hour = 10)
df["target_hour"] = df["target_time"].dt.hour
df_10am = df[(df["target_hour"] >= 9) & (df["target_hour"] <= 10)].copy()
df_10am = df_10am.groupby("tw_time").last().reset_index()
df_10am = df_10am.dropna(subset=feature_cols + [target_col, "tw_temp"])

# Train/test split based on training mode
if TRAINING_MODE == "operational":
    train_10am = df_10am[df_10am["_is_train"] == True]
    test_10am = df_10am[df_10am["_is_train"] == False]
else:
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
# Rename Oracle models (use hyphen, not underscore)
paper_results_v2["model"] = paper_results_v2["model"].replace(
    "NBEATSx-Oracle", "NBEATSx-Oracle"  # Keep as-is
)

# Add NBEATSx-Ridge predictions
paper_results_v2 = pd.concat([paper_results_v2, ridge_df], ignore_index=True)

# =============================================================================
# Add External Forecasts (Prophet, MeteoBlue)
# =============================================================================
print("\n[6] Adding external forecasts...")

# Get twilight events for external forecast matching
twilight_events = paper_results_v2[["twilight_time", "actual_temp"]].drop_duplicates()

# Process Prophet forecasts
prophet_df = process_prophet_forecasts(twilight_events)
if len(prophet_df) > 0:
    paper_results_v2 = pd.concat([paper_results_v2, prophet_df], ignore_index=True)

# Process MeteoBlue forecasts
meteoblue_df = process_meteoblue_forecasts(twilight_events)
if len(meteoblue_df) > 0:
    paper_results_v2 = pd.concat([paper_results_v2, meteoblue_df], ignore_index=True)

# Save final results
OUTPUT_PATH = RESULTS_PATH / "paper_results_final.csv"
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
    "RandomForest",
    "MLP",
    "NBEATSx-Oracle",
    "NBEATSx-Ridge",
    "Prophet",
    "MeteoBlue",
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
