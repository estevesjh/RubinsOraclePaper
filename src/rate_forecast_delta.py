"""Rate forecaster using Δ(dT/dt) formulation - mirrors temperature forecaster.

Key insight: Train on delta_dT_dt = dT_dt - Rate-Trend, where:
  Rate-Trend(t) = dT_dt_tw_last + (dT_dt_tw - dT_dt_tw_last)/24 * t

At inference (dT_dt_tw unknown), use flat baseline:
  rate_approx = delta_dT_dt_pred + dT_dt_tw_last

Residual reveals the slope: res(t) ≈ slope_rate * t
Ridge uses accumulated residuals to predict dT_dt_tw.
"""

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    FREQ,
    NBEATS_FUTR_EXOG,
    NBEATS_HIST_EXOG,
    NBEATS_MAX_STEPS,
    RESULTS_PATH,
    SAMPLES_PER_HOUR,
    TEST_START_DATE,
)

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing import (
    add_key_time_features,
    add_temp_variability_features,
    add_trend_2h,
    add_trend_features,
    add_twilight_features,
)

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# Prediction lead time
LEAD_TIME_HOURS = 3
LEAD_TIME_SAMPLES = LEAD_TIME_HOURS * SAMPLES_PER_HOUR

# Multiple forecast windows to cover t=1h to t=21h after last twilight
# Each window is 3h with hourly sampling
# (hours_before_tw, sample_times_after_last_tw)
FORECAST_WINDOWS = [
    (23, [1, 2, 3]),      # Window 0: start 23h before tw, covers t=1-3
    (20, [4, 5, 6]),      # Window 1: start 20h before tw, covers t=4-6
    (17, [7, 8, 9]),      # Window 2: start 17h before tw, covers t=7-9
    (14, [10, 11, 12]),   # Window 3: start 14h before tw, covers t=10-12
    (11, [13, 14, 15]),   # Window 4: start 11h before tw, covers t=13-15
    (8, [16, 17, 18]),    # Window 5: start 8h before tw, covers t=16-18
    (5, [19, 20, 21]),    # Window 6: start 5h before tw, covers t=19-21
]
HOURS_FROM_LAST_TW = list(range(1, 22))  # All t values: t=1 to t=21
MAX_HORIZON = 3 * SAMPLES_PER_HOUR  # 3h = 12 steps
HORIZON = MAX_HORIZON

# Savitzky-Golay filter parameters
SAVGOL_WINDOW = 13
SAVGOL_POLYORDER = 1

# Model hyperparameters
INPUT_SIZE = int(1 * 24 * SAMPLES_PER_HOUR)  # 96 = 24 hours
LEARNING_RATE = 0.01
MLP_UNITS = 3 * [[32, 32]]



@contextmanager
def suppress_output():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def compute_savgol_derivative(
    y: np.ndarray, window: int = 13, polyorder: int = 2, delta: float = 0.25
) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    return savgol_filter(
        y, window_length=window, polyorder=polyorder, deriv=1, delta=delta
    )


def main():
    print("=" * 70)
    print("RATE FORECASTER with Δ(dT/dt) FORMULATION")
    print(f"Train: up to {TEST_START_DATE.date()}, Test: 2025")
    print("=" * 70)

    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Add features
    df = add_twilight_features(df)
    df = add_trend_features(df)
    df = add_key_time_features(df)
    df = add_trend_2h(df)
    df = add_temp_variability_features(df)

    # Compute smooth derivative
    print(f"  Computing derivative with savgol_filter (window={SAVGOL_WINDOW})...")
    df["dT_dt"] = compute_savgol_derivative(
        df["y"].values, window=SAVGOL_WINDOW, delta=0.25
    )

    # Required features
    df["temp_raw"] = df["y"].copy()
    df["temp_last_sunrise"] = df["sunrise_temp"].ffill()

    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"  dT_dt std: {df['dT_dt'].std():.3f} °C/h")

    # =========================================================================
    # 2. Compute Δ(dT/dt) = dT_dt - Rate-Trend
    # =========================================================================
    print("\n[2] Computing Δ(dT/dt) target...")

    # Get twilight mask and rates
    tw_mask = df["twilight_temp"].notna()

    # dT_dt at twilight times
    df["dT_dt_tw"] = df["dT_dt"].where(tw_mask)

    # Forward-fill: dT_dt_tw_last (PREVIOUS twilight's rate)
    # IMPORTANT: shift(1) ensures at twilight, we get the PREVIOUS twilight's rate
    df["dT_dt_tw_last"] = df["dT_dt_tw"].shift(1).ffill()

    # Backward-fill: dT_dt_tw_next (next twilight's rate) - for training only
    # This is the actual twilight rate we're trying to predict
    df["dT_dt_tw_next"] = df["dT_dt_tw"].bfill()

    # Compute hours since last twilight
    # IMPORTANT: shift(1) ensures at twilight, we get time since PREVIOUS twilight (=24h)
    tw_times = df["ds"].where(tw_mask).shift(1).ffill()
    df["t_from_last_tw"] = (df["ds"] - tw_times).dt.total_seconds() / 3600

    # Rate-Trend (linear interpolation between twilights)
    # Rate-Trend(t) = dT_dt_tw_last + (dT_dt_tw_next - dT_dt_tw_last)/24 * t
    df["slope_rate"] = (df["dT_dt_tw_next"] - df["dT_dt_tw_last"]) / 24
    df["rate_trend"] = df["dT_dt_tw_last"] + df["slope_rate"] * df["t_from_last_tw"]

    # New target: rate relative to trend
    df["delta_dT_dt"] = df["dT_dt"] - df["rate_trend"]

    # Verify boundary condition
    tw_delta = df.loc[tw_mask, "delta_dT_dt"].dropna()
    print(
        f"  Δ(dT/dt) at twilight: mean={tw_delta.mean():.4f}, std={tw_delta.std():.4f}"
    )
    print("  (Should be ~0 at twilight boundaries)")
    print(
        f"  Δ(dT/dt) overall: mean={df['delta_dT_dt'].mean():.4f}, std={df['delta_dT_dt'].std():.4f}"
    )

    # Compute twilight_cos for future exogenous
    sunrise_times = df["ds"].where(df["sunrise_temp"].notna())
    df["last_sunrise_time"] = sunrise_times.ffill()
    twilight_times_col = df["ds"].where(tw_mask)
    df["last_tw_time"] = twilight_times_col.ffill()
    df["next_sunrise_time"] = sunrise_times.shift(-1).bfill()

    # =========================================================================
    # 3. Prepare NBEATSx data
    # =========================================================================
    print("\n[3] Preparing NBEATSx data...")

    hist_exog = NBEATS_HIST_EXOG.copy()
    futr_exog = NBEATS_FUTR_EXOG.copy()
    all_exog = hist_exog + futr_exog

    print(f"  hist_exog: {hist_exog}")
    print(f"  futr_exog: {futr_exog}")

    # Use delta_dT_dt as target
    all_cols = ["ds", "delta_dT_dt"] + all_exog
    nf_df = df[all_cols].dropna().copy()
    nf_df["y"] = nf_df["delta_dT_dt"]
    nf_df["unique_id"] = "delta_rate"
    nf_df = nf_df.reset_index(drop=True)

    print(f"  NBEATSx data shape: {len(nf_df)}")

    # =========================================================================
    # 4. Train NBEATSx model (up to TEST_START_DATE)
    # =========================================================================
    print("\n[4] Training NBEATSx model on Δ(dT/dt)...")

    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    train_df = nf_df[nf_df["ds"] < TEST_START_DATE].copy()
    print(f"  Training samples: {len(train_df)} (up to {TEST_START_DATE.date()})")
    print(f"  Input size: {INPUT_SIZE}, LR: {LEARNING_RATE}, MLP: {MLP_UNITS}")

    model = NBEATSx(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=NBEATS_MAX_STEPS,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=LEARNING_RATE,
        scaler_type="robust",
        enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "exogenous"],
        mlp_units=MLP_UNITS,
        n_blocks=[1, 1, 1],
    )

    with suppress_output():
        nf = NeuralForecast(models=[model], freq=FREQ)
        nf.fit(train_df)

    print("  Training complete!")

    # =========================================================================
    # 5. Build test twilight dataset (2025)
    # =========================================================================
    print("\n[5] Building test twilight dataset...")

    tw_df = df[tw_mask].copy()
    test_tw = tw_df[tw_df["ds"] >= TEST_START_DATE].dropna(subset=["dT_dt"]).copy()
    print(f"  Test twilights: {len(test_tw)} (from {TEST_START_DATE.date()})")

    # =========================================================================
    # 6. Prepare batched predictions (multiple windows per twilight)
    # =========================================================================
    print("\n[6] Preparing batched predictions...")
    print(f"  {len(FORECAST_WINDOWS)} windows per twilight, horizon={HORIZON} steps")
    print(f"  Will extract residuals at t={HOURS_FROM_LAST_TW} hours after last twilight")

    all_hist_dfs = []
    all_futr_dfs = []
    prediction_info = []  # One entry per (twilight, window) combination
    twilight_info = {}  # Aggregated info per twilight (tw_idx -> dict)

    for tw_idx, row in test_tw.iterrows():
        tw_time = row["ds"]
        dT_dt_tw_actual = row["dT_dt"]  # Actual twilight rate (target)
        dT_dt_tw_last = row["dT_dt_tw_last"]  # Previous twilight rate (known)
        last_tw_time = row["last_tw_time"]
        last_sunrise_time = row["last_sunrise_time"]
        next_sunrise_time = (
            row["next_sunrise_time"]
            if pd.notna(row["next_sunrise_time"])
            else tw_time + pd.Timedelta(hours=12)
        )

        # Get actual dT_dt at all sample times (t=1 to t=21 hours after last twilight)
        actual_rates = {}
        for t_h in HOURS_FROM_LAST_TW:
            time_at_t = tw_time - pd.Timedelta(hours=24 - t_h)
            idx_nearest = (df["ds"] - time_at_t).abs().idxmin()
            actual_rates[f"dT_dt_t{t_h}"] = df.loc[idx_nearest, "dT_dt"]

        # Store twilight-level info
        twilight_info[tw_idx] = {
            "tw_time": tw_time,
            "dT_dt_tw_actual": dT_dt_tw_actual,
            "dT_dt_tw_last": dT_dt_tw_last,
            **actual_rates,
        }

        # Loop over each forecast window
        for window_idx, (hours_before_tw, sample_times) in enumerate(FORECAST_WINDOWS):
            unique_id = f"pred_{tw_idx}_w{window_idx}"
            forecast_time = tw_time - pd.Timedelta(hours=hours_before_tw)

            # Historical data before forecast_time
            mask = df["ds"] < forecast_time
            recent_df = df.loc[mask]

            nf_recent = (
                recent_df[["ds", "delta_dT_dt"] + all_exog]
                .dropna()
                .tail(5 * INPUT_SIZE)
                .copy()
            )

            if len(nf_recent) < INPUT_SIZE:
                continue

            nf_recent["y"] = nf_recent["delta_dT_dt"]
            nf_recent["unique_id"] = unique_id

            # Future exogenous - twilight_cos (for 3h horizon)
            training_end_ds = nf_recent["ds"].max()
            futr_end = forecast_time + pd.Timedelta(minutes=15 * HORIZON)
            future_timestamps = pd.date_range(training_end_ds, futr_end, freq=FREQ)
            futr_df = pd.DataFrame({"unique_id": unique_id, "ds": future_timestamps})

            # Compute twilight_cos
            is_day = (
                last_sunrise_time > last_tw_time
                if pd.notna(last_sunrise_time) and pd.notna(last_tw_time)
                else True
            )
            ts_series = futr_df["ds"]

            if is_day:
                daylight_secs = (
                    (tw_time - last_sunrise_time).total_seconds()
                    if pd.notna(last_sunrise_time)
                    else 12 * 3600
                )
                if daylight_secs > 0:
                    elapsed = (ts_series - last_sunrise_time).dt.total_seconds()
                    progress = np.clip(elapsed / daylight_secs, 0, 1)
                else:
                    progress = np.zeros(len(ts_series))
            else:
                night_secs = (
                    (next_sunrise_time - last_tw_time).total_seconds()
                    if pd.notna(last_tw_time)
                    else 12 * 3600
                )
                if night_secs > 0:
                    elapsed = (ts_series - last_tw_time).dt.total_seconds()
                    progress = 1 + np.clip(elapsed / night_secs, 0, 1)
                else:
                    progress = np.ones(len(ts_series))

            futr_df["twilight_cos"] = np.cos(np.pi * progress)

            all_hist_dfs.append(nf_recent)
            all_futr_dfs.append(futr_df)

            prediction_info.append({
                "unique_id": unique_id,
                "tw_idx": tw_idx,
                "window_idx": window_idx,
                "sample_times": sample_times,
                "dT_dt_tw_last": dT_dt_tw_last,
            })

    print(f"  Prepared {len(all_hist_dfs)} prediction requests ({len(twilight_info)} twilights × {len(FORECAST_WINDOWS)} windows)")

    # Combine batches
    combined_hist = pd.concat(all_hist_dfs, ignore_index=True)
    combined_futr = pd.concat(all_futr_dfs, ignore_index=True)
    print(f"  Combined hist: {combined_hist.shape}, futr: {combined_futr.shape}")

    # =========================================================================
    # 7. Single batched prediction
    # =========================================================================
    print("\n[7] Running batched prediction...")

    with suppress_output():
        fc = nf.predict(combined_hist, futr_df=combined_futr)

    fc = fc.reset_index()
    model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
    print(f"  Forecast shape: {fc.shape}")

    # Extract residuals per (twilight, window) combination
    fc_grouped = {uid: group for uid, group in fc.groupby("unique_id")}

    # Collect residuals per twilight (aggregate across windows)
    tw_residuals = {}  # tw_idx -> dict of residuals {res_t1: ..., res_t2: ..., etc.}

    for info in prediction_info:
        uid = info["unique_id"]
        tw_idx = info["tw_idx"]
        sample_times = info["sample_times"]  # e.g., [1, 2, 3] for window 0
        dT_dt_tw_last = info["dT_dt_tw_last"]

        if uid not in fc_grouped:
            continue

        uid_fc = fc_grouped[uid].sort_values("ds").reset_index(drop=True)

        # Initialize residuals dict for this twilight if not exists
        if tw_idx not in tw_residuals:
            tw_residuals[tw_idx] = {}

        # Extract residuals for each sample time in this window
        for i, t_h in enumerate(sample_times):
            step = i * SAMPLES_PER_HOUR
            if step >= len(uid_fc):
                continue

            delta_dT_dt_pred = uid_fc.iloc[step][model_col]
            rate_approx = delta_dT_dt_pred + dT_dt_tw_last
            dT_dt_actual = twilight_info[tw_idx][f"dT_dt_t{t_h}"]
            res = dT_dt_actual - rate_approx

            tw_residuals[tw_idx][f"res_t{t_h}"] = res

    # Build predictions dataframe - one row per twilight
    predictions = []
    for tw_idx, residuals in tw_residuals.items():
        info = twilight_info[tw_idx]

        # Collect t values and residuals for slope estimation
        t_values = []
        res_values = []
        for t_h in HOURS_FROM_LAST_TW:
            res_key = f"res_t{t_h}"
            if res_key in residuals and pd.notna(residuals[res_key]):
                t_values.append(t_h)
                res_values.append(residuals[res_key])

        if len(t_values) < 3:  # Need at least 3 points for fit
            continue

        # Compute slope estimate from residual pattern using linear regression
        slope_est, intercept = np.polyfit(t_values, res_values, 1)

        # Mean of res/t estimates
        slope_estimates = [r / t for r, t in zip(res_values, t_values) if t > 0]
        slope_mean = np.mean(slope_estimates) if slope_estimates else 0

        predictions.append({
            "tw_time": info["tw_time"],
            "dT_dt_tw_actual": info["dT_dt_tw_actual"],
            "dT_dt_tw_last": info["dT_dt_tw_last"],
            "slope_est": slope_est,
            "slope_mean": slope_mean,
            "intercept": intercept,
            **residuals,  # res_t1, res_t2, ..., res_t21
        })

    pred_df = pd.DataFrame(predictions).dropna()
    print(f"  Total predictions: {len(pred_df)} twilights with {len(HOURS_FROM_LAST_TW)} residuals each")

    # =========================================================================
    # 8. Evaluate baseline (persistence and NBEATSx-Oracle)
    # =========================================================================
    print("\n[8] Baseline evaluation...")

    y_actual = pred_df["dT_dt_tw_actual"].values
    y_pers = pred_df["dT_dt_tw_last"].values  # Persistence: previous twilight rate

    # NBEATSx-Oracle: use actual slope (cheating - for comparison)
    actual_slope = (pred_df["dT_dt_tw_actual"] - pred_df["dT_dt_tw_last"]) / 24
    y_oracle = pred_df["dT_dt_tw_last"] + actual_slope * 24  # = dT_dt_tw_actual

    rmse_pers = np.sqrt(mean_squared_error(y_actual, y_pers))
    mae_pers = mean_absolute_error(y_actual, y_pers)

    signal_std = np.std(y_actual)
    nrmse_pers = rmse_pers / signal_std * 100

    print(f"\n{'Model':<20} {'RMSE (°C/h)':<14} {'MAE (°C/h)':<14} {'NRMSE (%)':<12}")
    print("-" * 60)
    print(
        f"{'Persistence':<20} {rmse_pers:<14.4f} {mae_pers:<14.4f} {nrmse_pers:<12.1f}"
    )
    print(f"Signal std: {signal_std:.4f} °C/h")

    # =========================================================================
    # 9. Ridge correction using accumulated residuals
    # =========================================================================
    print("\n[9] Ridge correction with accumulated residuals...")

    # Sort by time for proper lagging
    pred_df = pred_df.sort_values("tw_time").reset_index(drop=True)

    # Add lagged twilight rates
    pred_df["dT_dt_tw_lag1"] = pred_df["dT_dt_tw_actual"].shift(1)
    pred_df["dT_dt_tw_lag2"] = pred_df["dT_dt_tw_actual"].shift(2)
    pred_df["dT_dt_tw_lag3"] = pred_df["dT_dt_tw_actual"].shift(3)

    # Add lagged slopes
    pred_df["actual_slope"] = (
        pred_df["dT_dt_tw_actual"] - pred_df["dT_dt_tw_last"]
    ) / 24
    pred_df["slope_lag1"] = pred_df["actual_slope"].shift(1)
    pred_df["slope_lag2"] = pred_df["actual_slope"].shift(2)

    # Add seasonal features
    pred_df["doy"] = pred_df["tw_time"].dt.dayofyear
    pred_df["doy_sin"] = np.sin(2 * np.pi * pred_df["doy"] / 365)
    pred_df["doy_cos"] = np.cos(2 * np.pi * pred_df["doy"] / 365)

    # Train/test split: odd/even days
    pred_df["day_num"] = (pred_df["tw_time"] - pred_df["tw_time"].min()).dt.days

    # Feature columns - residuals at t=1 through t=21 + slope estimates
    res_cols = [
        f"res_t{t}" for t in HOURS_FROM_LAST_TW
    ]  # res_t1, res_t2, ..., res_t21
    feature_cols = (
        res_cols  # Residuals at each time point
        + ["slope_est", "slope_mean", "intercept"]  # Slope estimates from linear fit
        + ["dT_dt_tw_last"]  # Known baseline
        + ["dT_dt_tw_lag1", "dT_dt_tw_lag2", "dT_dt_tw_lag3"]  # Lagged twilight rates
        + ["slope_lag1", "slope_lag2"]  # Lagged slopes
        + ["doy_sin", "doy_cos"]  # Seasonal
    )

    # Drop rows with NaN features
    ridge_df = pred_df.dropna(subset=feature_cols + ["dT_dt_tw_actual"])
    train_ridge = ridge_df[ridge_df["day_num"] % 2 == 1]
    test_ridge = ridge_df[ridge_df["day_num"] % 2 == 0]

    print(f"  Training samples: {len(train_ridge)}, Test samples: {len(test_ridge)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Residual features: {res_cols}")

    # Train Ridge
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_ridge[feature_cols].values)
    X_test = scaler.transform(test_ridge[feature_cols].values)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, train_ridge["dT_dt_tw_actual"].values)

    # Predict
    dT_dt_ridge = ridge.predict(X_test)

    # Evaluate Ridge
    y_actual_ridge = test_ridge["dT_dt_tw_actual"].values
    y_pers_ridge = test_ridge["dT_dt_tw_last"].values

    rmse_ridge = np.sqrt(mean_squared_error(y_actual_ridge, dT_dt_ridge))
    mae_ridge = mean_absolute_error(y_actual_ridge, dT_dt_ridge)
    nrmse_ridge = rmse_ridge / signal_std * 100

    rmse_pers_test = np.sqrt(mean_squared_error(y_actual_ridge, y_pers_ridge))
    nrmse_pers_test = rmse_pers_test / signal_std * 100

    improvement = (rmse_pers_test - rmse_ridge) / rmse_pers_test * 100

    print(
        f"\n{'Model':<20} {'RMSE (°C/h)':<14} {'MAE (°C/h)':<14} {'NRMSE (%)':<12} {'Improv':<10}"
    )
    print("-" * 70)
    print(
        f"{'Persistence':<20} {rmse_pers_test:<14.4f} {'-':<14} {nrmse_pers_test:<12.1f} {'-':<10}"
    )
    print(
        f"{'NBEATSx-Ridge':<20} {rmse_ridge:<14.4f} {mae_ridge:<14.4f} {nrmse_ridge:<12.1f} {improvement:<10.1f}%"
    )
    print("-" * 70)

    # Feature importance (Ridge coefficients)
    print("\n  Feature coefficients (scaled):")
    coef_df = pd.DataFrame(
        {"feature": feature_cols, "coefficient": ridge.coef_}
    ).sort_values("coefficient", key=abs, ascending=False)
    for _, row in coef_df.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['coefficient']:>8.4f}")

    # Store Ridge predictions
    test_ridge = test_ridge.copy()
    test_ridge["dT_dt_ridge"] = dT_dt_ridge

    # =========================================================================
    # 10. Plot results (6 panels)
    # =========================================================================
    print("\n[10] Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Error distribution
    ax1 = axes[0, 0]
    ax1.hist(
        dT_dt_ridge - y_actual_ridge,
        bins=20,
        alpha=0.7,
        color="#e74c3c",
        edgecolor="black",
        label="NBEATSx-Ridge",
    )
    ax1.hist(
        y_pers_ridge - y_actual_ridge,
        bins=20,
        alpha=0.5,
        color="gray",
        edgecolor="black",
        label="Persistence",
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Error (°C/h)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Error Distribution\nNBEATSx-Ridge: {nrmse_ridge:.1f}% NRMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Predicted vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(
        y_actual_ridge,
        dT_dt_ridge,
        alpha=0.6,
        s=40,
        c="#e74c3c",
        edgecolor="white",
        linewidth=0.5,
    )
    lims = [y_actual_ridge.min() - 0.2, y_actual_ridge.max() + 0.2]
    ax2.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax2.set_xlabel("Actual dT/dt at twilight (°C/h)")
    ax2.set_ylabel("Predicted dT/dt (°C/h)")
    ax2.set_title(f"NBEATSx-Ridge vs Actual (NRMSE: {nrmse_ridge:.1f}%)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Residual pattern (res vs t) - individual lines
    ax3 = axes[0, 2]
    sample_indices = test_ridge.index[:50]
    for idx in sample_indices:
        row = test_ridge.loc[idx]
        t_vals = HOURS_FROM_LAST_TW  # t = 1, 2, ..., 21
        res_vals = [row.get(f"res_t{t}", np.nan) for t in t_vals]
        # Filter out NaN
        valid = [(t, r) for t, r in zip(t_vals, res_vals) if pd.notna(r)]
        if valid:
            ts, rs = zip(*valid)
            ax3.plot(ts, rs, "o-", alpha=0.3, markersize=2)
    ax3.set_xlabel("Hours since last twilight (t)")
    ax3.set_ylabel("Residual (°C/h)")
    ax3.set_title("Individual Residual Patterns: res(t) ≈ slope × t")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Flat baseline residual (scatter of all res vs t)
    ax4 = axes[1, 0]
    all_t = []
    all_res = []
    for _, row in test_ridge.iterrows():
        for t_h in HOURS_FROM_LAST_TW:
            res_val = row.get(f"res_t{t_h}", np.nan)
            if pd.notna(res_val):
                all_t.append(t_h)
                all_res.append(res_val)
    ax4.scatter(all_t, all_res, alpha=0.2, s=5, c="#3498db")
    # Add binned means
    t_unique = sorted(set(all_t))
    res_means = [
        np.mean([r for t, r in zip(all_t, all_res) if t == tu]) for tu in t_unique
    ]
    ax4.plot(t_unique, res_means, "ro-", linewidth=2, markersize=6, label="Mean")
    # Linear fit
    slope_fit, intercept_fit = np.polyfit(all_t, all_res, 1)
    t_line = np.array([min(all_t), max(all_t)])
    ax4.plot(
        t_line,
        slope_fit * t_line + intercept_fit,
        "k--",
        linewidth=2,
        label=f"Linear fit: {slope_fit:.4f}×t + {intercept_fit:.2f}",
    )
    ax4.set_xlabel("Hours since last twilight (t)")
    ax4.set_ylabel("Flat baseline residual (°C/h)")
    ax4.set_title(
        f"Residual vs Time (Flat Baseline)\nres ≈ slope × t, {len(all_t)} points"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: CDF of error
    ax5 = axes[1, 1]
    for errors, label, color in [
        (dT_dt_ridge - y_actual_ridge, "NBEATSx-Ridge", "#e74c3c"),
        (y_pers_ridge - y_actual_ridge, "Persistence", "gray"),
    ]:
        sorted_err = np.sort(np.abs(errors) / signal_std * 100)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax5.plot(sorted_err, cdf, label=label, color=color, linewidth=2)
    ax5.axvline(100, color="orange", linestyle="--", alpha=0.7, label="100% (1σ)")
    ax5.set_xlabel("Normalized Error (% of signal σ)")
    ax5.set_ylabel("Cumulative Probability")
    ax5.set_title("CDF of Normalized Error")
    ax5.legend(loc="lower right")
    ax5.set_xlim(0, 300)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Slope estimate vs actual slope
    ax6 = axes[1, 2]
    actual_slopes = (test_ridge["dT_dt_tw_actual"] - test_ridge["dT_dt_tw_last"]) / 24
    ax6.scatter(actual_slopes, test_ridge["slope_est"], alpha=0.6, s=40, c="#2ecc71")
    lims_slope = [actual_slopes.min() - 0.01, actual_slopes.max() + 0.01]
    ax6.plot(lims_slope, lims_slope, "k--", linewidth=1, label="1:1 line")
    ax6.set_xlabel("Actual slope (dT_dt_tw - dT_dt_tw_last) / 24")
    ax6.set_ylabel("Estimated slope from residuals")
    corr = np.corrcoef(actual_slopes, test_ridge["slope_est"])[0, 1]
    ax6.set_title(f"Slope Recovery (corr={corr:.3f})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    figures_dir = RESULTS_PATH / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_file = figures_dir / "rate_forecast_delta_results.png"
    plt.savefig(fig_file, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {fig_file}")
    plt.show()

    # =========================================================================
    # 11. Save results
    # =========================================================================
    print("\n[11] Saving results...")

    pred_df.to_csv(RESULTS_PATH / "rate_predictions_delta_3h.csv", index=False)

    metrics = pd.DataFrame(
        [
            {
                "Model": "Persistence",
                "RMSE": rmse_pers_test,
                "MAE": mean_absolute_error(y_actual_ridge, y_pers_ridge),
                "NRMSE_pct": nrmse_pers_test,
                "Improvement_pct": 0.0,
            },
            {
                "Model": "NBEATSx-Ridge (Delta)",
                "RMSE": rmse_ridge,
                "MAE": mae_ridge,
                "NRMSE_pct": nrmse_ridge,
                "Improvement_pct": improvement,
            },
        ]
    )
    metrics.to_csv(RESULTS_PATH / "rate_metrics_delta_3h.csv", index=False)
    print(f"  Metrics saved to: {RESULTS_PATH / 'rate_metrics_delta_3h.csv'}")

    test_ridge.to_csv(RESULTS_PATH / "rate_predictions_ridge_delta_3h.csv", index=False)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
