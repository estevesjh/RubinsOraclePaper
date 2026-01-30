"""Evaluate hourly temperature trajectory forecasts with Ridge correction.

This script evaluates the full temperature trajectory (not just twilight),
where lead_time = time between forecast issue and predicted time.

Two-stage approach:
1. NBEATSx predicts delta_T (with exponential gate)
2. Ridge corrects the trajectory residuals using features at forecast time

IMPORTANT: No future data leakage - Ridge only uses features available at forecast time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import (
    RESULTS_PATH,
    TWILIGHT_OFFSET_FILE,
    TWILIGHT_OFFSET_INSAMPLE_FILE,
    FIGURES_PATH,
    GATE_TAU,
    TRAINING_MODE,
    RIDGE_ALPHA,
)


# =============================================================================
# Configuration
# =============================================================================

# Lead times to evaluate (hours from forecast issue to prediction target)
# Regular 0.5h spacing from 0.5 to 12.0 hours
LEAD_TIMES_HOURS = [h / 2 for h in range(1, 25)]  # [0.5, 1.0, 1.5, ..., 11.5, 12.0]

# Output files
OUTPUT_FILE = RESULTS_PATH / "trajectory_results.csv"
SLOPE_FILE = RESULTS_PATH / "slope_predictions_3h.csv"


# =============================================================================
# Gate Function
# =============================================================================


def apply_exp_gate(delta_T_pred: np.ndarray, hours_to_tw: np.ndarray, tau: float) -> np.ndarray:
    """Apply exponential gate: gate = 1 - exp(-|hours_to_tw| / tau)"""
    gate = 1 - np.exp(-np.abs(hours_to_tw) / tau)
    return delta_T_pred * gate


# =============================================================================
# Data Loading
# =============================================================================


def load_predictions():
    """Load NBEATSx predictions for training and test."""
    print("Loading predictions...")

    if TRAINING_MODE == "operational":
        # Load in-sample predictions for training
        df_train = pd.read_csv(TWILIGHT_OFFSET_INSAMPLE_FILE)
        df_train["_is_train"] = True
        print(f"  Training data (in-sample): {len(df_train)} rows")

        # Load 2025 test predictions
        df_test = pd.read_csv(TWILIGHT_OFFSET_FILE)
        df_test["_is_train"] = False
        print(f"  Test data (2025): {len(df_test)} rows")

        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        # Paper mode: single file, split by odd/even days
        df = pd.read_csv(TWILIGHT_OFFSET_FILE)
        df["day_num"] = (pd.to_datetime(df["tw_date"]) - pd.to_datetime(df["tw_date"]).min()).dt.days
        df["_is_train"] = df["day_num"] % 2 == 1

    # Convert datetime columns
    for col in ["tw_time", "tw_date", "forecast_time", "target_time"]:
        df[col] = pd.to_datetime(df[col])

    print(f"  Total rows: {len(df)}")
    print(f"  Twilights: {df['tw_time'].nunique()}")

    return df


def load_slope_predictions():
    """Load Ridge slope predictions."""
    if not SLOPE_FILE.exists():
        print(f"  Warning: Slope file not found: {SLOPE_FILE}")
        return None

    slope_df = pd.read_csv(SLOPE_FILE)
    slope_df["tw_time"] = pd.to_datetime(slope_df["tw_time"])
    print(f"  Loaded {len(slope_df)} slope predictions")
    return slope_df


# =============================================================================
# Feature Engineering (No Future Leakage)
# =============================================================================


def build_trajectory_features(df, slope_df=None):
    """
    Build features for trajectory Ridge correction.

    IMPORTANT: All features must be available at forecast_time (no future leakage).

    Features computed at forecast_time:
    - Residual at forecast time (res_at_forecast)
    - Hours to twilight at forecast time
    - Hours to twilight at target time
    - Lead time (target - forecast)
    - Rate features (computed from historical data)
    - Seasonal features (doy, twilight_cos)
    """
    df = df.copy()

    # Compute hours_to_tw for target time
    df["hours_to_tw"] = df["h_from_tw"] - 24

    # Compute hours_to_tw at forecast time
    # h_from_tw is for target_time, so we need to adjust
    df["lead_time_hours"] = (df["target_time"] - df["forecast_time"]).dt.total_seconds() / 3600
    df["hours_to_tw_at_forecast"] = df["hours_to_tw"] - df["lead_time_hours"]

    # Apply gate to delta_T
    if GATE_TAU is not None:
        gate = 1 - np.exp(-np.abs(df["hours_to_tw"]) / GATE_TAU)
        df["delta_T_gated"] = df["delta_T_pred"] * gate
    else:
        df["delta_T_gated"] = df["delta_T_pred"]

    # Base prediction (before Ridge correction)
    df["T_pred_base"] = df["T_tw_last"] + df["delta_T_gated"]

    # Add slope correction if available
    if slope_df is not None:
        df = df.merge(slope_df[["tw_time", "slope"]], on="tw_time", how="left")
        # Cap h_from_tw at 24 - slope is only valid for [0, 24] interval
        h_capped = np.minimum(df["h_from_tw"], 24)
        slope_correction = (df["slope"].fillna(0) / 24) * h_capped
        df["T_pred_base"] = df["T_pred_base"] + slope_correction

    # Baseline residual (this is what Ridge will correct)
    df["res_base"] = df["temp_actual"] - df["T_pred_base"]

    # -------------------------------------------------------------------------
    # Features available at forecast_time (NO FUTURE LEAKAGE)
    # -------------------------------------------------------------------------

    # 1. Residual at forecast time
    # We need to find the residual at (forecast_time, forecast_time) - i.e., nowcast
    # This is approximated by the residual where target_time == forecast_time
    # For simplicity, use res_tw which is the residual pattern

    # 2. Seasonal features (known at any time)
    df["doy"] = df["tw_time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    df["tw_hour"] = df["tw_time"].dt.hour + df["tw_time"].dt.minute / 60
    df["twilight_sin"] = np.sin(2 * np.pi * df["tw_hour"] / 24)
    df["twilight_cos"] = np.cos(2 * np.pi * df["tw_hour"] / 24)

    # 3. Lead time features
    df["lead_time_norm"] = df["lead_time_hours"] / 12.0  # Normalize to [0, 1]

    # 4. Hours to twilight features (target time)
    df["hours_to_tw_norm"] = df["hours_to_tw"] / 12.0

    # 5. Rate features (available at forecast time - from historical patterns)
    # These are already in the data from train.py

    # 6. Temperature features available at forecast time
    df["temp_at_last_tw"] = df["T_tw_last"]

    # 7. Interaction features
    df["lead_x_doy_cos"] = df["lead_time_norm"] * df["doy_cos"]
    df["hours_to_tw_x_doy_cos"] = df["hours_to_tw_norm"] * df["doy_cos"]

    # 8. Lagged residuals from previous twilights
    # For each h_from_tw position, get residual from same position in previous days
    df = df.sort_values(["tw_time", "h_from_tw"])

    for lag in [1, 2, 3, 4, 5]:
        df[f"res_lag_{lag}"] = df.groupby("h_from_tw")["res_base"].shift(lag)

    # 9. Residual at forecast time (nowcast error)
    # Look up res_base where target_time == forecast_time for same twilight
    nowcast = df[df["target_time"] == df["forecast_time"]][["tw_time", "forecast_time", "res_base"]]
    nowcast = nowcast.rename(columns={"res_base": "res_at_forecast"})
    df = df.merge(nowcast, on=["tw_time", "forecast_time"], how="left")

    return df


# =============================================================================
# Ridge Training for Trajectory Correction
# =============================================================================


def train_trajectory_ridge(df, lead_time):
    """
    Train Ridge model to correct trajectory residuals at a specific lead time.

    Args:
        df: DataFrame with features and res_base
        lead_time: Lead time in hours to train for

    Returns:
        Trained Ridge model, scaler, and feature columns
    """
    # Filter to this lead time (with tolerance)
    mask = np.abs(df["lead_time_hours"] - lead_time) < 0.25
    df_lt = df[mask].copy()

    if len(df_lt) == 0:
        return None

    # Feature columns
    feature_cols = [
        "res_at_forecast",  # Residual at forecast time (nowcast error)
        "res_lag_1",        # Residual from same h_from_tw, previous twilight
        "res_lag_2",        # Residual from same h_from_tw, 2 twilights ago
        "res_lag_3",        # 3 twilights ago
        "res_lag_4",        # 4 twilights ago
        "res_lag_5",        # 5 twilights ago
        "delta_T_gated",    # Gated NBEATSx prediction
        "T_pred_base",      # Base temperature prediction
        "doy_cos",          # Seasonality
        "trend_2h",         # 2-hour temperature trend
    ]

    # Check which features exist
    feature_cols = [c for c in feature_cols if c in df_lt.columns]

    # Target: baseline residual
    target_col = "res_base"

    # Train/test split
    train = df_lt[df_lt["_is_train"] == True].dropna(subset=feature_cols + [target_col])
    test = df_lt[df_lt["_is_train"] == False].dropna(subset=feature_cols + [target_col])

    if len(train) < 50 or len(test) < 50:
        return None

    # Train Ridge
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Corrected temperature
    T_corrected_test = test["T_pred_base"].values + y_pred_test
    T_error_corrected = T_corrected_test - test["temp_actual"].values

    # Baseline error (without Ridge)
    T_error_base = test["T_pred_base"].values - test["temp_actual"].values

    rmse_base = np.sqrt((T_error_base ** 2).mean())
    rmse_corrected = np.sqrt((T_error_corrected ** 2).mean())

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "rmse_base": rmse_base,
        "rmse_corrected": rmse_corrected,
        "n_train": len(train),
        "n_test": len(test),
        "test_df": test,
        "y_pred_test": y_pred_test,
    }


def evaluate_trajectory_with_ridge(df):
    """
    Evaluate trajectory predictions with Ridge correction at each lead time.
    """
    print("\n" + "=" * 70)
    print("TRAINING RIDGE FOR TRAJECTORY CORRECTION")
    print("=" * 70)

    results = []

    print(f"\n{'Lead':<8} {'N_train':<10} {'N_test':<10} {'RMSE_base':<12} {'RMSE_Ridge':<12} {'Δ':<10}")
    print("-" * 70)

    for lead_time in LEAD_TIMES_HOURS:
        result = train_trajectory_ridge(df, lead_time)

        if result is None:
            print(f"{lead_time:<8.1f} Skipped (insufficient data)")
            continue

        delta = result["rmse_corrected"] - result["rmse_base"]
        sign = "+" if delta > 0 else ""

        print(f"{lead_time:<8.1f} {result['n_train']:<10} {result['n_test']:<10} "
              f"{result['rmse_base']:<12.3f} {result['rmse_corrected']:<12.3f} {sign}{delta:<9.3f}")

        # Compute additional metrics on test set
        test_df = result["test_df"]
        T_corrected = test_df["T_pred_base"].values + result["y_pred_test"]
        T_error = T_corrected - test_df["temp_actual"].values

        results.append({
            "lead_time_hours": lead_time,
            "rmse_base": result["rmse_base"],
            "rmse_corrected": result["rmse_corrected"],
            "mae": np.abs(T_error).mean(),
            "bias": T_error.mean(),
            "within_1c_pct": (np.abs(T_error) < 1.0).mean() * 100,
            "n_test": result["n_test"],
        })

    return pd.DataFrame(results)


# =============================================================================
# Evaluation by Hours to Twilight
# =============================================================================


def evaluate_by_hour_to_twilight(df):
    """
    Evaluate predictions grouped by hours to twilight.
    Uses simple baseline (no per-lead-time Ridge).

    Note: Expects df to already have features from build_trajectory_features().
    """
    df = df.copy()
    df["hour_to_tw_bin"] = df["hours_to_tw"].round().astype(int)

    # Filter to test data
    df_test = df[df["_is_train"] == False]

    results = []

    for h_to_tw in sorted(df_test["hour_to_tw_bin"].unique()):
        if h_to_tw < -12 or h_to_tw > 6:
            continue

        df_h = df_test[df_test["hour_to_tw_bin"] == h_to_tw]

        if len(df_h) < 100:
            continue

        # Baseline error
        error = df_h["T_pred_base"] - df_h["temp_actual"]
        rmse = np.sqrt((error ** 2).mean())
        mae = error.abs().mean()
        bias = error.mean()

        results.append({
            "hours_to_twilight": h_to_tw,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "n_samples": len(df_h),
        })

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================


def plot_trajectory_metrics(metrics_ridge, metrics_by_hour, output_path):
    """Create visualization of trajectory forecast metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: RMSE by lead time (base vs Ridge)
    ax = axes[0, 0]
    ax.plot(metrics_ridge["lead_time_hours"], metrics_ridge["rmse_base"],
            "ko--", markersize=6, linewidth=1.5, label="Baseline", alpha=0.7)
    ax.plot(metrics_ridge["lead_time_hours"], metrics_ridge["rmse_corrected"],
            "bo-", markersize=6, linewidth=2, label="Ridge corrected")
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Temperature RMSE by Lead Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 13)

    # Plot 2: RMSE by hours to twilight
    ax = axes[0, 1]
    ax.plot(metrics_by_hour["hours_to_twilight"], metrics_by_hour["rmse"],
            "go-", markersize=6, linewidth=2)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Twilight")
    ax.set_xlabel("Hours to Twilight")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Temperature RMSE by Hours to Twilight (Baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Bias by lead time
    ax = axes[1, 0]
    ax.plot(metrics_ridge["lead_time_hours"], metrics_ridge["bias"],
            "mo-", markersize=6, linewidth=2)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("Bias (°C)")
    ax.set_title("Temperature Bias by Lead Time (Ridge)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 13)

    # Plot 4: % within 1°C by lead time
    ax = axes[1, 1]
    ax.plot(metrics_ridge["lead_time_hours"], metrics_ridge["within_1c_pct"],
            "co-", markersize=6, linewidth=2)
    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("% within 1°C")
    ax.set_title("Forecast Accuracy by Lead Time (Ridge)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 105)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure to: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main evaluation."""
    print("=" * 70)
    print("TRAJECTORY FORECAST EVALUATION WITH RIDGE CORRECTION")
    print(f"Gate: τ={GATE_TAU}h" if GATE_TAU else "Gate: disabled")
    print(f"Training mode: {TRAINING_MODE}")
    print("=" * 70)

    # Load data
    df = load_predictions()

    # Load slope predictions
    print("\nLoading Ridge slope predictions...")
    slope_df = load_slope_predictions()

    # Build features
    print("\nBuilding trajectory features...")
    df = build_trajectory_features(df, slope_df)

    # Evaluate with Ridge correction
    metrics_ridge = evaluate_trajectory_with_ridge(df)

    # Evaluate by hours to twilight (baseline)
    print("\n" + "=" * 70)
    print("BASELINE METRICS BY HOURS TO TWILIGHT")
    print("=" * 70)

    metrics_by_hour = evaluate_by_hour_to_twilight(df)

    print(f"\n{'Hours to TW':<15} {'RMSE':>10} {'MAE':>10} {'Bias':>10} {'N':>10}")
    print("-" * 60)
    for _, row in metrics_by_hour.iterrows():
        print(f"{row['hours_to_twilight']:>12}h {row['rmse']:>10.3f} "
              f"{row['mae']:>10.3f} {row['bias']:>10.3f} {int(row['n_samples']):>10}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RIDGE-CORRECTED TRAJECTORY")
    print("=" * 70)

    print(f"\n{'Lead Time':<12} {'RMSE':>10} {'MAE':>10} {'Bias':>10} {'<1°C':>10}")
    print("-" * 60)
    for _, row in metrics_ridge.iterrows():
        print(f"{row['lead_time_hours']:>10.1f}h {row['rmse_corrected']:>10.3f} "
              f"{row['mae']:>10.3f} {row['bias']:>10.3f} "
              f"{row['within_1c_pct']:>9.1f}%")

    # Save results
    metrics_ridge.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to: {OUTPUT_FILE}")

    # Create visualization
    print("\nGenerating visualization...")
    plot_trajectory_metrics(
        metrics_ridge,
        metrics_by_hour,
        FIGURES_PATH / "trajectory_forecast_metrics.png"
    )

    # Overall improvement
    print("\n" + "=" * 70)
    print("OVERALL IMPROVEMENT FROM RIDGE CORRECTION")
    print("=" * 70)

    avg_base = metrics_ridge["rmse_base"].mean()
    avg_corrected = metrics_ridge["rmse_corrected"].mean()
    improvement = (avg_base - avg_corrected) / avg_base * 100

    print(f"\nAverage RMSE (baseline):  {avg_base:.3f}°C")
    print(f"Average RMSE (Ridge):     {avg_corrected:.3f}°C")
    print(f"Improvement:              {improvement:.1f}%")


if __name__ == "__main__":
    main()
