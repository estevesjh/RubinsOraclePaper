"""Ridge correction of NBEATSx-Diff residuals at twilight events.

Simple approach: predict the residual at each twilight from the residuals
of the N previous twilight events. Train on pre-2025, test on 2025+.

Features: res_prev_1, res_prev_2, ..., res_prev_N (lagged twilight residuals)

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/ridge_twilight.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
from config import RESULTS_PATH, TEST_START_DATE, RIDGE_ALPHA

N_LAGS = 5  # number of previous twilight residuals to use


def main():
    print("=" * 70)
    print("RIDGE CORRECTION ON TWILIGHT RESIDUALS")
    print(f"Features: {N_LAGS} lagged twilight residuals")
    print("=" * 70)

    # Load NBEATSx predictions
    results_file = RESULTS_PATH / "paper_results_diff.csv"
    df = pd.read_csv(results_file)
    df["twilight_time"] = pd.to_datetime(df["twilight_time"])
    nb = df[df["model"] == "NBEATSx-Diff"].copy()

    # Work per lead time
    lead_times = sorted(nb["lead_time_hours"].unique())
    print(f"  Lead times: {len(lead_times)} ({lead_times[0]:.1f}h to {lead_times[-1]:.1f}h)")
    print(f"  Total predictions: {len(nb)}")

    # For each lead time, build the residual series and train Ridge
    all_corrected = []

    print(f"\n{'Lead(h)':>8} {'RMSE_before':>11} {'RMSE_after':>10} {'Improve':>8} {'N_train':>8} {'N_test':>7}")
    print("-" * 60)

    for lead_h in lead_times:
        sub = nb[nb["lead_time_hours"] == lead_h].sort_values("twilight_time").copy()
        sub["residual"] = sub["actual_temp"] - sub["forecast_temp"]

        # Build lagged residual features
        for lag in range(1, N_LAGS + 1):
            sub[f"res_prev_{lag}"] = sub["residual"].shift(lag)

        # Drop rows without enough history
        sub = sub.dropna(subset=[f"res_prev_{N_LAGS}"])

        if len(sub) < 20:
            continue

        # Even/odd split: train on even-indexed events, test on odd (preserves seasons)
        sub = sub.reset_index(drop=True)
        train = sub.iloc[::2]   # even
        test = sub.iloc[1::2]   # odd

        if len(train) < 10 or len(test) < 10:
            continue

        feat_cols = [f"res_prev_{lag}" for lag in range(1, N_LAGS + 1)]

        # Train Ridge
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feat_cols].values)
        y_train = train["residual"].values

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train, y_train)

        # Predict on test
        X_test = scaler.transform(test[feat_cols].values)
        correction = ridge.predict(X_test)

        # Corrected forecast
        rmse_before = np.sqrt((test["residual"].values ** 2).mean())
        rmse_after = np.sqrt(((test["residual"].values - correction) ** 2).mean())
        improve = (rmse_before - rmse_after) / rmse_before * 100

        if lead_h in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 11.5]:
            print(f"{lead_h:8.1f} {rmse_before:11.3f} {rmse_after:10.3f} {improve:+7.1f}% {len(train):8d} {len(test):7d}")

        # Store corrected results
        corrected = test.copy()
        corrected["forecast_temp"] = corrected["forecast_temp"] + correction
        corrected["error"] = corrected["actual_temp"] - corrected["forecast_temp"]
        corrected["model"] = "NBEATSx-Diff-Ridge"
        all_corrected.append(corrected[["twilight_time", "forecast_time", "lead_time_hours",
                                         "actual_temp", "model", "forecast_temp", "error"]])

    corrected_df = pd.concat(all_corrected, ignore_index=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NBEATSx-Diff-Ridge")
    print("=" * 70)
    print(f"{'Lead(h)':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'<1C':>6} {'N':>5}")
    print("-" * 45)
    for lead_h in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 11.5]:
        sub = corrected_df[corrected_df["lead_time_hours"] == lead_h]
        if len(sub) > 0:
            rmse = np.sqrt((sub["error"] ** 2).mean())
            mae = sub["error"].abs().mean()
            bias = sub["error"].mean()
            lt1 = (sub["error"].abs() < 1.0).mean() * 100
            print(f"{lead_h:8.1f} {rmse:8.3f} {mae:8.3f} {bias:+8.3f} {lt1:5.1f}% {len(sub):5d}")

    # Save
    output_file = RESULTS_PATH / "paper_results_ridge.csv"
    corrected_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Also save combined (NBEATSx-Diff + Ridge)
    combined = pd.concat([nb[["twilight_time", "forecast_time", "lead_time_hours",
                               "actual_temp", "model", "forecast_temp", "error"]],
                          corrected_df], ignore_index=True)
    combined_file = RESULTS_PATH / "paper_results_all.csv"
    combined.to_csv(combined_file, index=False)
    print(f"Combined saved to {combined_file}")


if __name__ == "__main__":
    main()
