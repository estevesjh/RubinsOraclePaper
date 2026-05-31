"""Ridge correction using ALL ~800 twilight events (in-sample + test).

Train Ridge on even days, test on odd days (season-balanced).
Uses in-sample NBEATSx predictions for training twilights.

Usage:
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/ridge_full.py
"""

import os
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_PATH, RESULTS_PATH, MODEL_CACHE_PATH,
    TEST_START_DATE, HALFDAY_LAG_STEPS, SOLAR_GRID_FREQ,
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, STEPS_PER_DAY,
    SOLAR_GRID_STEP, RIDGE_ALPHA,
)
from run import (
    load_and_prepare, find_twilight_targets, get_hist_futr_exog,
    suppress_stdout, train_nbeats_diff, lead_hours_to_steps,
)

N_LAGS = 5
LEAD_H = 3.0  # primary lead time for Ridge


def generate_predictions_all_events(model, grid, tw_events, cfg):
    """Generate NBEATSx predictions for ALL twilight events (train + test)."""
    hist_exog, futr_exog = get_hist_futr_exog(cfg)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    y_arr = grid["y"].values
    ds_real_arr = grid["ds_real"].values
    offset_steps = lead_hours_to_steps(LEAD_H)

    requests = []
    for ev_i, (_, ev) in enumerate(tw_events.iterrows()):
        target_idx = ev["grid_idx"]
        issue_idx = target_idx - offset_steps
        if issue_idx < NBEATS_INPUT_SIZE:
            continue
        if (target_idx - issue_idx) >= NBEATS_HORIZON:
            continue
        anchor_idx = target_idx - HALFDAY_LAG_STEPS
        if anchor_idx < 0:
            continue
        requests.append({
            "ev_i": ev_i, "target_idx": target_idx,
            "issue_idx": issue_idx, "anchor_idx": anchor_idx,
            "actual_temp": ev["y_actual"], "tw_real_time": ev["ds_real"],
        })

    print(f"  Generating predictions for {len(requests)} twilight events...")

    BATCH_SIZE = 64
    results = []
    n_batches = (len(requests) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(n_batches):
        if batch_i % 20 == 0:
            print(f"    Batch {batch_i}/{n_batches}...")

        batch = requests[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        all_hist, all_futr, batch_meta = [], [], []

        for req in batch:
            issue_idx = req["issue_idx"]
            hist_start = issue_idx - NBEATS_INPUT_SIZE
            futr_end = issue_idx + NBEATS_HORIZON
            if futr_end > len(grid):
                continue

            uid = f"r_{batch_i}_{req['ev_i']}"
            hist_df = grid.iloc[hist_start:issue_idx][["ds", "D"] + all_exog].copy()
            if hist_df[all_exog].isna().any().any():
                continue
            hist_df["y"] = hist_df["D"]
            hist_df["unique_id"] = uid

            futr_df = grid.iloc[issue_idx:futr_end][["ds"] + futr_exog].copy()
            futr_df["unique_id"] = uid

            all_hist.append(hist_df)
            all_futr.append(futr_df)
            batch_meta.append((uid, req))

        if not all_hist:
            continue

        combined_hist = pd.concat(all_hist, ignore_index=True)
        combined_futr = pd.concat(all_futr, ignore_index=True)

        with suppress_stdout():
            fc = model.predict(combined_hist, futr_df=combined_futr)

        fc = fc.reset_index()
        model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]

        for uid, req in batch_meta:
            uid_fc = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            if len(uid_fc) == 0:
                continue
            pred_step = req["target_idx"] - req["issue_idx"]
            if pred_step >= len(uid_fc):
                continue

            D_pred = uid_fc[model_col].iloc[pred_step]
            anchor_temp = y_arr[req["anchor_idx"]]
            if np.isnan(anchor_temp):
                continue

            T_pred = D_pred + anchor_temp
            results.append({
                "tw_time": req["tw_real_time"],
                "target_idx": req["target_idx"],
                "issue_idx": req["issue_idx"],
                "actual_temp": req["actual_temp"],
                "forecast_temp": T_pred,
                "residual": req["actual_temp"] - T_pred,
            })

    return pd.DataFrame(results)


def build_ridge_features(pred_df, grid, tw_events):
    """Build rich feature matrix at issue time for each twilight."""
    y_arr = grid["y"].values
    offset_steps = lead_hours_to_steps(LEAD_H)

    features = []
    for _, row in pred_df.iterrows():
        tw_time = row["tw_time"]
        target_idx = row["target_idx"]
        issue_idx = row["issue_idx"]

        if issue_idx < 96 or issue_idx >= len(grid):
            continue

        feat = {"residual": row["residual"], "tw_time": tw_time}

        # Features at issue time
        for col in ["temp_raw", "temp_last_sunrise", "max_minus_sunrise",
                    "temp_solar_noon", "trend_2h", "temp_trend_3d",
                    "rate_sunrise_to_midday", "rate_twilight_to_midnight",
                    "rate_midnight_to_sunrise", "last_max_temp", "last_min_temp",
                    "max_min", "doy_sin", "doy_cos"]:
            if col in grid.columns:
                feat[col] = grid[col].iloc[issue_idx]

        # Day-to-day max changes
        if "last_max_temp" in grid.columns:
            max_today = grid["last_max_temp"].iloc[issue_idx]
            idx_1d = issue_idx - 48
            idx_3d = issue_idx - 48 * 3
            if idx_1d >= 0:
                feat["dmax_1d"] = max_today - grid["last_max_temp"].iloc[idx_1d]
            if idx_3d >= 0:
                feat["dmax_3d"] = max_today - grid["last_max_temp"].iloc[idx_3d]

        # Day-to-day min changes
        if "last_min_temp" in grid.columns:
            min_today = grid["last_min_temp"].iloc[issue_idx]
            idx_1d = issue_idx - 48
            if idx_1d >= 0:
                feat["dmin_1d"] = min_today - grid["last_min_temp"].iloc[idx_1d]

        # Range change
        if "max_min" in grid.columns:
            idx_1d = issue_idx - 48
            if idx_1d >= 0:
                feat["drange_1d"] = grid["max_min"].iloc[issue_idx] - grid["max_min"].iloc[idx_1d]

        # Lagged residuals from previous twilights
        prev = pred_df[pred_df["tw_time"] < tw_time].tail(N_LAGS)
        for lag_i in range(N_LAGS):
            if lag_i < len(prev):
                feat[f"res_prev_{lag_i+1}"] = prev.iloc[-(lag_i+1)]["residual"]
            else:
                feat[f"res_prev_{lag_i+1}"] = np.nan

        features.append(feat)

    return pd.DataFrame(features)


def main():
    print("=" * 70)
    print("RIDGE CORRECTION — ALL ~800 TWILIGHTS")
    print("Even/odd split, rich features")
    print("=" * 70)

    grid, cfg = load_and_prepare()

    tw_events = find_twilight_targets(grid)
    print(f"  Total twilight events: {len(tw_events)}")

    # Load cached model
    print("\nLoading cached model...")
    model = train_nbeats_diff(grid, cfg, target_col="D", cache_name="NBEATSx_diff_v2", use_cache=True)

    # Generate predictions for ALL events
    print("\nGenerating predictions for all twilights at 3h lead...")
    pred_df = generate_predictions_all_events(model, grid, tw_events, cfg)
    print(f"  Got {len(pred_df)} predictions")
    rmse_all = np.sqrt((pred_df["residual"] ** 2).mean())
    print(f"  Overall RMSE at 3h: {rmse_all:.3f}")

    # Build features
    print("\nBuilding Ridge feature matrix...")
    feat_df = build_ridge_features(pred_df, grid, tw_events)
    feat_df = feat_df.dropna()
    print(f"  Feature matrix: {len(feat_df)} events")

    # Even/odd split
    feat_df = feat_df.reset_index(drop=True)
    train = feat_df.iloc[::2]
    test = feat_df.iloc[1::2]
    print(f"  Train (even): {len(train)}, Test (odd): {len(test)}")

    # Feature columns (everything except residual and tw_time)
    feat_cols = [c for c in feat_df.columns if c not in ["residual", "tw_time"]]
    print(f"  Features: {len(feat_cols)}")

    # Correlations on training set
    print("\n  Top correlations with residual (train set):")
    corrs = train[feat_cols + ["residual"]].corr()["residual"].drop("residual").sort_values(key=abs, ascending=False)
    for col, r in corrs.head(10).items():
        print(f"    {col:30s} r = {r:+.3f}")

    # Train Ridge
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols].fillna(0).values)
    y_train = train["residual"].values

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train, y_train)

    # Evaluate on test
    X_test = scaler.transform(test[feat_cols].fillna(0).values)
    correction = ridge.predict(X_test)

    rmse_before = np.sqrt((test["residual"].values ** 2).mean())
    rmse_after = np.sqrt(((test["residual"].values - correction) ** 2).mean())
    bias_before = test["residual"].mean()
    bias_after = (test["residual"].values - correction).mean()

    print(f"\n{'='*60}")
    print(f"RESULTS AT 3h LEAD")
    print(f"{'='*60}")
    print(f"  N_train: {len(train)}, N_test: {len(test)}")
    print(f"  RMSE before: {rmse_before:.3f}")
    print(f"  RMSE after:  {rmse_after:.3f}")
    print(f"  Improvement: {(rmse_before-rmse_after)/rmse_before*100:.1f}%")
    print(f"  Bias before: {bias_before:+.3f}")
    print(f"  Bias after:  {bias_after:+.3f}")

    # Feature importance
    print(f"\n  Ridge coefficients (top 10):")
    coef_df = pd.DataFrame({"feature": feat_cols, "coef": ridge.coef_})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    for _, row in coef_df.head(10).iterrows():
        print(f"    {row['feature']:30s} coef = {row['coef']:+.4f}")


if __name__ == "__main__":
    main()
