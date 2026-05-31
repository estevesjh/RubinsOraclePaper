"""Causal NBEATSx-Diff twilight temperature forecasting.

Target: T at alt_sun = -15 deg (sun setting).
Method: Train NBEATSx on D(tn) = T(tn) - T(tn - 0.5 solar day).
        Reconstruct: T_pred = D_pred + T_observed(tn - 0.5).
        Fully causal — no future information used.

Uses FeatureBuilder from rubin-twilight-forecast for feature engineering.

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    python forecast/run.py
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

from contextlib import contextmanager

import numpy as np
import pandas as pd

# Add rubin-twilight-forecast to path for FeatureBuilder
sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(__file__))

from twilight.config import NBEATSxConfig
from twilight.features import FeatureBuilder
from twilight.utils import load_data

from config import (
    DATA_PATH, RESULTS_PATH, MODEL_CACHE_PATH,
    TEST_START_DATE, HALFDAY_LAG_STEPS, SOLAR_GRID_FREQ,
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, NBEATS_MAX_STEPS,
    LEAD_TIMES_HOURS, STEPS_PER_DAY, SOLAR_GRID_STEP,
    RIDGE_ALPHA,
)


@contextmanager
def suppress_stdout():
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


# ── Data loading & feature construction via FeatureBuilder ───────────────


def load_and_prepare():
    """Load data and run FeatureBuilder with solar_grid=True, step=0.01."""
    print("Loading data...")
    df = load_data(str(DATA_PATH))
    print(f"  Rows: {len(df)}, range: {df['ds'].min()} to {df['ds'].max()}")

    # Configure FeatureBuilder with solar grid enabled
    cfg = NBEATSxConfig(
        solar_grid=True,
        solar_grid_step=SOLAR_GRID_STEP,
        solar_grid_fillna=True,
        sun_alt_midpoint=-15.0,
        smooth_window_hours=1.0,  # 1h Gaussian smoothing
        input_size=NBEATS_INPUT_SIZE,
        horizon=NBEATS_HORIZON,
    )

    print("Running FeatureBuilder (sun altitude, events, rates, solar grid)...")
    fb = FeatureBuilder(cfg)
    grid = fb.transform(df)
    print(f"  Grid rows: {len(grid)}, DayCount range: [{grid['DayCount'].min()}, {grid['DayCount'].max()}]")

    # Add lag features (uniform grid makes these clean)
    print("Adding lag features on solar grid...")
    grid["y_raw"] = grid["y"].copy()
    grid["y_lag_6"] = grid["y"].shift(6)        # 3h ago
    grid["y_lag_12"] = grid["y"].shift(12)      # 6h ago
    grid["y_lag_24"] = grid["y"].shift(HALFDAY_LAG_STEPS)  # 12h ago (anchor)
    grid["y_lag_48"] = grid["y"].shift(48)      # 24h ago

    # Backward OLS slope over 4 grid steps (~2h at 30-min cadence)
    window = 4
    y_arr = grid["y"].values
    slopes = np.full(len(y_arr), np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()
    for i in range(window, len(y_arr)):
        seg = y_arr[i - window:i]
        if np.isnan(seg).any():
            continue
        cov = ((x - x_mean) * (seg - seg.mean())).sum()
        slopes[i] = cov / var_x
    grid["trend_solar_2h"] = slopes

    # Deterministic phase features
    grid["solar_sin"] = np.sin(2 * np.pi * grid["SolarTime"])
    grid["solar_cos"] = np.cos(2 * np.pi * grid["SolarTime"])
    doy = grid["ds_real"].dt.dayofyear
    grid["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    grid["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Add differenced target: D = y - y.shift(50)
    print("Building differenced target D = y - y.shift(50)...")
    grid["D"] = grid["y"] - grid["y"].shift(HALFDAY_LAG_STEPS)
    grid["y_anchor"] = grid["y"].shift(HALFDAY_LAG_STEPS)

    # Drop warmup rows
    n_warmup = NBEATS_INPUT_SIZE + HALFDAY_LAG_STEPS
    grid = grid.iloc[n_warmup:].reset_index(drop=True)
    print(f"  After warmup: {len(grid)} rows")

    return grid, cfg


def find_twilight_targets(grid):
    """Find grid indices where alt_sun crosses -15 deg (setting)."""
    mask = grid["twilight_event_sunset"].values.astype(bool)
    idxs = np.where(mask)[0]

    events = []
    for i in idxs:
        events.append({
            "grid_idx": i,
            "solarDayHour": grid["solarDayHour"].iloc[i],
            "ds_real": grid["ds_real"].iloc[i],
            "DayCount": grid["DayCount"].iloc[i],
            "y_actual": grid["y"].iloc[i],
        })
    return pd.DataFrame(events)


# ── NBEATSx training ─────────────────────────────────────────────────────


def get_hist_futr_exog(cfg):
    """Causal hist/futr exog for the solar-grid differenced model.

    On a uniform solar grid, raw y lags are the most natural and powerful
    features. FeatureBuilder rate features are interpolation-smeared and
    underperform simple lags. Keep FeatureBuilder for grid construction +
    twilight_cos (futr), add lag features manually after transform.
    """
    hist_exog = [
        "y_raw",             # absolute temperature
        "y_lag_6",           # 3h ago
        "y_lag_12",          # 6h ago
        "y_lag_24",          # 12h ago (the reconstruction anchor)
        "y_lag_48",          # 24h ago
        "trend_solar_2h",   # backward OLS slope
    ]
    futr_exog = [
        "solar_sin",
        "solar_cos",
        "doy_sin",
        "doy_cos",
    ]
    return hist_exog, futr_exog


def train_nbeats_diff(grid, cfg, use_cache=True):
    """Train NBEATSx on the differenced target D."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    cache_path = MODEL_CACHE_PATH / "NBEATSx_diff_v2"

    if use_cache and cache_path.exists():
        print(f"  Loading cached model from {cache_path.name}")
        nf = NeuralForecast.load(str(cache_path))
        nf.models[0].inference_windows_batch_size = 64
        return nf

    hist_exog, futr_exog = get_hist_futr_exog(cfg)

    # Filter to columns actually present in grid
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]

    all_exog = hist_exog + futr_exog

    # Use ALL data for training; val_size for early stopping
    nf_train = grid[["ds", "D"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["D"]
    nf_train["unique_id"] = "temp"

    # val_size = last 10% of data for validation / early stopping
    val_size = int(len(nf_train) * 0.1)

    print(f"  Training NBEATSx-Diff on {len(nf_train)} samples (val_size={val_size})...")
    print(f"  hist_exog: {hist_exog}")
    print(f"  futr_exog: {futr_exog}")

    model = NBEATSx(
        h=NBEATS_HORIZON,
        input_size=NBEATS_INPUT_SIZE,
        max_steps=NBEATS_MAX_STEPS,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=0.01,
        scaler_type="robust",
        enable_progress_bar=True,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[32, 32]],
        n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=5,
        val_check_steps=10,
    )

    nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
    nf.fit(nf_train, val_size=val_size)

    MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    nf.save(str(cache_path))
    print(f"  Cached to {cache_path.name}")

    return nf


# ── Forecast + evaluation ────────────────────────────────────────────────


def lead_hours_to_steps(lead_hours):
    """Convert real-clock lead hours to solar-grid step offset."""
    return round(lead_hours * STEPS_PER_DAY / 24.0)


def predict_and_evaluate(model, grid, tw_events, cfg):
    """Batch predictions for all twilight events × lead times."""
    hist_exog, futr_exog = get_hist_futr_exog(cfg)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    y_arr = grid["y"].values
    ds_real_arr = grid["ds_real"].values

    # Build all requests
    requests = []
    for ev_i, (_, ev) in enumerate(tw_events.iterrows()):
        target_idx = ev["grid_idx"]
        for lead_h in LEAD_TIMES_HOURS:
            offset_steps = lead_hours_to_steps(lead_h)
            issue_idx = target_idx - offset_steps
            if issue_idx < NBEATS_INPUT_SIZE:
                continue
            if (target_idx - issue_idx) >= NBEATS_HORIZON:
                continue
            requests.append({
                "ev_i": ev_i,
                "target_idx": target_idx,
                "issue_idx": issue_idx,
                "lead_h": lead_h,
                "actual_temp": ev["y_actual"],
                "tw_real_time": ev["ds_real"],
            })

    print(f"  Total prediction requests: {len(requests)}")

    # Process in batches
    BATCH_SIZE = 64
    results = []
    n_batches = (len(requests) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(n_batches):
        if batch_i % 20 == 0:
            print(f"  Batch {batch_i}/{n_batches}...")

        batch = requests[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        all_hist, all_futr, batch_meta = [], [], []

        for req in batch:
            issue_idx = req["issue_idx"]
            hist_start = issue_idx - NBEATS_INPUT_SIZE
            futr_end = issue_idx + NBEATS_HORIZON

            if futr_end > len(grid):
                continue

            uid = f"r_{batch_i}_{req['ev_i']}_{req['lead_h']}"

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

            anchor_idx = req["target_idx"] - HALFDAY_LAG_STEPS
            if anchor_idx < 0 or anchor_idx >= len(y_arr):
                continue
            anchor_temp = y_arr[anchor_idx]
            if np.isnan(anchor_temp):
                continue

            T_pred = D_pred + anchor_temp

            results.append({
                "twilight_time": req["tw_real_time"],
                "forecast_time": pd.Timestamp(ds_real_arr[req["issue_idx"]]),
                "lead_time_hours": req["lead_h"],
                "actual_temp": req["actual_temp"],
                "model": "NBEATSx-Diff",
                "forecast_temp": T_pred,
                "error": req["actual_temp"] - T_pred,
            })

    return pd.DataFrame(results)


# ── Optional Ridge correction on residuals ───────────────────────────────


def ridge_correction(model, grid, tw_events, nbeats_results_test, cfg):
    """Train Ridge on NBEATSx residuals at twilight points."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    hist_exog, futr_exog = get_hist_futr_exog(cfg)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    y_arr = grid["y"].values
    lead_h = 3.0
    offset_steps = lead_hours_to_steps(lead_h)

    tw_train = tw_events[tw_events["ds_real"] < TEST_START_DATE].copy()
    tw_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE].copy()
    print(f"  Generating in-sample predictions for {len(tw_train)} training twilights...")

    def get_residuals(tw_subset):
        """Get NBEATSx residuals at 3h lead."""
        residuals = []
        requests = []
        for ev_i, (_, ev) in enumerate(tw_subset.iterrows()):
            target_idx = ev["grid_idx"]
            issue_idx = target_idx - offset_steps
            if issue_idx < NBEATS_INPUT_SIZE:
                continue
            if (target_idx - issue_idx) >= NBEATS_HORIZON:
                continue
            requests.append({
                "ev_i": ev_i, "target_idx": target_idx,
                "issue_idx": issue_idx, "actual_temp": ev["y_actual"],
                "tw_real_time": ev["ds_real"],
            })

        BATCH_SIZE = 64
        n_batches = (len(requests) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_i in range(n_batches):
            batch = requests[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
            all_hist, all_futr, batch_meta = [], [], []
            for req in batch:
                issue_idx = req["issue_idx"]
                hist_start = issue_idx - NBEATS_INPUT_SIZE
                futr_end = issue_idx + NBEATS_HORIZON
                if futr_end > len(grid):
                    continue
                uid = f"ridge_{batch_i}_{req['ev_i']}"
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
                anchor_idx = req["target_idx"] - HALFDAY_LAG_STEPS
                if anchor_idx < 0 or anchor_idx >= len(y_arr):
                    continue
                anchor_temp = y_arr[anchor_idx]
                if np.isnan(anchor_temp):
                    continue
                T_pred = D_pred + anchor_temp
                residuals.append({
                    "tw_time": req["tw_real_time"],
                    "residual": req["actual_temp"] - T_pred,
                    "target_idx": req["target_idx"],
                    "issue_idx": req["issue_idx"],
                })
        return pd.DataFrame(residuals)

    train_res = get_residuals(tw_train)
    test_res = nbeats_results_test[nbeats_results_test["lead_time_hours"] == lead_h].copy()
    test_res = test_res.rename(columns={"twilight_time": "tw_time"})
    test_res["residual"] = test_res["actual_temp"] - test_res["forecast_temp"]

    print(f"  Train residuals: {len(train_res)}, Test residuals: {len(test_res)}")
    if len(train_res) < 10 or len(test_res) < 10:
        print("  Insufficient data — skipping Ridge.")
        return pd.DataFrame()

    # Build causal features at issue time
    def build_features(res_df, tw_sub, all_residuals):
        feats = []
        for _, row in res_df.iterrows():
            tw_time = row["tw_time"]
            mask = tw_sub["ds_real"] == tw_time
            if not mask.any():
                continue
            ev = tw_sub[mask].iloc[0]
            target_idx = ev["grid_idx"]
            issue_idx = target_idx - offset_steps
            if issue_idx < 100 or issue_idx >= len(grid):
                continue

            feat = {
                "residual": row["residual"],
                "tw_time": tw_time,
                "y_at_issue": y_arr[issue_idx],
                "y_anchor": y_arr[target_idx - HALFDAY_LAG_STEPS],
                "doy_sin": grid["doy_sin"].iloc[issue_idx] if "doy_sin" in grid.columns else 0,
                "doy_cos": grid["doy_cos"].iloc[issue_idx] if "doy_cos" in grid.columns else 0,
            }
            # Add available hist features at issue
            for col in ["trend_2h", "temp_trend_3d", "rate_sunrise_to_midday",
                        "rate_twilight_to_midnight", "rate_midnight_to_sunrise"]:
                if col in grid.columns:
                    feat[col] = grid[col].iloc[issue_idx]

            # Lagged residuals from previous twilights
            prev = all_residuals[all_residuals["tw_time"] < tw_time].tail(3)
            for lag_i in range(3):
                if lag_i < len(prev):
                    feat[f"res_prev_{lag_i+1}"] = prev.iloc[-(lag_i+1)]["residual"]
                else:
                    feat[f"res_prev_{lag_i+1}"] = 0.0
            feats.append(feat)
        return pd.DataFrame(feats)

    all_res = pd.concat([
        train_res[["tw_time", "residual"]],
        test_res[["tw_time", "residual"]],
    ]).sort_values("tw_time")

    train_feats = build_features(train_res, tw_train, all_res)
    test_feats = build_features(test_res, tw_test, all_res)

    feat_cols = [c for c in train_feats.columns
                 if c not in ["residual", "tw_time"]]

    if len(train_feats) < 10 or len(test_feats) < 10:
        print(f"  Insufficient features — skipping.")
        return pd.DataFrame()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats[feat_cols].fillna(0).values)
    y_train = train_feats["residual"].values

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train, y_train)

    X_test = scaler.transform(test_feats[feat_cols].fillna(0).values)
    res_pred = ridge.predict(X_test)

    rmse_before = np.sqrt((test_feats["residual"].values ** 2).mean())
    rmse_after = np.sqrt(((test_feats["residual"].values - res_pred) ** 2).mean())
    print(f"  Ridge at 3h: RMSE before={rmse_before:.3f}, after={rmse_after:.3f}")

    # Apply correction to all lead times
    correction_map = dict(zip(test_feats["tw_time"], res_pred))
    ridge_results = nbeats_results_test[
        nbeats_results_test["twilight_time"].isin(test_feats["tw_time"].values)
    ].copy()
    ridge_results["correction"] = ridge_results["twilight_time"].map(correction_map)
    ridge_results["forecast_temp"] = ridge_results["forecast_temp"] + ridge_results["correction"]
    ridge_results["error"] = ridge_results["actual_temp"] - ridge_results["forecast_temp"]
    ridge_results["model"] = "NBEATSx-Diff-Ridge"
    ridge_results = ridge_results.drop(columns=["correction"])

    return ridge_results


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("CAUSAL NBEATSx-Diff TWILIGHT FORECASTING")
    print("Target: T at alt_sun = -15 deg (sun setting)")
    print("Method: D(tn) = T(tn) - T(tn-0.5), solar grid step=0.01")
    print("Features: FeatureBuilder from rubin-twilight-forecast")
    print("=" * 70)

    # Step 1: Prepare solar grid with FeatureBuilder
    grid, cfg = load_and_prepare()

    # Step 2: Find twilight target events
    print("\nFinding twilight events (alt_sun crossing -15 deg setting)...")
    tw_events = find_twilight_targets(grid)
    tw_events_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE].copy()
    print(f"  Total: {len(tw_events)}, Test (2025+): {len(tw_events_test)}")

    # Step 3: Train NBEATSx
    print("\n" + "=" * 70)
    print("TRAINING NBEATSx-Diff")
    print("=" * 70)
    model = train_nbeats_diff(grid, cfg)

    # Step 4: Predict and evaluate
    print("\n" + "=" * 70)
    print("FORECASTING & EVALUATION")
    print("=" * 70)
    results = predict_and_evaluate(model, grid, tw_events_test, cfg)
    print(f"\n  Total result rows: {len(results)}")

    # Step 5: RMSE by lead time
    print("\n" + "=" * 70)
    print("RMSE BY LEAD TIME")
    print("=" * 70)
    print(f"{'Lead (h)':>10} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'N':>6} {'<1C':>6}")
    print("-" * 55)
    for lead_h in LEAD_TIMES_HOURS:
        subset = results[results["lead_time_hours"] == lead_h]
        if len(subset) > 0:
            rmse = np.sqrt((subset["error"] ** 2).mean())
            mae = subset["error"].abs().mean()
            bias = subset["error"].mean()
            lt1 = (subset["error"].abs() < 1.0).mean() * 100
            print(f"{lead_h:10.1f} {rmse:8.3f} {mae:8.3f} {bias:+8.3f} {len(subset):6d} {lt1:5.1f}%")

    # Step 6: Ridge correction
    print("\n" + "=" * 70)
    print("RIDGE CORRECTION")
    print("=" * 70)
    ridge_results = ridge_correction(model, grid, tw_events, results, cfg)

    if len(ridge_results) > 0:
        print("\nRidge-corrected RMSE:")
        print(f"{'Lead (h)':>10} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'N':>6} {'<1C':>6}")
        print("-" * 55)
        for lead_h in LEAD_TIMES_HOURS:
            subset = ridge_results[ridge_results["lead_time_hours"] == lead_h]
            if len(subset) > 0:
                rmse = np.sqrt((subset["error"] ** 2).mean())
                mae = subset["error"].abs().mean()
                bias = subset["error"].mean()
                lt1 = (subset["error"].abs() < 1.0).mean() * 100
                print(f"{lead_h:10.1f} {rmse:8.3f} {mae:8.3f} {bias:+8.3f} {len(subset):6d} {lt1:5.1f}%")

    # Save
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_PATH / "paper_results_diff.csv"
    all_results = pd.concat([results, ridge_results], ignore_index=True)
    all_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Summary
    rmse_3h = np.sqrt((results[results["lead_time_hours"] == 3.0]["error"] ** 2).mean())
    print(f"\n*** NBEATSx-Diff RMSE at 3h: {rmse_3h:.3f} deg ***")


if __name__ == "__main__":
    main()
