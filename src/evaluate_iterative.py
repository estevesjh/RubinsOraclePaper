"""Iterative ΔT Correction for Improved Ridge at 9am.

Two-stage approach:
1. At 10h (9am): Train Ridge to estimate slope_v1
2. Correct delta_T_approx for all times: delta_T_corrected = y - (T_tw_last + (slope_v1/24) * h)
3. Re-run NBEATSx with corrected delta_T_approx
4. Train Ridge on new residuals to predict remaining slope error
5. Final: tw_temp_pred = T_tw_last + slope_v1 + slope_v2

Usage:
    conda activate /opt/homebrew/Caskroom/mambaforge/base/envs/astro
    python src/evaluate_iterative.py
"""

import os
import sys
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PATH, NBEATS_CACHE_PATH, RESULTS_PATH, RIDGE_ALPHA
from config import (
    NBEATS_FUTR_EXOG as FUTR_EXOG,
)
from config import (
    NBEATS_HIST_EXOG as HIST_EXOG,
)
from config import (
    NBEATS_INPUT_SIZE as INPUT_SIZE,
)
from data.preprocessing import add_key_time_features, add_trend_2h
from models.nbeats import load_and_prepare_data, suppress_stdout

# Model path - use the cached model from training
MODEL_PATH = NBEATS_CACHE_PATH / "NBEATSx_deltaT_v1_20250101"
# MODEL_PATH = TWILIGHT_OFFSET_CACHE / "NBEATSx_deltaT_20241231"

# Ridge feature columns
RIDGE_FEATURES = [
    "res",
    "res_midday",
    "res_sunrise",
    "res_midnight",
    "res_prev_tw",
    "rate_twilight_to_midnight",
    "rate_sunrise_to_midday",
    "rate_midnight_to_sunrise",
    "twilight_cos_feat",
    "doy_sin",
    "doy_cos",
    "res_sq_temp",
]


def load_model():
    """Load pre-trained NBEATSx model."""
    from neuralforecast import NeuralForecast

    print(f"Loading NBEATSx model from {MODEL_PATH.name}...")
    nf = NeuralForecast.load(str(MODEL_PATH))
    nf.models[0].inference_windows_batch_size = 64
    return nf


def predict_batch_corrected(model, df, requests, slope_dict):
    """Make batched predictions using CORRECTED delta_T_approx.

    Unlike the original predict_batch, this uses:
    - delta_T for past cycles (ds <= last_tw_time)
    - CORRECTED delta_T_approx for current cycle (ds > last_tw_time)

    The correction: delta_T_corrected = y - (T_tw_last + slope_v1/24 * h_from_tw)
    """
    all_exog = HIST_EXOG + FUTR_EXOG

    all_hist_dfs = []
    all_futr_dfs = []
    prediction_info = []

    for idx, req in enumerate(requests):
        forecast_time = req["forecast_time"]
        tw_time = req["tw_time"]
        last_tw_time = req["last_tw_time"]
        unique_id = f"pred_{idx}"

        # Get slope_v1 for this twilight
        slope_v1 = slope_dict.get(tw_time, 0)

        # Historical data before forecast_time
        mask = df["ds"] < forecast_time
        recent_df = df.loc[mask]

        # Get columns needed - T_tw_last needed for correction calculation
        base_cols = ["ds", "delta_T", "delta_T_approx", "y", "h_from_tw", "T_tw_last"]
        all_cols = list(dict.fromkeys(base_cols + all_exog))  # Remove duplicates
        nf_recent = recent_df[all_cols].dropna().tail(5 * INPUT_SIZE).copy()

        if len(nf_recent) < INPUT_SIZE:
            continue

        # Build y column with CORRECTED delta_T_approx for current cycle
        # Past cycles (ds <= last_tw_time): use delta_T
        # Current cycle (ds > last_tw_time): use corrected delta_T_approx
        corrected_approx = nf_recent["y"] - (
            nf_recent["T_tw_last"] + (slope_v1 / 24) * nf_recent["h_from_tw"]
        )

        nf_recent["y_input"] = np.where(
            nf_recent["ds"] > last_tw_time,
            corrected_approx,  # Corrected for current cycle
            nf_recent["delta_T"],  # True delta_T for past cycles
        )

        # Get sunrise times for twilight_cos calculation
        sunrise_mask = recent_df["sunrise_temp"].notna()
        if sunrise_mask.any():
            last_sunrise_time = recent_df.loc[sunrise_mask, "ds"].iloc[-1]
        else:
            last_sunrise_time = last_tw_time - pd.Timedelta(hours=12)
        nf_recent["y"] = nf_recent["y_input"]
        nf_recent["unique_id"] = unique_id

        # Future dataframe - use model.make_future_dataframe to ensure correct format
        # This guarantees timestamps match what the model expects
        futr_df = model.make_future_dataframe(
            nf_recent[
                ["ds", "y", "unique_id"]
                + [c for c in all_exog if c in nf_recent.columns]
            ]
        )

        # Future sunrise (simplified)
        next_sunrise_time = last_sunrise_time + pd.Timedelta(hours=24)

        # Compute twilight_cos for future
        is_day = last_sunrise_time > last_tw_time
        ts_series = futr_df["ds"]

        if is_day:
            daylight_secs = (tw_time - last_sunrise_time).total_seconds()
            if daylight_secs > 0:
                elapsed = (ts_series - last_sunrise_time).dt.total_seconds()
                progress = np.clip(elapsed / daylight_secs, 0, 1)
            else:
                progress = np.zeros(len(ts_series))
        else:
            night_secs = (next_sunrise_time - last_tw_time).total_seconds()
            if night_secs > 0:
                elapsed = (ts_series - last_tw_time).dt.total_seconds()
                progress = 1 + np.clip(elapsed / night_secs, 0, 1)
            else:
                progress = np.ones(len(ts_series))

        futr_df["twilight_cos"] = np.cos(np.pi * progress)

        # Keep only needed columns
        keep_cols = ["ds", "y", "unique_id"] + [
            c for c in all_exog if c in nf_recent.columns
        ]
        nf_recent = nf_recent[keep_cols]

        all_hist_dfs.append(nf_recent)
        all_futr_dfs.append(futr_df)
        prediction_info.append(
            {
                "unique_id": unique_id,
                "req_idx": idx,
                "forecast_time": forecast_time,
            }
        )

    if not all_hist_dfs:
        return {}

    # Predict one at a time to avoid batching issues
    results = {}
    for i, (hist_df, futr_df, info) in enumerate(
        zip(all_hist_dfs, all_futr_dfs, prediction_info)
    ):
        # try:
        # Debug first prediction
        if i == 0:
            print(
                f"      DEBUG hist_df: {len(hist_df)} rows, ds range: {hist_df['ds'].min()} to {hist_df['ds'].max()}"
            )
            print(
                f"      DEBUG futr_df: {len(futr_df)} rows, ds range: {futr_df['ds'].min()} to {futr_df['ds'].max()}"
            )
            print(f"      DEBUG unique_id: {hist_df['unique_id'].iloc[0]}")

        # expected = model.make_future_dataframe(hist_df)
        # print(
        #     f"      Expected futr_df: {len(expected)} rows, ds range: {expected['ds'].min()} to {expected['ds'].max()}"
        # )

        with suppress_stdout():
            fc = model.predict(hist_df, futr_df=futr_df)

        fc = fc.reset_index()
        model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        uid_fc = fc.sort_values("ds").reset_index(drop=True)

        results[info["req_idx"]] = {
            "forecast_time": info["forecast_time"],
            "predictions": uid_fc[["ds", model_col]].rename(
                columns={model_col: "delta_T_pred"}
            ),
        }
        # except Exception as e:
        #     if i == 0:
        #         print(f"      First prediction error: {str(e)[:80]}")
        #         # Try to get expected timestamps
        #         try:
        #             expected = model.make_future_dataframe(hist_df)
        #             print(
        #                 f"      Expected futr_df: {len(expected)} rows, ds range: {expected['ds'].min()} to {expected['ds'].max()}"
        #             )
        #         except:
        #             pass

        if (i + 1) % 50 == 0:
            print(f"      Processed {i + 1}/{len(all_hist_dfs)}...")

    return results


def prepare_ridge_features(df_offset):
    """Add Ridge feature columns to offset predictions dataframe."""
    df = df_offset.copy()

    # Lagged residuals
    for name, lag_h in {
        "res_midday": 6,
        "res_sunrise": 10,
        "res_midnight": 17,
        "res_prev_tw": 21,
    }.items():
        df[name] = df["res"].shift(int(lag_h * 4))

    # Seasonal features
    df["doy"] = df["tw_time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    df["tw_hour"] = df["tw_time"].dt.hour + df["tw_time"].dt.minute / 60
    df["twilight_cos_feat"] = np.cos(2 * np.pi * df["tw_hour"] / 24)

    # Bias correction
    df["res_sq_temp"] = (df["res"] ** 2) / (np.abs(df["temp_actual"]) + 5)

    return df


def train_ridge_at_leadtime(df_offset, lead_time_hours, feature_cols):
    """Train Ridge model at specified lead time."""
    df = df_offset.copy()

    # Filter for lead time
    mask = (df["hours_to_tw"] >= lead_time_hours - 0.25) & (
        df["hours_to_tw"] < lead_time_hours + 0.25
    )
    df_lt = df[mask].groupby("tw_time").last().reset_index()
    df_lt = df_lt.dropna(subset=feature_cols + ["tw_slope"])

    # Train/test split
    train = df_lt[df_lt["day_num"] % 2 == 1]
    test = df_lt[df_lt["day_num"] % 2 == 0]

    if len(train) < 10 or len(test) < 10:
        return None, None, None, None

    # Train Ridge
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)
    X_test = scaler.transform(test[feature_cols].values)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, train["tw_slope"].values)

    # Predict
    slope_pred_train = model.predict(X_train)
    slope_pred_test = model.predict(X_test)

    return (
        model,
        scaler,
        dict(zip(train["tw_time"], slope_pred_train)),
        dict(zip(test["tw_time"], slope_pred_test)),
    )


def correct_delta_T_approx(df, slope_dict):
    """Create corrected delta_T_approx based on slope_v1 estimates.

    Only corrects current cycle values (h_from_tw > 0).
    Past cycle values keep original delta_T (used by predict_batch).

    delta_T_corrected = y - (T_tw_last + (slope_v1/24) * h_from_tw)
    """
    df_corrected = df.copy()

    # Map each row to its upcoming twilight using next_tw_time
    df_corrected["slope_v1"] = df_corrected["next_tw_time"].map(slope_dict)

    # Fill missing slope_v1 with 0 (no correction for those cycles)
    df_corrected["slope_v1"] = df_corrected["slope_v1"].fillna(0)

    # Compute corrected baseline for current cycle only
    df_corrected["baseline_corrected"] = (
        df_corrected["T_tw_last"]
        + (df_corrected["slope_v1"] / 24) * df_corrected["h_from_tw"]
    )

    # Only correct delta_T_approx for current cycle (h_from_tw > 0)
    # Past values remain as original delta_T_approx (which equals delta_T for past cycles)
    mask_current_cycle = df_corrected["h_from_tw"] > 0
    df_corrected.loc[mask_current_cycle, "delta_T_approx"] = (
        df_corrected.loc[mask_current_cycle, "y"]
        - df_corrected.loc[mask_current_cycle, "baseline_corrected"]
    )

    return df_corrected


def main():
    """Run iterative ΔT correction evaluation.

    Approach:
    - Stage 1 at SUNRISE: Get initial slope_v1 from Ridge
    - Stage 2: Re-run NBEATSx with corrected baseline
    - Stage 3 at 10am LOCAL: Get slope_v2 from new residuals
    """
    print("=" * 70)
    print("ITERATIVE ΔT CORRECTION: SUNRISE → 10am")
    print("=" * 70)

    # Load and prepare main data
    print("\n[1] Loading data...")
    df = load_and_prepare_data(DATA_PATH)
    df = add_trend_2h(df)
    df = add_key_time_features(df)
    print(f"    Loaded {len(df)} rows")

    # Get sunrise times from main df
    sunrise_times = df[df["sunrise_temp"].notna()][["ds"]].copy()
    sunrise_times = sunrise_times.rename(columns={"ds": "sunrise_time"})
    print(f"    Found {len(sunrise_times)} sunrise events")

    # Load offset predictions
    print("\n[2] Loading offset predictions...")
    df_offset = pd.read_csv(RESULTS_PATH / "twilight_offset_predictions.csv")
    df_offset["tw_time"] = pd.to_datetime(df_offset["tw_time"])
    df_offset["tw_date"] = pd.to_datetime(df_offset["tw_date"])
    df_offset["target_time"] = pd.to_datetime(df_offset["target_time"])
    df_offset = df_offset.rename(columns={"res_tw": "res"})
    df_offset["hours_to_tw"] = (
        df_offset["tw_time"] - df_offset["target_time"]
    ).dt.total_seconds() / 3600
    df_offset["tw_slope"] = df_offset["tw_temp"] - df_offset["T_tw_last"]
    df_offset["day_num"] = (df_offset["tw_date"] - df_offset["tw_date"].min()).dt.days

    # Map each twilight to its sunrise time (same day)
    twilight_sunrise = {}
    for tw_time in df_offset["tw_time"].unique():
        # Find sunrise within 18h before twilight
        mask = (sunrise_times["sunrise_time"] < tw_time) & \
               (sunrise_times["sunrise_time"] >= tw_time - pd.Timedelta(hours=18))
        nearby = sunrise_times[mask]
        if len(nearby) > 0:
            twilight_sunrise[tw_time] = nearby["sunrise_time"].max()

    df_offset["sunrise_time"] = df_offset["tw_time"].map(twilight_sunrise)
    df_offset["hours_sunrise_to_tw"] = (
        df_offset["tw_time"] - df_offset["sunrise_time"]
    ).dt.total_seconds() / 3600

    # Add Ridge features
    df_offset = prepare_ridge_features(df_offset)
    print(f"    Loaded {len(df_offset)} offset predictions")

    # ==========================================================================
    # STAGE 1: Train Ridge at SUNRISE to get slope_v1
    # ==========================================================================
    print("\n[3] STAGE 1: Train Ridge at SUNRISE...")

    # Get data at sunrise time for each twilight
    # Filter: target_time is within 30min of sunrise_time
    df_offset["time_from_sunrise"] = (
        df_offset["target_time"] - df_offset["sunrise_time"]
    ).dt.total_seconds() / 3600

    df_sunrise = df_offset[
        (df_offset["time_from_sunrise"] >= -0.5) & (df_offset["time_from_sunrise"] < 0.5)
    ].copy()
    df_sunrise = df_sunrise.groupby("tw_time").last().reset_index()
    df_sunrise = df_sunrise.dropna(subset=RIDGE_FEATURES + ["tw_slope"])

    print(f"    Samples at sunrise: {len(df_sunrise)}")
    print(f"    Mean hours_to_tw at sunrise: {df_sunrise['hours_to_tw'].mean():.1f}h")

    # Train/test split
    train_s1 = df_sunrise[df_sunrise["day_num"] % 2 == 1]
    test_s1 = df_sunrise[df_sunrise["day_num"] % 2 == 0]

    if len(train_s1) < 10 or len(test_s1) < 10:
        print("    Not enough data for Stage 1!")
        return

    # Train Ridge
    scaler_s1 = StandardScaler()
    X_train_s1 = scaler_s1.fit_transform(train_s1[RIDGE_FEATURES].values)
    X_test_s1 = scaler_s1.transform(test_s1[RIDGE_FEATURES].values)

    model_s1 = Ridge(alpha=RIDGE_ALPHA)
    model_s1.fit(X_train_s1, train_s1["tw_slope"].values)

    slope_v1_train = dict(zip(train_s1["tw_time"], model_s1.predict(X_train_s1)))
    slope_v1_test = dict(zip(test_s1["tw_time"], model_s1.predict(X_test_s1)))
    slope_v1_all = {**slope_v1_train, **slope_v1_test}
    print(f"    Got slope_v1 for {len(slope_v1_all)} twilights")

    # Evaluate sunrise baseline (single-stage)
    test_s1["slope_v1"] = test_s1["tw_time"].map(slope_v1_test)
    error_sunrise = (test_s1["T_tw_last"] + test_s1["slope_v1"]) - test_s1["tw_temp"]
    rmse_sunrise = np.sqrt((error_sunrise**2).mean())
    print(
        f"    Single-stage RMSE @ sunrise: {rmse_sunrise:.3f}°C, <1°C: {(error_sunrise.abs() < 1).mean() * 100:.1f}%"
    )

    # ==========================================================================
    # STAGE 2: Re-run NBEATSx with corrected baseline for ALL times
    # ==========================================================================
    print("\n[4] STAGE 2: Re-run NBEATSx with corrected baseline...")

    # Cache file for predictions
    cache_file = RESULTS_PATH / "iterative_nbeats_predictions.csv"

    if cache_file.exists():
        print(f"    Loading cached predictions from {cache_file.name}...")
        stage2_preds = pd.read_csv(cache_file)
        stage2_preds["ds"] = pd.to_datetime(stage2_preds["ds"])
        stage2_preds["tw_time"] = pd.to_datetime(stage2_preds["tw_time"])
        print(f"    Loaded {len(stage2_preds)} cached predictions")
    else:
        # Load NBEATSx model
        print("    Loading NBEATSx model...")
        nf = load_model()

        # Build prediction requests - TWO per twilight to cover full 24h cycle
        # NBEATSx predicts 12h ahead, so we need:
        # 1. From last_tw to last_tw+12h (overnight)
        # 2. From last_tw+12h to next_tw (morning to evening)
        print("    Building prediction requests (2 per twilight for 24h coverage)...")
        requests = []
        metadata = []

        for tw_time in slope_v1_all.keys():
            tw_time = pd.Timestamp(tw_time)

            # Get last_tw_time from df
            tw_row = df[(df["ds"] == tw_time) & df["twilight_temp"].notna()]
            if len(tw_row) == 0:
                continue
            last_tw_time = tw_row["last_tw_time"].iloc[0]
            T_tw_last = tw_row["T_tw_last"].iloc[0]
            tw_temp = tw_row["twilight_temp"].iloc[0]

            # Window 1: From last_tw (covers overnight ~19:00 to ~07:00)
            forecast_time_1 = last_tw_time + pd.Timedelta(minutes=15)
            requests.append({
                "tw_time": tw_time,
                "forecast_time": forecast_time_1,
                "last_tw_time": last_tw_time,
                "window": 1,
            })
            metadata.append({
                "tw_time": tw_time,
                "T_tw_last": T_tw_last,
                "tw_temp": tw_temp,
                "slope_v1": slope_v1_all[tw_time],
                "window": 1,
            })

            # Window 2: From last_tw+12h (covers morning ~07:00 to ~19:00)
            forecast_time_2 = last_tw_time + pd.Timedelta(hours=12)
            requests.append({
                "tw_time": tw_time,
                "forecast_time": forecast_time_2,
                "last_tw_time": last_tw_time,
                "window": 2,
            })
            metadata.append({
                "tw_time": tw_time,
                "T_tw_last": T_tw_last,
                "tw_temp": tw_temp,
                "slope_v1": slope_v1_all[tw_time],
                "window": 2,
            })

        print(f"    Total requests: {len(requests)}")

        # Run predictions with corrected baseline
        print("    Running NBEATSx predictions with corrected baseline...")
        preds = predict_batch_corrected(nf, df, requests, slope_v1_all)
        print(f"    Got {len(preds)} predictions")

        # Collect all predictions into a dataframe (combining both windows)
        all_preds = []
        for req_idx, meta in enumerate(metadata):
            if req_idx not in preds:
                continue

            pred_df = preds[req_idx]["predictions"].copy()
            pred_df["tw_time"] = meta["tw_time"]
            pred_df["T_tw_last"] = meta["T_tw_last"]
            pred_df["tw_temp"] = meta["tw_temp"]
            pred_df["slope_v1"] = meta["slope_v1"]
            pred_df["window"] = meta["window"]
            all_preds.append(pred_df)

        stage2_preds = pd.concat(all_preds, ignore_index=True)

        # Remove duplicate timestamps (keep window 2 which is more recent)
        stage2_preds = stage2_preds.sort_values(["tw_time", "ds", "window"])
        stage2_preds = stage2_preds.drop_duplicates(subset=["tw_time", "ds"], keep="last")

        # Save cache
        stage2_preds.to_csv(cache_file, index=False)
        print(f"    Saved predictions to {cache_file.name}")

    print(f"    Total predictions: {len(stage2_preds)}")

    # ==========================================================================
    # STAGE 3 DATA: Compute NEW residuals from corrected NBEATSx at 10am
    # ==========================================================================
    print("\n[5] Computing residuals from corrected NBEATSx...")

    # Merge predictions with actual temperatures
    stage2_preds["h_from_tw"] = (stage2_preds["ds"] - stage2_preds["tw_time"].map(
        lambda x: df[df["ds"] == x]["last_tw_time"].iloc[0] if len(df[df["ds"] == x]) > 0 else pd.NaT
    )).dt.total_seconds() / 3600

    # Get actual temps from df
    df_actual = df[["ds", "y"]].rename(columns={"y": "temp_actual"})
    stage2_preds = stage2_preds.merge(df_actual, on="ds", how="left")

    # Compute corrected baseline and new residual
    # baseline_corrected = T_tw_last + (slope_v1/24) * h_from_tw
    stage2_preds["hours_to_tw"] = (stage2_preds["tw_time"] - stage2_preds["ds"]).dt.total_seconds() / 3600
    stage2_preds["h_from_tw"] = 24 - stage2_preds["hours_to_tw"]
    stage2_preds["baseline_corrected"] = stage2_preds["T_tw_last"] + (stage2_preds["slope_v1"] / 24) * stage2_preds["h_from_tw"]
    stage2_preds["pred_temp"] = stage2_preds["baseline_corrected"] + stage2_preds["delta_T_pred"]
    stage2_preds["res_new"] = stage2_preds["temp_actual"] - stage2_preds["pred_temp"]

    # Filter to 10am window (hours_to_tw ~8-9h)
    df_10am_new = stage2_preds[
        (stage2_preds["hours_to_tw"] >= 7.5) & (stage2_preds["hours_to_tw"] < 9.5)
    ].copy()
    df_10am_new = df_10am_new.groupby("tw_time").last().reset_index()

    # Get day_num and other features from df_offset
    df_offset_meta = df_offset[["tw_time", "tw_date", "day_num", "tw_slope"]].drop_duplicates("tw_time")
    df_10am_new = df_10am_new.merge(df_offset_meta, on="tw_time", how="left")

    # Remaining slope after slope_v1 correction
    df_10am_new["remaining_slope"] = df_10am_new["tw_slope"] - df_10am_new["slope_v1"]

    # Add lagged residuals from stage2_preds
    # res at different hours from last twilight
    KEY_HOURS = {
        "res_h3": 3,    # 3h from last tw
        "res_h6": 6,    # midday area
        "res_h10": 10,  # sunrise area
        "res_h12": 12,  # noon
        "res_h14": 14,  # ~10am for 8h lead
    }

    for name, h in KEY_HOURS.items():
        # Get res_new at h hours from last twilight
        df_at_h = stage2_preds[
            (stage2_preds["h_from_tw"] >= h - 0.5) & (stage2_preds["h_from_tw"] < h + 0.5)
        ].groupby("tw_time")["res_new"].last().reset_index()
        df_at_h = df_at_h.rename(columns={"res_new": name})
        df_10am_new = df_10am_new.merge(df_at_h, on="tw_time", how="left")

    df_10am_new = df_10am_new.dropna(subset=["remaining_slope", "res_new", "day_num"])

    print(f"    Got {len(df_10am_new)} samples at 10am with new residuals")
    print(f"    Mean hours_to_tw: {df_10am_new['hours_to_tw'].mean():.1f}h")

    # Compute single-stage baseline at 10am for comparison (using original offset predictions)
    df_10am_orig = df_offset[
        (df_offset["hours_to_tw"] >= 7.5) & (df_offset["hours_to_tw"] < 9.5)
    ].copy()
    df_10am_orig = df_10am_orig.groupby("tw_time").last().reset_index()
    df_10am_orig["slope_v1"] = df_10am_orig["tw_time"].map(slope_v1_all)
    df_10am_orig = df_10am_orig.dropna(subset=["slope_v1"] + RIDGE_FEATURES + ["tw_slope"])

    train_single = df_10am_orig[df_10am_orig["day_num"] % 2 == 1]
    test_single = df_10am_orig[df_10am_orig["day_num"] % 2 == 0]

    scaler_single = StandardScaler()
    X_train_single = scaler_single.fit_transform(train_single[RIDGE_FEATURES].values)
    X_test_single = scaler_single.transform(test_single[RIDGE_FEATURES].values)

    model_single = Ridge(alpha=RIDGE_ALPHA)
    model_single.fit(X_train_single, train_single["tw_slope"].values)

    slope_single = model_single.predict(X_test_single)
    error_single = (test_single["T_tw_last"].values + slope_single) - test_single["tw_temp"].values
    rmse_10am_single = np.sqrt((error_single**2).mean())
    print(f"    Single-stage RMSE @ 10am: {rmse_10am_single:.3f}°C, <1°C: {(np.abs(error_single) < 1).mean() * 100:.1f}%")

    # ==========================================================================
    # STAGE 3: Train Ridge on NEW residuals from corrected NBEATSx
    # ==========================================================================
    print("\n[6] STAGE 3: Train Ridge on new residuals at 10am...")

    # Use new residual features from corrected NBEATSx
    stage3_features = ["res_new"] + list(KEY_HOURS.keys())

    # Drop rows with missing features
    df_10am_new = df_10am_new.dropna(subset=stage3_features)
    print(f"    Features: {stage3_features}")
    print(f"    Samples with all features: {len(df_10am_new)}")

    # Train/test split (same as Stage 1)
    train_s3 = df_10am_new[df_10am_new["day_num"] % 2 == 1]
    test_s3 = df_10am_new[df_10am_new["day_num"] % 2 == 0]

    if len(train_s3) >= 10 and len(test_s3) >= 10:
        # Train Ridge to predict remaining_slope using same features
        X_train_s3 = train_s3[stage3_features].values
        y_train_s3 = train_s3["remaining_slope"].values
        X_test_s3 = test_s3[stage3_features].values

        scaler_s3 = StandardScaler()
        X_train_s3_scaled = scaler_s3.fit_transform(X_train_s3)
        X_test_s3_scaled = scaler_s3.transform(X_test_s3)

        model_s3 = Ridge(alpha=RIDGE_ALPHA)
        model_s3.fit(X_train_s3_scaled, y_train_s3)

        slope_v2_pred = model_s3.predict(X_test_s3_scaled)
        test_s3 = test_s3.copy()
        test_s3["slope_v2"] = slope_v2_pred

        # Final prediction: T_tw_last + slope_v1 + slope_v2
        test_s3["pred_tw"] = test_s3["T_tw_last"] + test_s3["slope_v1"] + test_s3["slope_v2"]
        test_s3["error"] = test_s3["pred_tw"] - test_s3["tw_temp"]

        print(f"    Train: {len(train_s3)}, Test: {len(test_s3)}")
        print(f"    Correlation (res_new vs remaining_slope): {np.corrcoef(train_s3['res_new'], train_s3['remaining_slope'])[0, 1]:.3f}")
    else:
        print("    Not enough data for Stage 3 Ridge")
        test_s3 = pd.DataFrame()

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Results with Stage 3 (full iterative)
    if len(test_s3) > 0:
        rmse_iter = np.sqrt((test_s3["error"] ** 2).mean())
        mae_iter = test_s3["error"].abs().mean()
        bias_iter = test_s3["error"].mean()
        pct_05 = (test_s3["error"].abs() < 0.5).mean() * 100
        pct_10 = (test_s3["error"].abs() < 1.0).mean() * 100

        print(f"\nWith Stage 3 Ridge (slope_v1 + slope_v2):")
        print(f"  N samples: {len(test_s3)}")
        print(f"  RMSE:  {rmse_iter:.3f}°C")
        print(f"  MAE:   {mae_iter:.3f}°C")
        print(f"  Bias:  {bias_iter:+.3f}°C")
        print(f"  <0.5°C: {pct_05:.1f}%")
        print(f"  <1.0°C: {pct_10:.1f}%")

        print("\n" + "-" * 50)
        print("COMPARISON (forecast issued at 10am):")
        print("-" * 50)
        print(f"  Single-stage @ sunrise (~13h): RMSE={rmse_sunrise:.3f}°C, <1°C: {(error_sunrise.abs() < 1).mean() * 100:.1f}%")
        print(f"  Single-stage @ 10am (~9h):     RMSE={rmse_10am_single:.3f}°C, <1°C: {(np.abs(error_single) < 1).mean() * 100:.1f}%")
        print(f"  Iterative (sunrise→10am):      RMSE={rmse_iter:.3f}°C, <1°C: {pct_10:.1f}%")
        print(f"  Mean lead time: {test_s3['hours_to_tw'].mean():.1f}h before twilight")

        # Save results
        output_file = RESULTS_PATH / "iterative_correction_results.csv"
        test_s3.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("No Stage 3 results generated!")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
