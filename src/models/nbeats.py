"""Twilight Offset Validation: Two-stage forecasting with slope recovery.

Methodology:
1. Training: NBEATSx predicts ΔT = T - Twilight-Trend (uses both twilight temps)
2. Operational: Approximate with Twilight-Trend ≈ T_tw_last (future unknown)
3. Correction: Ridge/MLP recovers slope from residuals to predict T_tw

Prediction Schedule (4 times per day, 8h horizon each):
1. Last Twilight → tw_last + 8h
2. Midnight → sunrise
3. Sunrise → sunrise + 8h
4. Midday → twilight
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
logging.getLogger("lightning").setLevel(logging.CRITICAL)

from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


@contextmanager
def suppress_stdout():
    """Suppress stdout/stderr."""
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


# Configuration - imported from config.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PATH,
    FREQ,
    RESULTS_PATH,
)
from config import (
    NBEATS_CACHE_PATH as MODEL_CACHE_PATH,
)
from config import (
    NBEATS_HORIZON as HORIZON,
)
from config import (
    NBEATS_INPUT_SIZE as INPUT_SIZE,
)
from config import (
    NBEATS_MAX_STEPS as MAX_STEPS,
)
from config import (
    TEST_START_DATE as TEST_START,
)


def load_and_prepare_data(data_path: Path, slope_file: Path = None) -> pd.DataFrame:
    """Load data and compute twilight-based features.

    Args:
        data_path: Path to the main CSV data file
        slope_file: Optional path to CSV with slope predictions (tw_time, slope columns).
                   If provided, uses slope for baseline correction. If None, uses flat baseline.

    Key boundary conditions:
    - At h_from_tw = 0 (last twilight): ΔT = 0 by construction
    - At h_from_tw = 24 (next twilight): ΔT = 0 by construction
    """
    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["timestamp"])

    # Use ACTUAL temperature y at twilight (ensures ΔT = 0 at boundary)
    tw_actual_temp = df["y"].where(df["twilight_temp"].notna())
    twilight_times = df["ds"].where(df["twilight_temp"].notna())

    # At twilight row: last_tw_time = current time, T_tw_last = y (NO shift)
    # This ensures h_from_tw = 0 and ΔT = 0 at twilight rows
    df["last_tw_time"] = twilight_times.ffill()
    df["T_tw_last"] = tw_actual_temp.ffill()

    # Next twilight (for Twilight-Trend calculation during training)
    df["next_tw_time"] = twilight_times.shift(-1).bfill()
    df["T_tw_next"] = tw_actual_temp.shift(-1).bfill()

    # Get sunrise times and temps
    sunrise_times = df["ds"].where(df["sunrise_temp"].notna())
    df["last_sunrise_time"] = sunrise_times.ffill()
    df["next_sunrise_time"] = sunrise_times.shift(-1).bfill()
    df["temp_last_sunrise"] = df["sunrise_temp"].ffill()

    # Compute h_from_tw (hours since last twilight)
    df["h_from_tw"] = (df["ds"] - df["last_tw_time"]).dt.total_seconds() / 3600

    # Compute actual twilight cycle length (varies slightly due to daylight changes)
    df["tw_cycle_hours"] = (
        df["next_tw_time"] - df["last_tw_time"]
    ).dt.total_seconds() / 3600

    # tw_progress: 0 at last twilight, 1 at next twilight (accounts for varying cycle length)
    df["tw_progress"] = df["h_from_tw"] / df["tw_cycle_hours"]

    # Compute Twilight-Trend (training - uses T_tw_next)
    # Uses tw_progress instead of h/24 to account for varying cycle length
    df["twilight_trend"] = (
        df["T_tw_last"] + (df["T_tw_next"] - df["T_tw_last"]) * df["tw_progress"]
    )

    # Compute ΔT (target for training)
    df["delta_T"] = df["y"] - df["twilight_trend"]

    # Operational approximation: Twilight-Trend ≈ T_tw_last + (slope/24) * h_from_tw
    # slope is per-twilight, loaded from slope_file if provided
    if slope_file is not None and Path(slope_file).exists():
        slope_df = pd.read_csv(slope_file)
        slope_df["tw_time"] = pd.to_datetime(slope_df["tw_time"])
        slope_map = dict(zip(slope_df["tw_time"], slope_df["slope"]))
        df["slope"] = df["next_tw_time"].map(slope_map).fillna(0)
        print(f"  Loaded slopes for {len(slope_map)} twilights from {slope_file}")
    else:
        df["slope"] = 0  # Flat baseline (iteration 0)

    df["twilight_trend_approx"] = df["T_tw_last"] + (df["slope"] / 24) * df["h_from_tw"]
    df["delta_T_approx"] = df["y"] - df["twilight_trend_approx"]

    # Add time features
    df["hour_sin"] = np.sin(2 * np.pi * df["ds"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["ds"].dt.hour / 24)
    df["month"] = df["ds"].dt.month
    df["is_winter"] = df["month"].isin([6, 7, 8]).astype(int)
    df["temp_raw"] = df["y"].copy()

    # 3-day trend features
    df["temp_mean_d1"] = df["y"].rolling(window=96, min_periods=1).mean()  # 24h
    df["temp_mean_d2"] = df["y"].rolling(window=192, min_periods=1).mean()  # 48h
    df["temp_mean_d3"] = df["y"].rolling(window=288, min_periods=1).mean()  # 72h
    df["temp_trend_3d"] = df["temp_mean_d1"] - df["temp_mean_d3"]

    # Lagged features
    df["y_lag_4"] = df["y"].shift(4).bfill()
    df["y_lag_12"] = df["y"].shift(12).bfill()
    df["y_lag_24"] = df["y"].shift(24).bfill()

    # === Twilight progress features (matching preprocessing.py) ===
    # Two-phase cycle: day (sunrise→twilight, 0→1) + night (twilight→sunrise, 1→2)

    # Determine if daytime: last sunrise is more recent than last twilight
    is_daytime = df["last_sunrise_time"] > df["last_tw_time"]
    is_daytime = is_daytime | df["last_tw_time"].isna()  # Handle edge case

    # Daylight duration: from last sunrise to next twilight
    daylight_seconds = (df["next_tw_time"] - df["last_sunrise_time"]).dt.total_seconds()

    # Night duration: from last twilight to next sunrise
    night_seconds = (df["next_sunrise_time"] - df["last_tw_time"]).dt.total_seconds()

    # Daytime progress: 0 (sunrise) to 1 (twilight)
    day_elapsed = (df["ds"] - df["last_sunrise_time"]).dt.total_seconds()
    day_progress = (day_elapsed / daylight_seconds).clip(0, 1)

    # Nighttime progress: 1 (twilight) to 2 (next sunrise)
    night_elapsed = (df["ds"] - df["last_tw_time"]).dt.total_seconds()
    night_progress = 1 + (night_elapsed / night_seconds).clip(0, 1)

    # Combined progress: 0→1 (day) then 1→2 (night)
    df["twilight_progress"] = np.where(is_daytime, day_progress, night_progress)

    # Twilight-relative features (full cycle: 0→1→2 maps to cos/sin over π)
    df["twilight_sin"] = np.sin(np.pi * df["twilight_progress"])
    df["twilight_cos"] = np.cos(np.pi * df["twilight_progress"])

    # Drop rows with missing values
    df = df.dropna(subset=["y", "delta_T", "T_tw_last"])

    return df


def get_twilight_events(df: pd.DataFrame) -> pd.DataFrame:
    """Get all twilight events."""
    return df[df["twilight_temp"].notna()].copy()


def get_sunrise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Get all sunrise events."""
    return df[df["sunrise_temp"].notna()].copy()


def train_nbeats_model(
    df: pd.DataFrame,
    cutoff_time: pd.Timestamp,
    use_cache: bool = True,
):
    """Train NBEATSx model on ΔT target."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    # Cache path
    cutoff_str = cutoff_time.strftime("%Y%m%d")
    cache_path = MODEL_CACHE_PATH / f"NBEATSx_deltaT_{cutoff_str}"

    if use_cache and cache_path.exists():
        print(f"  Loading cached model from {cache_path.name}")
        nf = NeuralForecast.load(str(cache_path))
        nf.models[0].inference_windows_batch_size = 64
        return nf

    # Optimized v1 features (Jan 2025) - must match paper_validation.py
    hist_exog = [
        "temp_raw",
        "temp_last_sunrise",
        "trend_2h",
        "temp_trend_3d",
        "rate_sunrise_to_midday",
        "rate_midday_to_twilight",
        "rate_twilight_to_midnight",
        "rate_midnight_to_sunrise",
    ]
    futr_exog = ["twilight_cos"]

    # Prepare training data
    train_cutoff = cutoff_time - pd.Timedelta(days=1)
    train_df = df[df["ds"] < train_cutoff].copy()

    all_exog = hist_exog + futr_exog
    nf_train = train_df[["ds", "delta_T"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["delta_T"]
    nf_train["unique_id"] = "temp"

    if len(nf_train) < INPUT_SIZE * 2:
        print(f"  Insufficient data: {len(nf_train)} samples")
        return None

    print(f"  Training NBEATSx on {len(nf_train)} samples...")

    model = NBEATSx(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=0.01,
        scaler_type="robust",
        enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "exogenous"],
        mlp_units=3 * [[32, 32]],
        n_blocks=[1, 1, 1],
    )

    with suppress_stdout():
        nf = NeuralForecast(models=[model], freq=FREQ)
        nf.fit(nf_train)

    # Save cache
    MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    nf.save(str(cache_path))
    print(f"  Cached to {cache_path.name}")

    return nf


def _prepare_single_request(args):
    """Prepare historical and future data for a single prediction request (for parallel processing)."""
    idx, req, df_values, df_index, all_exog, input_size, horizon, freq = args

    forecast_time = req["forecast_time"]
    tw_time = req["tw_time"]
    last_tw_time = req["last_tw_time"]
    unique_id = f"pred_{idx}"

    # Use numpy indexing for speed
    mask = df_index < forecast_time
    recent_idx = np.where(mask)[0]
    if len(recent_idx) < input_size:
        return None

    # Get tail of recent data
    start_idx = max(0, len(recent_idx) - 5 * input_size)
    recent_indices = recent_idx[start_idx:]

    # Extract columns
    cols_needed = ["ds", "delta_T", "delta_T_approx", "sunrise_temp"] + all_exog
    col_indices = {
        col: i for i, col in enumerate(df_values.dtype.names) if col in cols_needed
    }

    # Build nf_recent from structured array
    nf_data = {
        col: df_values[col][recent_indices]
        for col in cols_needed
        if col in df_values.dtype.names
    }
    nf_recent = pd.DataFrame(nf_data)
    nf_recent = nf_recent.dropna().tail(5 * input_size).copy()

    if len(nf_recent) < input_size:
        return None

    # Adjust delta_T_approx
    nf_recent["delta_T_approx"] = np.where(
        nf_recent["ds"] > last_tw_time,
        nf_recent["delta_T_approx"],
        nf_recent["delta_T"],
    )
    nf_recent["y"] = nf_recent["delta_T_approx"]
    nf_recent["unique_id"] = unique_id

    # Get sunrise times
    sunrise_mask = (
        nf_recent["sunrise_temp"].notna()
        if "sunrise_temp" in nf_recent.columns
        else pd.Series([False] * len(nf_recent))
    )
    if sunrise_mask.any():
        last_sunrise_time = nf_recent.loc[sunrise_mask, "ds"].iloc[-1]
    else:
        last_sunrise_time = last_tw_time - pd.Timedelta(hours=12)

    # Future sunrise (simplified - use last_sunrise_time + 24 as fallback)
    next_sunrise_time = last_sunrise_time + pd.Timedelta(hours=24)

    # Future exogenous - VECTORIZED
    training_end_ds = nf_recent["ds"].max()
    futr_end = forecast_time + pd.Timedelta(minutes=15 * horizon)
    future_timestamps = pd.date_range(training_end_ds, futr_end, freq=freq)
    futr_df = pd.DataFrame({"unique_id": unique_id, "ds": future_timestamps})

    # Vectorized twilight_cos computation
    is_day = last_sunrise_time > last_tw_time
    ts_array = futr_df["ds"].values

    if is_day:
        daylight_secs = (tw_time - last_sunrise_time).total_seconds()
        if daylight_secs > 0:
            elapsed = (pd.to_datetime(ts_array) - last_sunrise_time).total_seconds()
            progress = np.clip(elapsed / daylight_secs, 0, 1)
        else:
            progress = np.zeros(len(ts_array))
    else:
        night_secs = (next_sunrise_time - last_tw_time).total_seconds()
        if night_secs > 0:
            elapsed = (pd.to_datetime(ts_array) - last_tw_time).total_seconds()
            progress = 1 + np.clip(elapsed / night_secs, 0, 1)
        else:
            progress = np.ones(len(ts_array))

    futr_df["twilight_cos"] = np.cos(np.pi * progress)

    # Keep only needed columns for hist
    keep_cols = ["ds", "y", "unique_id"] + [
        c for c in all_exog if c in nf_recent.columns
    ]
    nf_recent = nf_recent[keep_cols]

    return {
        "hist_df": nf_recent,
        "futr_df": futr_df,
        "unique_id": unique_id,
        "req_idx": idx,
        "forecast_time": forecast_time,
    }


def predict_batch(
    model,
    df: pd.DataFrame,
    prediction_requests: list[dict],
) -> dict:
    """Make batched predictions for multiple forecast origins (matrix-based).

    Uses numpy arrays for ~180x faster data preparation than pandas-based approach.
    """
    from collections import defaultdict

    # Feature definitions (must include twilight_cos for hist_exog)
    hist_exog = [
        "temp_raw",
        "temp_last_sunrise",
        "trend_2h",
        "temp_trend_3d",
        "rate_sunrise_to_midday",
        "rate_midday_to_twilight",
        "rate_twilight_to_midnight",
        "rate_midnight_to_sunrise",
        "twilight_cos",
    ]
    feature_cols = ["delta_T", "delta_T_approx"] + hist_exog

    # Pre-convert to numpy for fast indexing
    df_sorted = df.sort_values("ds").reset_index(drop=True)
    t0 = df_sorted["ds"].iloc[0]
    timestamps = df_sorted["ds"].values
    features = df_sorted[feature_cols].values
    sunrise_temps = df_sorted["sunrise_temp"].values

    # Pre-compute sunrise indices for fast lookup
    sunrise_mask = ~np.isnan(sunrise_temps)
    sunrise_indices = np.where(sunrise_mask)[0]
    sunrise_times = timestamps[sunrise_indices]

    FREQ_MINUTES = 15
    FREQ_SECONDS = FREQ_MINUTES * 60

    def time_to_idx(t):
        return int((t - t0).total_seconds() // FREQ_SECONDS)

    def get_last_sunrise_before(idx):
        pos = np.searchsorted(sunrise_indices, idx, side="right") - 1
        if pos >= 0:
            return sunrise_times[pos], sunrise_indices[pos]
        return None, None

    # Group requests by tw_time
    requests_by_tw = defaultdict(list)
    for idx, req in enumerate(prediction_requests):
        requests_by_tw[req["tw_time"]].append((idx, req))

    # Pre-allocate matrices
    n_requests = len(prediction_requests)
    n_hist_features = len(feature_cols)

    hist_matrix = np.full((n_requests, INPUT_SIZE, n_hist_features), np.nan)
    futr_matrix = np.full((n_requests, HORIZON, 1), np.nan)
    hist_timestamps = np.empty((n_requests, INPUT_SIZE), dtype="datetime64[ns]")
    futr_timestamps = np.empty((n_requests, HORIZON), dtype="datetime64[ns]")
    valid_mask = np.zeros(n_requests, dtype=bool)
    forecast_times_out = [None] * n_requests

    futr_offsets = np.arange(1, HORIZON + 1) * np.timedelta64(FREQ_MINUTES, "m")

    # Process each twilight group
    for tw_time, tw_requests in requests_by_tw.items():
        last_tw_time = tw_requests[0][1]["last_tw_time"]
        last_tw_idx = time_to_idx(last_tw_time)
        tw_idx = time_to_idx(tw_time)

        last_sunrise_time, last_sunrise_idx = get_last_sunrise_before(tw_idx)

        if last_sunrise_time is None:
            continue

        last_sunrise_time = pd.Timestamp(last_sunrise_time)
        # next_sunrise = last_sunrise + 24h
        next_sunrise_time = last_sunrise_time + pd.Timedelta(hours=24)

        is_day = last_sunrise_idx > last_tw_idx

        if is_day:
            ref_time = last_sunrise_time
            base_progress = 0
            cycle_secs = (tw_time - last_sunrise_time).total_seconds()
        else:
            ref_time = last_tw_time
            base_progress = 1
            cycle_secs = (next_sunrise_time - last_tw_time).total_seconds()
        cycle_secs = max(cycle_secs, 1)

        # Process each request in the group
        for idx, req in tw_requests:
            forecast_time = req["forecast_time"]
            end_idx = time_to_idx(forecast_time)
            start_idx = end_idx - INPUT_SIZE

            if start_idx < 0 or end_idx > len(features):
                continue

            hist_slice = features[start_idx:end_idx]

            if np.isnan(hist_slice).any():
                continue

            hist_slice = hist_slice.copy()

            # Adjust delta_T_approx: use delta_T for rows before last_tw_time
            rows_before_tw = max(0, min(INPUT_SIZE, last_tw_idx - start_idx))
            if rows_before_tw > 0:
                hist_slice[:rows_before_tw, 1] = hist_slice[:rows_before_tw, 0]

            hist_matrix[idx] = hist_slice
            hist_timestamps[idx] = timestamps[start_idx:end_idx]

            # Future timestamps start from end of history
            training_end = timestamps[end_idx - 1]
            training_end_np = np.datetime64(training_end)
            futr_timestamps[idx] = training_end_np + futr_offsets

            # twilight_cos calculation
            training_end_ts = pd.Timestamp(training_end)
            elapsed = (
                (training_end_ts - ref_time).total_seconds()
                + np.arange(1, HORIZON + 1) * FREQ_SECONDS
            )
            progress = base_progress + np.clip(elapsed / cycle_secs, 0, 1)
            futr_matrix[idx, :, 0] = np.cos(np.pi * progress)

            valid_mask[idx] = True
            forecast_times_out[idx] = forecast_time

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return {}

    # Convert to DataFrames for model input
    combined_hist = pd.DataFrame(
        {
            "unique_id": np.repeat(
                [f"pred_{i}" for i in valid_indices], INPUT_SIZE
            ),
            "ds": hist_timestamps[valid_indices].ravel(),
            "y": hist_matrix[valid_indices, :, 1].ravel(),
            **{
                col: hist_matrix[valid_indices, :, i].ravel()
                for i, col in enumerate(feature_cols)
            },
        }
    )

    combined_futr = pd.DataFrame(
        {
            "unique_id": np.repeat(
                [f"pred_{i}" for i in valid_indices], HORIZON
            ),
            "ds": futr_timestamps[valid_indices].ravel(),
            "twilight_cos": futr_matrix[valid_indices, :, 0].ravel(),
        }
    )

    prediction_info = [
        {
            "unique_id": f"pred_{i}",
            "req_idx": i,
            "forecast_time": forecast_times_out[i],
        }
        for i in valid_indices
    ]

    # Run model inference
    with suppress_stdout():
        fc = model.predict(combined_hist, futr_df=combined_futr)

    fc = fc.reset_index()
    model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]

    # Extract results - use groupby for speed
    results = {}
    fc_grouped = {uid: group for uid, group in fc.groupby("unique_id")}

    for info in prediction_info:
        uid = info["unique_id"]
        if uid in fc_grouped:
            uid_fc = fc_grouped[uid].sort_values("ds").reset_index(drop=True)
            results[info["req_idx"]] = {
                "forecast_time": info["forecast_time"],
                "predictions": uid_fc[["ds", model_col]].rename(
                    columns={model_col: "delta_T_pred"}
                ),
            }

    return results


def main():
    """Run twilight offset validation."""
    print("=" * 70)
    print("TWILIGHT OFFSET VALIDATION")
    print("Two-stage forecasting with slope recovery")
    print("=" * 70)

    # Load data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Get twilight events for test period
    twilight_events = get_twilight_events(df)
    test_twilights = twilight_events[twilight_events["ds"] >= TEST_START].copy()
    print(f"\nTest twilight events: {len(test_twilights)}")

    # Train model (on data before test period)
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    model = train_nbeats_model(df, TEST_START, use_cache=True)
    if model is None:
        print("Failed to train model!")
        return

    # Generate forecasts for test period
    print("\n" + "=" * 70)
    print("GENERATING FORECASTS")
    print("=" * 70)

    # Step 1: Collect all prediction requests
    print("  Collecting prediction requests...")
    all_requests = []
    request_metadata = []

    # Get PREVIOUS twilight temperature and time for each twilight
    test_twilights = test_twilights.copy()
    test_twilights["T_tw_prev"] = test_twilights["twilight_temp"].shift(1)
    test_twilights["prev_tw_time"] = test_twilights["ds"].shift(1)

    for _, tw_row in test_twilights.iterrows():
        tw_time = tw_row["ds"]
        tw_temp = tw_row["twilight_temp"]
        # Use PREVIOUS twilight temp as baseline, not current
        T_tw_last = tw_row["T_tw_prev"]
        prev_tw_time = tw_row["prev_tw_time"]
        if pd.isna(T_tw_last):
            continue  # Skip first twilight (no previous)
        last_tw_time = tw_row["last_tw_time"]
        temp_last_sunrise = tw_row["temp_last_sunrise"]
        last_sunrise_time = tw_row["last_sunrise_time"]

        # 8 prediction times per day (every 3 hours):
        # Issue forecasts at 24, 21, 18, 15, 12, 9, 6, 3 hours before twilight
        prediction_times = [
            ("24h", tw_time - pd.Timedelta(hours=24)),
            ("21h", tw_time - pd.Timedelta(hours=21)),
            ("18h", tw_time - pd.Timedelta(hours=18)),
            ("15h", tw_time - pd.Timedelta(hours=15)),
            ("12h", tw_time - pd.Timedelta(hours=12)),
            ("9h", tw_time - pd.Timedelta(hours=9)),
            ("6h", tw_time - pd.Timedelta(hours=6)),
            ("3h", tw_time - pd.Timedelta(hours=3)),
        ]

        for pred_name, forecast_time in prediction_times:
            if pd.isna(forecast_time):
                continue
            if forecast_time < df["ds"].min() + pd.Timedelta(days=1):
                continue

            all_requests.append(
                {
                    "forecast_time": forecast_time,
                    "tw_time": tw_time,
                    "last_tw_time": last_tw_time,
                    "pred_time": pred_name,
                }
            )
            request_metadata.append(
                {
                    "tw_time": tw_time,
                    "tw_temp": tw_temp,
                    "T_tw_last": T_tw_last,
                    "prev_tw_time": prev_tw_time,
                    "forecast_time": forecast_time,
                    "temp_last_sunrise": temp_last_sunrise,
                    "pred_time": pred_name,
                }
            )

    print(f"  Total requests: {len(all_requests)}")

    # Step 2: Make single batched prediction
    print("  Making batched predictions...")
    preds = predict_batch(model, df, all_requests)
    print(f"  Got {len(preds)} prediction sets")

    # Step 3: Process results
    print("  Processing results...")
    all_results = []

    for req_idx, meta in enumerate(request_metadata):
        if req_idx not in preds:
            continue

        pred_df = preds[req_idx]["predictions"]
        tw_time = meta["tw_time"]
        tw_temp = meta["tw_temp"]
        T_tw_last = meta["T_tw_last"]
        prev_tw_time = meta["prev_tw_time"]
        forecast_time = meta["forecast_time"]
        temp_last_sunrise = meta["temp_last_sunrise"]
        pred_time = meta["pred_time"]

        # Merge with actual data
        for _, pred_row in pred_df.iterrows():
            target_time = pred_row["ds"]
            delta_T_pred = pred_row["delta_T_pred"]

            # Get actual values
            actual_row = df[df["ds"] == target_time]
            if len(actual_row) == 0:
                continue

            actual_temp = actual_row["y"].values[0]

            # h_from_tw: hours since PREVIOUS twilight (not current)
            # At twilight, this should be ~24 hours
            h_from_tw = (target_time - prev_tw_time).total_seconds() / 3600

            # hour_to_tw: hours until current twilight (negative before, 0 at twilight)
            hour_to_tw = (target_time - tw_time).total_seconds() / 3600

            # Compute approximated prediction (using T_tw_last only)
            temp_approx = delta_T_pred + T_tw_last

            # Compute residual
            res_tw = actual_temp - temp_approx

            all_results.append(
                {
                    "tw_date": tw_time.date(),
                    "tw_time": tw_time,
                    "tw_temp": tw_temp,
                    "T_tw_last": T_tw_last,
                    "pred_time": pred_time,
                    "forecast_time": forecast_time,
                    "target_time": target_time,
                    "h_from_tw": h_from_tw,
                    "hour_to_tw": hour_to_tw,
                    "temp_actual": actual_temp,
                    "temp_approx": temp_approx,
                    "delta_T_pred": delta_T_pred,
                    "res_tw": res_tw,
                    "temp_last_sunrise": temp_last_sunrise,
                    "trend_temp_3d": actual_row["temp_trend_3d"].values[0],
                }
            )

    # Save results
    results_df = pd.DataFrame(all_results)
    RESULTS_PATH.mkdir(exist_ok=True)
    output_file = RESULTS_PATH / "twilight_offset_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total predictions: {len(results_df)}")

    # Quick evaluation
    print("\n" + "=" * 70)
    print("QUICK EVALUATION")
    print("=" * 70)

    # RMSE at twilight (hour_to_tw = 0)
    tw_preds = results_df[results_df["hour_to_tw"].abs() < 0.5]
    if len(tw_preds) > 0:
        rmse_approx = np.sqrt(
            np.mean((tw_preds["temp_actual"] - tw_preds["temp_approx"]) ** 2)
        )
        mae_approx = np.mean(np.abs(tw_preds["temp_actual"] - tw_preds["temp_approx"]))
        print("\nAt twilight (approximated, no correction):")
        print(f"  RMSE: {rmse_approx:.2f}°C")
        print(f"  MAE:  {mae_approx:.2f}°C")
        print(f"  Samples: {len(tw_preds)}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
