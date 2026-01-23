"""Paper Validation: Twilight Temperature Forecasting at Rubin Observatory.

Evaluation strategy:
- Training: Even days (day 0, 2, 4, ...)
- Testing: Odd days (day 1, 3, 5, ...) from Jan 1, 2025
- Lead times: 0.5h to 12h before twilight

Models:
- Persistence (use current temperature)
- Persistence-Twilight (use previous twilight temperature)
- Linear (hourly temps before twilight)
- Random Forest (same features)
- MLP (same features)
- NBEATSx (delta_T model from twilight_offset_validation)
"""

import gc
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
from tqdm import tqdm

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

import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.models import NBEATSx

from data.loader import load_data_linear_offset
from data.preprocessing import (
    add_time_features,
    add_trend_features,
    add_twilight_features,
    add_yearly_features,
    add_trend_2h,
    add_key_time_features,
)
from models.nbeats import load_and_prepare_data, predict_batch

from config import (
    FREQ,
    LEAD_TIMES_HOURS,
    TEST_START_DATE,
    DATA_PATH,
    RESULTS_PATH,
    SKLEARN_CACHE_PATH,
    NBEATS_CACHE_PATH,
    MAX_HOURS_BEFORE,
    NBEATS_HORIZON,
    NBEATS_INPUT_SIZE,
    NBEATS_MAX_STEPS,
    NBEATS_HIST_EXOG,
    NBEATS_FUTR_EXOG,
    PRED_TIMES,
    RF_CONFIG,
    MLP_CONFIG,
    RIDGE_ALPHA,
    SLOPE_FILE,
)

def get_twilight_events(df: pd.DataFrame) -> pd.DataFrame:
    """Get all twilight events."""
    return df[df["twilight_temp"].notna()].copy()


def get_preceding_sunrise(
    twilight_time: pd.Timestamp, df: pd.DataFrame
) -> pd.Timestamp:
    """Find the sunrise that precedes this twilight event."""
    sunrise_rows = df[
        (df["sunrise_temp"].notna())
        & (df["ds"] < twilight_time)
        & (df["ds"] >= twilight_time - pd.Timedelta(hours=18))
    ]
    if len(sunrise_rows) == 0:
        return None
    return sunrise_rows["ds"].max()


def build_twilight_features(
    df: pd.DataFrame, max_hours: int = MAX_HOURS_BEFORE
) -> pd.DataFrame:
    """Build feature matrix with temperatures at various hours before twilight."""
    twilight_df = df[df["twilight_temp"].notna()][["ds", "twilight_temp"]].copy()
    twilight_df = twilight_df.rename(columns={"ds": "twilight_time"})

    trend_cols = ["temp_mean_d1", "temp_mean_d2", "temp_mean_d3", "temp_trend_3d"]
    has_trend_features = all(col in df.columns for col in trend_cols)

    results = []
    for _, row in twilight_df.iterrows():
        twilight_time = row["twilight_time"]
        twilight_temp = row["twilight_temp"]

        day_data = {
            "twilight_time": twilight_time,
            "twilight_temp": twilight_temp,
        }

        for hours_before in range(1, max_hours + 1):
            target_time = twilight_time - pd.Timedelta(hours=hours_before)
            mask = (df["ds"] >= target_time - pd.Timedelta(minutes=10)) & (
                df["ds"] <= target_time + pd.Timedelta(minutes=10)
            )
            nearby = df[mask]
            if len(nearby) > 0:
                day_data[f"temp_h-{hours_before}"] = nearby["y"].mean()
            else:
                day_data[f"temp_h-{hours_before}"] = np.nan

        if has_trend_features:
            twilight_row = df[df["ds"] == twilight_time]
            if len(twilight_row) > 0:
                for col in trend_cols:
                    day_data[col] = twilight_row[col].iloc[0]
            else:
                for col in trend_cols:
                    day_data[col] = np.nan

        results.append(day_data)

    feature_df = pd.DataFrame(results)
    min_time = df["ds"].min()
    feature_df["day_of_data"] = (feature_df["twilight_time"] - min_time).dt.days
    feature_df["is_even_day"] = feature_df["day_of_data"] % 2 == 0

    return feature_df


def get_feature_cols_for_lead(
    lead_hours: float, max_hours: int = MAX_HOURS_BEFORE, include_trend: bool = True
) -> list:
    """Get feature column names for a specific lead time."""
    min_hour = int(np.ceil(lead_hours))
    features = [f"temp_h-{h}" for h in range(min_hour, max_hours + 1)]
    if include_trend:
        features += ["temp_mean_d1", "temp_mean_d2", "temp_mean_d3", "temp_trend_3d"]
    return features


# =============================================================================
# SKLEARN MODEL CACHING
# =============================================================================


def get_sklearn_cache_path(model_type: str, lead_hours: float) -> Path:
    """Get cache path for sklearn model."""
    return SKLEARN_CACHE_PATH / f"{model_type}_lead_{lead_hours:.1f}h.joblib"


def load_sklearn_model(model_type: str, lead_hours: float):
    """Load cached sklearn model if exists."""
    cache_path = get_sklearn_cache_path(model_type, lead_hours)
    if cache_path.exists():
        return joblib.load(cache_path)
    return None


def save_sklearn_model(model, model_type: str, lead_hours: float):
    """Save sklearn model to cache."""
    SKLEARN_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_path = get_sklearn_cache_path(model_type, lead_hours)
    joblib.dump(model, cache_path)


# =============================================================================
# BASELINE MODELS
# =============================================================================


def predict_persistence(df: pd.DataFrame, forecast_time: pd.Timestamp) -> float:
    """Persistence: use current temperature."""
    row = df[df["ds"] == forecast_time]
    if len(row) > 0:
        return row["y"].values[0]
    recent = df[df["ds"] <= forecast_time].tail(1)
    if len(recent) > 0:
        return recent["y"].values[0]
    return np.nan


def predict_persistence_twilight(
    twilight_time: pd.Timestamp, twilight_features: pd.DataFrame
) -> float:
    """Persistence-Twilight: use previous twilight temperature.

    This baseline predicts that today's twilight temp equals yesterday's.
    Independent of lead time - same prediction at any forecast horizon.
    """
    # Get previous twilight (sorted by twilight_time)
    prev_twilights = twilight_features[
        twilight_features["twilight_time"] < twilight_time
    ].sort_values("twilight_time")

    if len(prev_twilights) > 0:
        return prev_twilights["twilight_temp"].iloc[-1]
    return np.nan


def train_linear_model(feature_df: pd.DataFrame, lead_hours: float):
    """Train linear regression model for a specific lead time."""
    feature_cols = get_feature_cols_for_lead(lead_hours)
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    if not feature_cols:
        return None

    df_clean = feature_df[feature_cols + ["twilight_temp"]].dropna()
    if len(df_clean) < 50:
        return None

    X = df_clean[feature_cols].values
    y = df_clean["twilight_temp"].values

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y)
    return model


def train_rf_model(feature_df: pd.DataFrame, lead_hours: float):
    """Train Random Forest model for a specific lead time."""
    feature_cols = get_feature_cols_for_lead(lead_hours)
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    if not feature_cols:
        return None

    df_clean = feature_df[feature_cols + ["twilight_temp"]].dropna()
    if len(df_clean) < 50:
        return None

    X = df_clean[feature_cols].values
    y = df_clean["twilight_temp"].values

    model = RandomForestRegressor(**RF_CONFIG)
    model.fit(X, y)
    return model


def train_mlp_model(feature_df: pd.DataFrame, lead_hours: float):
    """Train MLP model for a specific lead time."""
    feature_cols = get_feature_cols_for_lead(lead_hours)
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    if not feature_cols:
        return None, None

    df_clean = feature_df[feature_cols + ["twilight_temp"]].dropna()
    if len(df_clean) < 50:
        return None, None

    X = df_clean[feature_cols].values
    y = df_clean["twilight_temp"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(**MLP_CONFIG)
    model.fit(X_scaled, y)
    return model, scaler


def predict_sklearn_batched(
    model,
    feature_df: pd.DataFrame,
    twilight_times: list[pd.Timestamp],
    lead_hours: float,
    scaler=None,
) -> dict[pd.Timestamp, float]:
    """Batched prediction for sklearn models."""
    feature_cols = get_feature_cols_for_lead(lead_hours)
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    if not feature_cols:
        return {}

    mask = feature_df["twilight_time"].isin(twilight_times)
    rows = feature_df[mask].copy()

    if len(rows) == 0:
        return {}

    X = rows[feature_cols].values
    valid_mask = ~np.any(np.isnan(X), axis=1)
    valid_rows = rows[valid_mask]
    X_valid = X[valid_mask]

    if len(X_valid) == 0:
        return {}

    if scaler is not None:
        X_valid = scaler.transform(X_valid)

    predictions = model.predict(X_valid)
    return dict(zip(valid_rows["twilight_time"], predictions))


# =============================================================================
# NBEATSx DELTA_T MODEL
# =============================================================================


def train_nbeats_model(
    df: pd.DataFrame,
    cutoff_time: pd.Timestamp,
    use_cache: bool = True,
):
    """Train NBEATSx model on delta_T target with 12h horizon."""
    # Cache path - v1 suffix indicates optimized features
    cutoff_str = cutoff_time.strftime("%Y%m%d")
    cache_path = NBEATS_CACHE_PATH / f"NBEATSx_deltaT_v1_{cutoff_str}"

    if use_cache and cache_path.exists():
        print(f"  Loading cached model from {cache_path.name}")
        nf = NeuralForecast.load(str(cache_path))
        nf.models[0].inference_windows_batch_size = 64
        return nf

    # Use feature configuration from config.py
    hist_exog = NBEATS_HIST_EXOG
    futr_exog = NBEATS_FUTR_EXOG

    # Prepare training data
    train_cutoff = cutoff_time - pd.Timedelta(days=1)
    train_df = df[df["ds"] < train_cutoff].copy()

    all_exog = hist_exog + futr_exog
    nf_train = train_df[["ds", "delta_T"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["delta_T"]
    nf_train["unique_id"] = "temp"

    if len(nf_train) < NBEATS_INPUT_SIZE * 2:
        print(f"  Insufficient data: {len(nf_train)} samples")
        return None

    print(f"  Training NBEATSx on {len(nf_train)} samples (horizon={NBEATS_HORIZON})...")

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
    NBEATS_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    nf.save(str(cache_path))
    print(f"  Cached to {cache_path.name}")

    return nf


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def main():
    """Run paper validation with even/odd day split."""
    print("=" * 70)
    print("PAPER VALIDATION: Twilight Temperature Forecasting")
    print(f"Using pre-trained delta_T model, predicting from {TEST_START_DATE.date()}")
    print("=" * 70)

    # Load data with linear offset (for sklearn models)
    print("\nLoading data...")
    df = load_data_linear_offset(DATA_PATH)
    df = add_time_features(df)
    df = add_yearly_features(df)
    df = add_twilight_features(df)
    df = add_trend_features(df)
    df["month"] = df["ds"].dt.month
    df["is_winter"] = df["month"].isin([6, 7, 8]).astype(int)
    df["temp_raw"] = df["y"].copy()

    # Load delta_T data (for NBEATSx delta model)
    # If SLOPE_FILE is set, use it for iterative slope correction
    if SLOPE_FILE is not None:
        print(f"Loading delta_T data with slope correction from {SLOPE_FILE}...")
    else:
        print("Loading delta_T data (flat baseline)...")
    df_delta = load_and_prepare_data(DATA_PATH, slope_file=SLOPE_FILE)

    # Add optimized v1 features for NBEATSx
    print("Adding optimized features...")
    df_delta = add_trend_2h(df_delta)
    df_delta = add_key_time_features(df_delta)

    # Merge delta_T columns into main df
    delta_cols = [
        "ds", "delta_T", "T_tw_last", "last_tw_time",
        "twilight_sin", "twilight_cos", "temp_last_sunrise",
    ]
    df_delta_subset = df_delta[delta_cols].copy()
    df = df.merge(df_delta_subset, on="ds", how="left", suffixes=("", "_delta"))

    # Even/Odd day split
    df["day_of_data"] = (df["ds"] - df["ds"].min()).dt.days
    train_df = df[df["day_of_data"] % 2 == 0].copy()
    test_df = df[df["day_of_data"] % 2 == 1].copy()

    print(f"Total rows: {len(df)}")
    print(f"Training rows (even days): {len(train_df)}")
    print(f"Testing rows (odd days): {len(test_df)}")

    test_twilights = get_twilight_events(test_df)
    print(f"Test twilight events: {len(test_twilights)}")
    print(f"Lead times: {LEAD_TIMES_HOURS}")

    all_results = []

    # ==========================================================================
    # BUILD TWILIGHT FEATURES (for Linear/RF/MLP)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BUILDING TWILIGHT FEATURES")
    print("=" * 70)

    print("\nBuilding twilight feature matrix...")
    twilight_features = build_twilight_features(df, max_hours=MAX_HOURS_BEFORE)
    print(f"  Total twilight events: {len(twilight_features)}")

    train_features = twilight_features[twilight_features["is_even_day"]].copy()
    test_features = twilight_features[~twilight_features["is_even_day"]].copy()

    print(f"  Training twilights (even days): {len(train_features)}")
    print(f"  Testing twilights (odd days): {len(test_features)}")

    # ==========================================================================
    # TRAIN ALL MODELS ONCE (on even days) - with caching
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING PHASE (on even days)")
    print("=" * 70)

    print("\nLoading/Training baseline models for each lead time...")
    linear_models = {}
    rf_models = {}
    mlp_models = {}
    mlp_scalers = {}

    for lead_hours in tqdm(LEAD_TIMES_HOURS, desc="Lead times"):
        linear_cached = load_sklearn_model("linear", lead_hours)
        rf_cached = load_sklearn_model("rf", lead_hours)
        mlp_cached = load_sklearn_model("mlp", lead_hours)

        if linear_cached is not None:
            linear_models[lead_hours] = linear_cached
        else:
            linear_models[lead_hours] = train_linear_model(train_features, lead_hours)
            if linear_models[lead_hours] is not None:
                save_sklearn_model(linear_models[lead_hours], "linear", lead_hours)

        if rf_cached is not None:
            rf_models[lead_hours] = rf_cached
        else:
            rf_models[lead_hours] = train_rf_model(train_features, lead_hours)
            if rf_models[lead_hours] is not None:
                save_sklearn_model(rf_models[lead_hours], "rf", lead_hours)

        if mlp_cached is not None:
            mlp_models[lead_hours], mlp_scalers[lead_hours] = mlp_cached
        else:
            mlp_models[lead_hours], mlp_scalers[lead_hours] = train_mlp_model(
                train_features, lead_hours
            )
            if mlp_models[lead_hours] is not None:
                save_sklearn_model(
                    (mlp_models[lead_hours], mlp_scalers[lead_hours]), "mlp", lead_hours
                )

    print(f"  Sklearn models cached at: {SKLEARN_CACHE_PATH}")

    # ==========================================================================
    # EVALUATION PHASE (on odd days)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION PHASE (on odd days)")
    print("=" * 70)

    print(f"\nCollecting prediction requests for {len(test_features)} twilights...")
    requests_by_lead = {lt: [] for lt in LEAD_TIMES_HOURS}

    for _, tw_row in tqdm(test_features.iterrows(), total=len(test_features), desc="Collecting"):
        twilight_time = tw_row["twilight_time"]
        actual_temp = tw_row["twilight_temp"]

        sunrise_time = get_preceding_sunrise(twilight_time, df)
        if sunrise_time is None:
            continue

        for lead_hours in LEAD_TIMES_HOURS:
            forecast_time = twilight_time - pd.Timedelta(hours=lead_hours)
            if forecast_time < sunrise_time:
                continue

            result_base = {
                "twilight_time": twilight_time,
                "forecast_time": forecast_time,
                "lead_time_hours": lead_hours,
                "actual_temp": actual_temp,
            }

            requests_by_lead[lead_hours].append({
                "twilight_time": twilight_time,
                "forecast_time": forecast_time,
                "result_base": result_base,
            })

    # Run batched sklearn predictions
    print("\nRunning batched sklearn predictions...")

    print("  Persistence...", end=" ", flush=True)
    for lead_hours, requests in requests_by_lead.items():
        for req in requests:
            pred = predict_persistence(df, req["forecast_time"])
            all_results.append(
                {**req["result_base"], "model": "Persistence", "forecast_temp": pred}
            )
    print("done")

    print("  Persistence-Twilight...", end=" ", flush=True)
    for lead_hours, requests in requests_by_lead.items():
        for req in requests:
            pred = predict_persistence_twilight(req["twilight_time"], twilight_features)
            all_results.append(
                {**req["result_base"], "model": "Persistence-Twilight", "forecast_temp": pred}
            )
    print("done")

    print("  Linear...", end=" ", flush=True)
    for lead_hours in LEAD_TIMES_HOURS:
        if linear_models[lead_hours] is None:
            continue
        twilight_times = [r["twilight_time"] for r in requests_by_lead[lead_hours]]
        preds = predict_sklearn_batched(
            linear_models[lead_hours], test_features, twilight_times, lead_hours
        )
        for req in requests_by_lead[lead_hours]:
            tt = req["twilight_time"]
            if tt in preds:
                all_results.append({
                    **req["result_base"],
                    "model": "Linear",
                    "forecast_temp": preds[tt],
                })
    print("done")

    print("  RandomForest...", end=" ", flush=True)
    for lead_hours in LEAD_TIMES_HOURS:
        if rf_models[lead_hours] is None:
            continue
        twilight_times = [r["twilight_time"] for r in requests_by_lead[lead_hours]]
        preds = predict_sklearn_batched(
            rf_models[lead_hours], test_features, twilight_times, lead_hours
        )
        for req in requests_by_lead[lead_hours]:
            tt = req["twilight_time"]
            if tt in preds:
                all_results.append({
                    **req["result_base"],
                    "model": "RandomForest",
                    "forecast_temp": preds[tt],
                })
    print("done")

    print("  MLP...", end=" ", flush=True)
    for lead_hours in LEAD_TIMES_HOURS:
        if mlp_models[lead_hours] is None:
            continue
        twilight_times = [r["twilight_time"] for r in requests_by_lead[lead_hours]]
        preds = predict_sklearn_batched(
            mlp_models[lead_hours],
            test_features,
            twilight_times,
            lead_hours,
            scaler=mlp_scalers[lead_hours],
        )
        for req in requests_by_lead[lead_hours]:
            tt = req["twilight_time"]
            if tt in preds:
                all_results.append({
                    **req["result_base"],
                    "model": "MLP",
                    "forecast_temp": preds[tt],
                })
    print("done")

    print(f"\nBaseline models: {len(all_results)} predictions")

    # ==========================================================================
    # NBEATSx DELTA_T VALIDATION (train new model with 12h horizon)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("NBEATSx DELTA_T VALIDATION")
    print(f"Training model with {NBEATS_HORIZON}-step horizon (12h)")
    print(f"Predicting from: {TEST_START_DATE.date()}")
    print("=" * 70)

    # Train model on data before TEST_START_DATE
    nbeats_delta_model = train_nbeats_model(
        df_delta,
        cutoff_time=TEST_START_DATE,
        use_cache=True,
    )
    nbeats_results = []
    offset_predictions = []  # For twilight_offset_predictions.csv

    if nbeats_delta_model is not None:
        tw_events = df_delta[df_delta["twilight_temp"].notna()].copy()
        tw_events = tw_events[tw_events["ds"] >= TEST_START_DATE]

        # Add previous twilight info
        tw_events["T_tw_prev"] = tw_events["twilight_temp"].shift(1)
        tw_events["prev_tw_time"] = tw_events["ds"].shift(1)

        print(f"NBEATSx twilight events: {len(tw_events)}")

        # =====================================================================
        # PART 1: Generate paper_results at all LEAD_TIMES_HOURS
        # =====================================================================
        print("Collecting paper_results requests at all lead times...")
        paper_requests = []
        paper_metadata = []

        for _, tw_row in tw_events.iterrows():
            tw_time = tw_row["ds"]
            tw_temp = tw_row["twilight_temp"]
            T_tw_last = tw_row["T_tw_prev"]  # Use PREVIOUS twilight temp as baseline
            last_tw_time = tw_row["last_tw_time"]

            if pd.isna(T_tw_last):
                continue  # Skip first twilight (no previous)

            for lead_hours in LEAD_TIMES_HOURS:
                forecast_time = tw_time - pd.Timedelta(hours=lead_hours)
                if forecast_time < df_delta["ds"].min() + pd.Timedelta(days=1):
                    continue

                # Calculate steps to twilight (15-min intervals)
                steps_to_twilight = int((tw_time - forecast_time).total_seconds() / 60 / 15)

                paper_requests.append({
                    "tw_time": tw_time,
                    "forecast_time": forecast_time,
                    "last_tw_time": last_tw_time,
                })
                paper_metadata.append({
                    "tw_time": tw_time,
                    "tw_temp": tw_temp,
                    "T_tw_last": T_tw_last,
                    "lead_hours": lead_hours,
                    "steps_to_twilight": steps_to_twilight,
                })

        print(f"  Paper requests: {len(paper_requests)}")
        print("Running paper_results predictions...")
        paper_preds = predict_batch(nbeats_delta_model, df_delta, paper_requests)

        for req_idx, result in paper_preds.items():
            meta = paper_metadata[req_idx]
            pred_df = result["predictions"].sort_values("ds").reset_index(drop=True)

            # Use step-based indexing (more efficient than timestamp matching)
            step = meta["steps_to_twilight"]
            if step > 0 and step <= len(pred_df):
                delta_T_pred = pred_df["delta_T_pred"].iloc[step - 1]

                # NBEATSx (operational): uses T_tw_last approximation
                pred_temp = delta_T_pred + meta["T_tw_last"]
                nbeats_results.append({
                    "twilight_time": meta["tw_time"],
                    "forecast_time": paper_requests[req_idx]["forecast_time"],
                    "lead_time_hours": meta["lead_hours"],
                    "actual_temp": meta["tw_temp"],
                    "model": "NBEATSx",
                    "forecast_temp": pred_temp,
                })

                # NBEATSx-Oracle: uses actual twilight temp (future info)
                # At twilight, Twilight-Trend = tw_temp, so Oracle = delta_T_pred + tw_temp
                pred_temp_oracle = delta_T_pred + meta["tw_temp"]
                nbeats_results.append({
                    "twilight_time": meta["tw_time"],
                    "forecast_time": paper_requests[req_idx]["forecast_time"],
                    "lead_time_hours": meta["lead_hours"],
                    "actual_temp": meta["tw_temp"],
                    "model": "NBEATSx-Oracle",
                    "forecast_temp": pred_temp_oracle,
                })

        if len(nbeats_results) > 0:
            results_temp = pd.DataFrame(nbeats_results)
            results_temp["error"] = results_temp["actual_temp"] - results_temp["forecast_temp"]
            at_3h = results_temp[results_temp["lead_time_hours"] == 3.0]
            if len(at_3h) > 0:
                rmse_3h = np.sqrt(np.mean(at_3h["error"] ** 2))
                print(f"  NBEATSx RMSE @ 3h: {rmse_3h:.2f}°C (n={len(at_3h)})")

        # =====================================================================
        # PART 2: Generate twilight_offset_predictions.csv at 30min intervals
        # =====================================================================
        # PRED_TIMES imported from config.py (48 predictions: 0.5h to 24h)

        print(f"\nCollecting offset_predictions requests ({len(PRED_TIMES)} times)...")
        offset_requests = []
        offset_metadata = []

        for _, tw_row in tw_events.iterrows():
            tw_time = tw_row["ds"]
            tw_temp = tw_row["twilight_temp"]
            T_tw_prev = tw_row["T_tw_prev"]
            prev_tw_time = tw_row["prev_tw_time"]
            last_tw_time = tw_row["last_tw_time"]
            temp_last_sunrise = tw_row.get("temp_last_sunrise", np.nan)

            if pd.isna(T_tw_prev):
                continue

            for pred_name, hours_before in PRED_TIMES:
                forecast_time = tw_time - pd.Timedelta(hours=hours_before)
                if forecast_time < df_delta["ds"].min() + pd.Timedelta(days=1):
                    continue

                offset_requests.append({
                    "tw_time": tw_time,
                    "forecast_time": forecast_time,
                    "last_tw_time": last_tw_time,
                })
                offset_metadata.append({
                    "tw_time": tw_time,
                    "tw_temp": tw_temp,
                    "T_tw_last": T_tw_prev,
                    "prev_tw_time": prev_tw_time,
                    "forecast_time": forecast_time,
                    "temp_last_sunrise": temp_last_sunrise,
                    "pred_time": pred_name,
                })

        print(f"  Offset requests: {len(offset_requests)}")
        print("Running offset_predictions...")
        offset_preds = predict_batch(nbeats_delta_model, df_delta, offset_requests)

        for req_idx, result in offset_preds.items():
            meta = offset_metadata[req_idx]
            tw_time = meta["tw_time"]
            tw_temp = meta["tw_temp"]
            T_tw_last = meta["T_tw_last"]
            prev_tw_time = meta["prev_tw_time"]
            forecast_time = meta["forecast_time"]
            pred_time = meta["pred_time"]
            temp_last_sunrise = meta["temp_last_sunrise"]

            pred_df = result["predictions"]

            for _, pred_row in pred_df.iterrows():
                target_time = pred_row["ds"]
                delta_T_pred = pred_row["delta_T_pred"]

                actual_row = df_delta[df_delta["ds"] == target_time]
                if len(actual_row) == 0:
                    continue

                actual_temp = actual_row["y"].values[0]
                h_from_tw = (target_time - prev_tw_time).total_seconds() / 3600
                hour_to_tw = (target_time - tw_time).total_seconds() / 3600
                temp_approx = delta_T_pred + T_tw_last
                res_tw = actual_temp - temp_approx
                trend_temp_3d = actual_row["temp_trend_3d"].values[0] if "temp_trend_3d" in actual_row.columns else np.nan

                # Extract rate features for Ridge correction
                rate_sunrise_to_midday = actual_row["rate_sunrise_to_midday"].values[0] if "rate_sunrise_to_midday" in actual_row.columns else np.nan
                rate_midday_to_twilight = actual_row["rate_midday_to_twilight"].values[0] if "rate_midday_to_twilight" in actual_row.columns else np.nan
                rate_twilight_to_midnight = actual_row["rate_twilight_to_midnight"].values[0] if "rate_twilight_to_midnight" in actual_row.columns else np.nan
                rate_midnight_to_sunrise = actual_row["rate_midnight_to_sunrise"].values[0] if "rate_midnight_to_sunrise" in actual_row.columns else np.nan
                trend_2h = actual_row["trend_2h"].values[0] if "trend_2h" in actual_row.columns else np.nan

                offset_predictions.append({
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
                    "trend_temp_3d": trend_temp_3d,
                    "trend_2h": trend_2h,
                    "rate_sunrise_to_midday": rate_sunrise_to_midday,
                    "rate_midday_to_twilight": rate_midday_to_twilight,
                    "rate_twilight_to_midnight": rate_twilight_to_midnight,
                    "rate_midnight_to_sunrise": rate_midnight_to_sunrise,
                })

        # Save twilight_offset_predictions.csv (or _iter1.csv if using slope correction)
        if offset_predictions:
            offset_df = pd.DataFrame(offset_predictions)
            if SLOPE_FILE is not None:
                offset_file = RESULTS_PATH / "twilight_offset_predictions_iter1.csv"
            else:
                offset_file = RESULTS_PATH / "twilight_offset_predictions.csv"
            offset_df.to_csv(offset_file, index=False)
            print(f"\n  Saved {len(offset_df)} rows to {offset_file.name}")

        del nbeats_delta_model
        gc.collect()
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    all_results.extend(nbeats_results)
    print(f"\nNBEATSx total predictions: {len(nbeats_results)}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df["error"] = results_df["actual_temp"] - results_df["forecast_temp"]

    # Save results
    RESULTS_PATH.mkdir(exist_ok=True)
    output_file = RESULTS_PATH / "paper_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY: RMSE (°C) by Model and Lead Time")
    print("=" * 90)

    models = results_df["model"].unique()
    lead_times = sorted(results_df["lead_time_hours"].unique())

    header = f"{'Model':<20}" + "".join([f"{lt:>6.1f}h" for lt in lead_times])
    print(header)
    print("-" * len(header))

    for model in models:
        row = f"{model:<20}"
        for lt in lead_times:
            data = results_df[
                (results_df["model"] == model) & (results_df["lead_time_hours"] == lt)
            ]
            if len(data) > 0:
                rmse = np.sqrt(np.mean(data["error"] ** 2))
                row += f"{rmse:>6.2f} "
            else:
                row += f"{'N/A':>6} "
        print(row)

    print("\n" + "-" * 50)
    print("DETAILED METRICS at 3h Lead Time:")
    print("-" * 50)
    at_3h = results_df[results_df["lead_time_hours"] == 3.0]
    for model in models:
        model_data = at_3h[at_3h["model"] == model]
        if len(model_data) > 0:
            rmse = np.sqrt(np.mean(model_data["error"] ** 2))
            mae = np.mean(np.abs(model_data["error"]))
            bias = model_data["error"].mean()
            pct_lt_1 = (np.abs(model_data["error"]) < 1.0).mean() * 100
            print(
                f"{model:<20}: RMSE={rmse:.2f}°C, MAE={mae:.2f}°C, Bias={bias:+.2f}°C, <1°C={pct_lt_1:.0f}%"
            )

    print("\n" + "=" * 90)
    print("DONE!")
    print("=" * 90)


if __name__ == "__main__":
    main()
