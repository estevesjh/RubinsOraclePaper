"""NBEATSx-Diff WITHOUT solar grid — native 15-min time series.

D(tn) = T(tn) - T(tn - 48 steps)  [48 steps = 12h at 15-min cadence]
Reconstruct: T_pred = D_pred + T_observed(tn - 48)

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    python forecast/run_nosolar.py
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
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(__file__))

from twilight.utils import load_data, get_sun_altitude

# ── Config ───────────────────────────────────────────────────────────────

DATA_PATH = Path("/sdf/home/e/esteves/sitcom-analysis/RubinThermal/data/temp_history_all_dec2025_sunrise_sunset.csv")
RESULTS_PATH = Path(__file__).parent.parent / "results"
MODEL_CACHE_PATH = RESULTS_PATH / "nbeats_diff_nosolar_cache"

TEST_START_DATE = pd.Timestamp("2025-01-01")
FREQ = "15min"
SAMPLES_PER_HOUR = 4

# Differencing lag: 12h = 48 steps (half a day in real time)
LAG_STEPS = 48

# NBEATSx
INPUT_SIZE = 96       # 24h lookback
HORIZON = 48          # 12h ahead (covers up to 12h lead)
MAX_STEPS = 500

# Eval lead times (hours before twilight)
LEAD_TIMES_HOURS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

# Features
HIST_EXOG = [
    "y_raw",
    "y_lag_4",       # 1h ago
    "y_lag_16",      # 4h ago
    "y_lag_48",      # 12h ago (the anchor)
    "y_lag_96",      # 24h ago
    "trend_2h",      # backward slope
]
FUTR_EXOG = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


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


def load_and_prepare():
    """Load native 15-min data, add features, find twilight events."""
    print("Loading data...")
    df = load_data(str(DATA_PATH))
    print(f"  Rows: {len(df)}, range: {df['ds'].min()} to {df['ds'].max()}")

    # Sun altitude for twilight detection
    print("Computing sun altitude...")
    df["alt_sun"] = get_sun_altitude(pd.DatetimeIndex(df["ds"]))

    # Detect twilight events: alt_sun crosses -15 (setting)
    alt = df["alt_sun"].values
    alt_prev = np.roll(alt, 1)
    alt_prev[0] = alt[0]
    adjusted = alt - (-15.0)
    adjusted_prev = np.roll(adjusted, 1)
    adjusted_prev[0] = adjusted[0]
    df["twilight_event"] = (adjusted_prev >= 0) & (adjusted < 0)

    # Features
    print("Building features...")
    df["y_raw"] = df["y"].copy()
    df["y_lag_4"] = df["y"].shift(4)
    df["y_lag_16"] = df["y"].shift(16)
    df["y_lag_48"] = df["y"].shift(LAG_STEPS)
    df["y_lag_96"] = df["y"].shift(96)

    # Backward OLS slope over 8 steps (2h)
    window = 8
    y_arr = df["y"].values
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
    df["trend_2h"] = slopes

    # Deterministic time features
    hour_frac = df["ds"].dt.hour + df["ds"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    doy = df["ds"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Differenced target
    df["D"] = df["y"] - df["y"].shift(LAG_STEPS)

    # Drop warmup
    n_warmup = INPUT_SIZE + LAG_STEPS
    df = df.iloc[n_warmup:].reset_index(drop=True)

    # Extract twilight events
    tw_mask = df["twilight_event"].values.astype(bool)
    tw_idxs = np.where(tw_mask)[0]
    tw_events = pd.DataFrame({
        "grid_idx": tw_idxs,
        "ds": df["ds"].iloc[tw_idxs].values,
        "y_actual": df["y"].iloc[tw_idxs].values,
    })

    print(f"  After prep: {len(df)} rows, {len(tw_events)} twilight events")
    return df, tw_events


def train_model(df, use_cache=True):
    """Train NBEATSx on D target."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    cache_path = MODEL_CACHE_PATH / "NBEATSx_diff_nosolar_v1"
    if use_cache and cache_path.exists():
        print(f"  Loading cached model from {cache_path.name}")
        nf = NeuralForecast.load(str(cache_path))
        nf.models[0].inference_windows_batch_size = 64
        return nf

    train_df = df[df["ds"] < TEST_START_DATE].copy()
    all_exog = HIST_EXOG + FUTR_EXOG
    nf_train = train_df[["ds", "D"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["D"]
    nf_train["unique_id"] = "temp"

    print(f"  Training on {len(nf_train)} samples...")
    print(f"  hist_exog: {HIST_EXOG}")
    print(f"  futr_exog: {FUTR_EXOG}")

    model = NBEATSx(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        hist_exog_list=HIST_EXOG,
        futr_exog_list=FUTR_EXOG,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=0.01,
        scaler_type="robust",
        enable_progress_bar=True,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[32, 32]],
        n_blocks=[1, 1, 1, 1],
    )

    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(nf_train)

    MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    nf.save(str(cache_path))
    print(f"  Cached to {cache_path.name}")
    return nf


def predict_and_evaluate(model, df, tw_events):
    """Predict at each twilight × lead time."""
    all_exog = HIST_EXOG + FUTR_EXOG
    y_arr = df["y"].values
    ds_arr = df["ds"].values

    tw_test = tw_events[tw_events["ds"] >= TEST_START_DATE].copy()
    print(f"  Test twilights: {len(tw_test)}")

    requests = []
    for _, ev in tw_test.iterrows():
        target_idx = ev["grid_idx"]
        for lead_h in LEAD_TIMES_HOURS:
            offset = int(lead_h * SAMPLES_PER_HOUR)
            issue_idx = target_idx - offset
            if issue_idx < INPUT_SIZE:
                continue
            if (target_idx - issue_idx) >= HORIZON:
                continue
            requests.append({
                "target_idx": target_idx,
                "issue_idx": issue_idx,
                "lead_h": lead_h,
                "actual_temp": ev["y_actual"],
                "tw_time": ev["ds"],
            })

    print(f"  Requests: {len(requests)}")

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
            hist_start = issue_idx - INPUT_SIZE
            futr_end = issue_idx + HORIZON
            if futr_end > len(df):
                continue

            uid = f"r_{batch_i}_{req['target_idx']}_{req['lead_h']}"
            hist_df = df.iloc[hist_start:issue_idx][["ds", "D"] + all_exog].copy()
            if hist_df[all_exog].isna().any().any():
                continue
            hist_df["y"] = hist_df["D"]
            hist_df["unique_id"] = uid

            futr_df = df.iloc[issue_idx:futr_end][["ds"] + FUTR_EXOG].copy()
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
            anchor_idx = req["target_idx"] - LAG_STEPS
            if anchor_idx < 0:
                continue
            anchor_temp = y_arr[anchor_idx]
            if np.isnan(anchor_temp):
                continue

            T_pred = D_pred + anchor_temp

            results.append({
                "twilight_time": req["tw_time"],
                "forecast_time": pd.Timestamp(ds_arr[req["issue_idx"]]),
                "lead_time_hours": req["lead_h"],
                "actual_temp": req["actual_temp"],
                "model": "NBEATSx-Diff-NoSolar",
                "forecast_temp": T_pred,
                "error": req["actual_temp"] - T_pred,
            })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("NBEATSx-Diff WITHOUT SOLAR GRID (native 15-min)")
    print(f"D = T(tn) - T(tn - {LAG_STEPS} steps = {LAG_STEPS/4}h)")
    print("=" * 70)

    df, tw_events = load_and_prepare()

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    model = train_model(df)

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    results = predict_and_evaluate(model, df, tw_events)

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

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH / "paper_results_nosolar.csv", index=False)
    rmse_3h = np.sqrt((results[results["lead_time_hours"] == 3.0]["error"] ** 2).mean())
    print(f"\n*** RMSE at 3h: {rmse_3h:.3f} deg ***")


if __name__ == "__main__":
    main()
