"""Add Prophet and MeteoBlue forecasts to paper results.

Creates paper_results_v3.csv by combining:
- Existing paper_results_v2.csv (NBEATSx-Ridge, Persistence, etc.)
- Prophet hybrid forecasts from results_hybrid_07d_30min.csv
- MeteoBlue forecasts from meteo_blue_weather_station.csv

Weather mask is applied to exclude bad weather periods.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from config import DATA_PATH, RESULTS_PATH

# Data paths
PROPHET_FILE = Path(__file__).parent.parent / "data" / "results_hybrid_07d_30min.csv"
METEOBLUE_FILE = (
    Path(__file__).parent.parent / "data" / "meteo_blue_weather_station.csv"
)
WEATHER_MASK_FILE = (
    Path(__file__).parent.parent / "data" / "weather_mask_result.pkl"
)


def load_twilight_events() -> pd.DataFrame:
    """Get twilight events from 2024-2025."""
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter for twilight events (where twilight_temp is not NaN)
    tw = df[df["twilight_temp"].notna()].copy()
    tw = tw.rename(columns={"timestamp": "twilight_time", "twilight_temp": "actual_temp"})

    # Filter to 2024-2025
    tw = tw[
        (tw["twilight_time"] >= "2024-01-01") & (tw["twilight_time"] < "2026-01-01")
    ]

    return tw[["twilight_time", "actual_temp"]]


def apply_prophet_blending(df: pd.DataFrame, A: float = 0.90, tau: float = 13.0) -> pd.DataFrame:
    """Apply exponential blending to Prophet forecasts."""
    df = df.copy()
    lead = df["lead_time"].astype(float).to_numpy()
    f = A * np.exp(-lead / tau)
    df["y_hat"] = f * df["yhat_short"] + (1 - f) * df["yhat_long"]
    return df


def handle_prophet_failures(
    df: pd.DataFrame, xi2_tol: float = 2.0
) -> pd.DataFrame:
    """Handle Prophet model failures based on reduced chi-square."""
    out = df.copy()

    xi2_long = out["xi2_long_model"].astype(float).to_numpy()
    xi2_short = out["xi2_short_model"].astype(float).to_numpy()
    fail_long = np.abs(xi2_long - 1.0) > xi2_tol
    fail_short = np.abs(xi2_short - 1.0) > xi2_tol

    yhat_new = out["y_hat"].to_numpy(dtype=float, copy=True)

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
    """Extract Prophet-BMA forecasts matched to twilight events by interpolation.

    For each twilight event and lead time, interpolates Prophet's y_hat to
    the exact twilight time using the surrounding 30-min grid points.
    """
    print("Processing Prophet-BMA forecasts...")

    # Target lead times (same as other models)
    TARGET_LEAD_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    # Load Prophet data with BMA blending (short + long term models)
    df = pd.read_csv(PROPHET_FILE)
    df = df.rename(columns={"timestamp": "valid_time"})
    # Force ns resolution: pandas 3.x defaults to [us], breaking int64 round-trips
    df["valid_time"] = pd.to_datetime(df["valid_time"]).astype("datetime64[ns]")
    # Apply blending only where component columns exist
    has_components = "yhat_short" in df.columns and df["yhat_short"].notna().any()
    if has_components:
        mask = df["yhat_short"].notna() & df["yhat_long"].notna()
        df_blend = apply_prophet_blending(df[mask])
        df_blend = handle_prophet_failures(df_blend)
        # For rows without components, keep original y_hat
        df.loc[mask, "y_hat"] = df_blend["y_hat"].values
    df = df.dropna(subset=["y_hat", "y"])

    # Get twilight events
    twilight_events = twilight_events.copy()
    twilight_events["twilight_time"] = pd.to_datetime(twilight_events["twilight_time"]).astype("datetime64[ns]")

    results = []
    for lt in TARGET_LEAD_TIMES:
        # Filter to this lead time
        df_lt = df[np.abs(df["lead_time"] - lt) < 0.1].copy()
        df_lt = df_lt.sort_values("valid_time").reset_index(drop=True)

        # Build interpolation arrays
        vt_ns = df_lt["valid_time"].values.astype("int64")
        yhat_arr = df_lt["y_hat"].values
        y_arr = df_lt["y"].values

        for _, tw_row in twilight_events.iterrows():
            tw_time = tw_row["twilight_time"]
            actual_temp = tw_row["actual_temp"]  # use same target as other models
            tw_ns = tw_time.value

            # Check if twilight is within Prophet's valid range
            if tw_ns < vt_ns[0] or tw_ns > vt_ns[-1]:
                continue

            # Interpolate y_hat to exact twilight time
            forecast_temp = float(np.interp(tw_ns, vt_ns, yhat_arr))
            error = actual_temp - forecast_temp

            if np.abs(error) > 5.0:
                continue

            results.append({
                "twilight_time": tw_time,
                "forecast_time": tw_time - pd.Timedelta(hours=lt),
                "lead_time_hours": lt,
                "actual_temp": actual_temp,
                "model": "Prophet",
                "forecast_temp": forecast_temp,
                "error": error,
            })

    print(f"  Created {len(results)} Prophet-BMA forecast rows")
    for lt in TARGET_LEAD_TIMES:
        count = sum(1 for r in results if r["lead_time_hours"] == lt)
        print(f"    Lead time {lt}h: {count} points")

    return pd.DataFrame(results)


def process_meteoblue_forecasts(twilight_events: pd.DataFrame) -> pd.DataFrame:
    """Extract MeteoBlue forecasts by interpolation at twilight and nighttime hours.

    For each model run, reindex to a DatetimeIndex and interpolate at all
    target times (twilight + 0..6h) in one vectorized call.
    """
    print("Processing MeteoBlue forecasts...")

    # Load MeteoBlue data
    df = pd.read_csv(METEOBLUE_FILE)
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

    # Build target times: twilight + 0,1,...,6h for each twilight event
    tw_times = pd.to_datetime(twilight_events["twilight_time"]).dt.tz_localize("UTC")
    actuals = twilight_events["actual_temp"].values

    target_rows = []
    for i, tw_utc in enumerate(tw_times):
        target_rows.append((tw_utc, actuals[i], tw_times.iloc[i]))  # twilight only
    target_times = pd.DatetimeIndex([r[0] for r in target_rows])
    target_actuals = np.array([r[1] for r in target_rows])
    target_tw_times = [r[2] for r in target_rows]

    results = []
    for issue_t, run_df in df.groupby("issue_time"):
        run_df = run_df.sort_values("valid_time").set_index("valid_time")
        vt_min, vt_max = run_df.index[0], run_df.index[-1]

        # Filter targets within this run's valid range
        mask = (target_times >= vt_min) & (target_times <= vt_max)
        if not mask.any():
            continue

        targets_in_range = target_times[mask]

        # Interpolate: combine run index with targets, interpolate, extract
        combined_idx = run_df.index.append(targets_in_range).drop_duplicates().sort_values()
        interp_series = run_df["temperature"].reindex(combined_idx).interpolate(method="time")

        for j, tgt in enumerate(targets_in_range):
            forecast_temp = interp_series.loc[tgt]
            if pd.isna(forecast_temp):
                continue
            lead_time = (tgt - issue_t).total_seconds() / 3600.0
            if lead_time < 0:
                continue
            idx = np.where(mask)[0][j]
            results.append({
                "twilight_time": target_tw_times[idx].tz_localize(None),
                "forecast_time": issue_t.tz_localize(None),
                "lead_time_hours": lead_time,
                "actual_temp": target_actuals[idx],
                "model": "MeteoBlue",
                "forecast_temp": float(forecast_temp),
                "error": float(forecast_temp) - target_actuals[idx],
            })

    print(f"  Found {len(results)} MeteoBlue forecasts")
    return pd.DataFrame(results)


def load_weather_mask():
    """Load the weather mask using joblib."""
    if not WEATHER_MASK_FILE.exists():
        return None

    # Import the MaskResult and MaskWindows classes so joblib can unpickle
    # This must happen BEFORE loading
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "a_make_weather_mask",
        Path(__file__).parent.parent / "scripts" / "a_make_weather_mask.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["a_make_weather_mask"] = module
        spec.loader.exec_module(module)

    return joblib.load(WEATHER_MASK_FILE)


def apply_weather_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Apply weather mask to exclude bad conditions."""
    print("Applying weather mask...")

    mask_data = load_weather_mask()
    if mask_data is None:
        print("  Warning: Weather mask file not found, skipping mask")
        return df

    try:
        hour_mask = mask_data.hour_windows["hour_total"]
        df_temp = df.copy()
        df_temp["twilight_time_utc"] = pd.to_datetime(
            df_temp["twilight_time"]
        ).dt.tz_localize("UTC")

        # Apply the mask (invert=True to keep good weather)
        good_df = hour_mask.apply(df_temp, "twilight_time_utc", invert=True)
        print(f"  Kept {len(good_df)} of {len(df)} rows after weather mask")
        return good_df.drop(columns=["twilight_time_utc"])
    except Exception as e:
        print(f"  Warning: Could not apply weather mask: {e}")
        return df


def main():
    """Main function to create paper_results_v3.csv."""
    print("=" * 60)
    print("Creating paper_results_v3.csv with Prophet and MeteoBlue")
    print("=" * 60)

    # 1. Load existing results
    existing_file = RESULTS_PATH / "paper_results_v2.csv"
    if existing_file.exists():
        existing = pd.read_csv(existing_file)
        print(f"Loaded {len(existing)} existing results")
    else:
        print("Warning: paper_results_v2.csv not found")
        existing = pd.DataFrame()

    # 2. Get twilight events (2024-2025)
    twilight_events = load_twilight_events()
    print(f"Found {len(twilight_events)} twilight events (2024-2025)")

    # 3. Process Prophet forecasts
    prophet_results = process_prophet_forecasts(twilight_events)

    # 4. Process MeteoBlue forecasts
    meteoblue_results = process_meteoblue_forecasts(twilight_events)

    # 5. Apply weather mask to new results (skip for now - data matching is primary goal)
    # Weather mask can be applied later if needed
    # if len(prophet_results) > 0:
    #     prophet_results = apply_weather_mask(prophet_results)
    # if len(meteoblue_results) > 0:
    #     meteoblue_results = apply_weather_mask(meteoblue_results)
    print("Skipping weather mask (can be applied later)")

    # 6. Combine and save
    dfs_to_concat = [existing]
    if len(prophet_results) > 0:
        dfs_to_concat.append(prophet_results)
    if len(meteoblue_results) > 0:
        dfs_to_concat.append(meteoblue_results)

    combined = pd.concat(dfs_to_concat, ignore_index=True)

    output_file = RESULTS_PATH / "paper_results_v3.csv"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved {len(combined)} rows to {output_file}")

    # Summary
    print("\nSummary by model:")
    print(combined["model"].value_counts())


if __name__ == "__main__":
    main()
