"""Investigate short-lead performance: compare with persistence + noise floor."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(__file__))

from twilight.config import NBEATSxConfig
from twilight.features import FeatureBuilder
from twilight.utils import load_data

from config import (
    DATA_PATH, RESULTS_PATH, TEST_START_DATE,
    SOLAR_GRID_STEP, HALFDAY_LAG_STEPS, STEPS_PER_DAY,
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, LEAD_TIMES_HOURS,
)


def load_grid():
    """Load solar grid (same as run.py)."""
    df = load_data(str(DATA_PATH))
    cfg = NBEATSxConfig(
        solar_grid=True,
        solar_grid_step=SOLAR_GRID_STEP,
        solar_grid_fillna=True,
        sun_alt_midpoint=-15.0,
        smooth_window_hours=1.0,
        input_size=NBEATS_INPUT_SIZE,
        horizon=NBEATS_HORIZON,
    )
    fb = FeatureBuilder(cfg)
    grid = fb.transform(df)

    # Warmup
    n_warmup = NBEATS_INPUT_SIZE + HALFDAY_LAG_STEPS
    grid = grid.iloc[n_warmup:].reset_index(drop=True)
    return grid


def find_twilight_targets(grid):
    mask = grid["twilight_event_sunset"].values.astype(bool)
    idxs = np.where(mask)[0]
    events = []
    for i in idxs:
        events.append({
            "grid_idx": i,
            "ds_real": grid["ds_real"].iloc[i],
            "y_actual": grid["y"].iloc[i],
        })
    return pd.DataFrame(events)


def lead_hours_to_steps(lead_hours):
    return round(lead_hours * STEPS_PER_DAY / 24.0)


def main():
    print("=" * 70)
    print("INVESTIGATION: Short-Lead Performance + Baselines")
    print("=" * 70)

    print("\nLoading solar grid...")
    grid = load_grid()
    y_arr = grid["y"].values

    tw_events = find_twilight_targets(grid)
    tw_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE].copy()
    print(f"  Test twilights: {len(tw_test)}")

    # ── 1. Data noise floor ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. DATA NOISE FLOOR")
    print("=" * 70)

    # Step-to-step variability (30-min differences)
    step_diff = np.diff(y_arr)
    step_diff = step_diff[~np.isnan(step_diff)]
    print(f"  Step-to-step (30min) std: {step_diff.std():.3f} °C")
    print(f"  Step-to-step (30min) MAE: {np.abs(step_diff).mean():.3f} °C")

    # D target variability at twilight points
    D_tw = []
    for _, ev in tw_test.iterrows():
        tidx = ev["grid_idx"]
        anchor_idx = tidx - HALFDAY_LAG_STEPS
        if anchor_idx >= 0:
            D_tw.append(y_arr[tidx] - y_arr[anchor_idx])
    D_tw = np.array(D_tw)
    print(f"  D at twilight (target): mean={D_tw.mean():.3f}, std={D_tw.std():.3f} °C")

    # ── 2. Persistence baseline ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. PERSISTENCE vs NBEATSx-Diff")
    print("=" * 70)

    # Load NBEATSx results
    nbeats_df = pd.read_csv(RESULTS_PATH / "paper_results_diff.csv")
    nbeats_df = nbeats_df[nbeats_df["model"] == "NBEATSx-Diff"]

    print(f"\n{'Lead(h)':>8} {'Persist':>9} {'NBEATSx':>9} {'Ratio':>7} {'Noise':>7}")
    print("-" * 50)

    for lead_h in LEAD_TIMES_HOURS:
        offset = lead_hours_to_steps(lead_h)

        # Persistence: T_pred = T at issue time
        errors_pers = []
        for _, ev in tw_test.iterrows():
            tidx = ev["grid_idx"]
            issue_idx = tidx - offset
            if issue_idx < 0 or issue_idx >= len(y_arr):
                continue
            errors_pers.append(ev["y_actual"] - y_arr[issue_idx])
        errors_pers = np.array(errors_pers)
        rmse_pers = np.sqrt((errors_pers ** 2).mean()) if len(errors_pers) > 0 else np.nan

        # NBEATSx from CSV
        sub = nbeats_df[nbeats_df["lead_time_hours"] == lead_h]
        rmse_nbeats = np.sqrt((sub["error"] ** 2).mean()) if len(sub) > 0 else np.nan

        # Noise floor: std of step_diff * sqrt(offset steps)
        # Random walk noise accumulation
        noise_floor = step_diff.std() * np.sqrt(offset)

        ratio = rmse_nbeats / rmse_pers if rmse_pers > 0 else np.nan

        print(f"{lead_h:8.1f} {rmse_pers:9.3f} {rmse_nbeats:9.3f} {ratio:7.2f} {noise_floor:7.3f}")

    # ── 3. R² of D prediction ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. R² OF D PREDICTION (how much of D variance explained)")
    print("=" * 70)

    # For each lead time, compute D_actual and D_pred = forecast - anchor
    print(f"\n{'Lead(h)':>8} {'R²':>8} {'D_RMSE':>8} {'D_std':>8}")
    print("-" * 40)

    for lead_h in [0.5, 1.0, 2.0, 3.0, 6.0, 9.0, 11.5]:
        sub = nbeats_df[nbeats_df["lead_time_hours"] == lead_h].copy()
        if len(sub) == 0:
            continue

        # D_pred = forecast_temp - anchor
        # D_actual = actual_temp - anchor
        # But we don't have anchor in CSV... reconstruct from error
        # error = actual - forecast => D_actual - D_pred = error
        # We need D_actual at twilight = y_actual - y(anchor)
        D_actual_list = []
        D_pred_list = []
        for _, row in sub.iterrows():
            tw_time = pd.Timestamp(row["twilight_time"])
            mask = tw_test["ds_real"] == tw_time
            if not mask.any():
                continue
            ev = tw_test[mask].iloc[0]
            tidx = ev["grid_idx"]
            anchor_idx = tidx - HALFDAY_LAG_STEPS
            if anchor_idx < 0:
                continue
            anchor = y_arr[anchor_idx]
            D_actual_list.append(row["actual_temp"] - anchor)
            D_pred_list.append(row["forecast_temp"] - anchor)

        D_actual = np.array(D_actual_list)
        D_pred = np.array(D_pred_list)
        if len(D_actual) < 10:
            continue

        ss_res = ((D_actual - D_pred) ** 2).sum()
        ss_tot = ((D_actual - D_actual.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        d_rmse = np.sqrt((( D_actual - D_pred) ** 2).mean())

        print(f"{lead_h:8.1f} {r2:8.3f} {d_rmse:8.3f} {D_actual.std():8.3f}")

    # ── 4. Outlier analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. OUTLIER ANALYSIS (|error| > 2°C at 3h lead)")
    print("=" * 70)

    sub_3h = nbeats_df[nbeats_df["lead_time_hours"] == 3.0].copy()
    outliers = sub_3h[sub_3h["error"].abs() > 2.0].sort_values("error", key=abs, ascending=False)

    print(f"\n  Outliers: {len(outliers)} / {len(sub_3h)} ({100*len(outliers)/len(sub_3h):.1f}%)")
    print(f"\n{'Date':>22} {'Error':>7} {'Actual':>7} {'Forecast':>9}")
    print("-" * 50)
    for _, row in outliers.head(15).iterrows():
        print(f"{row['twilight_time']:>22} {row['error']:+7.2f} {row['actual_temp']:7.2f} {row['forecast_temp']:9.2f}")

    # Check if outliers are in filled/imputed periods
    print("\n  Checking if outliers fall in imputed periods...")
    raw_df = pd.read_csv(DATA_PATH, comment="#")
    raw_df["ds"] = pd.to_datetime(raw_df["timestamp"], utc=True).dt.tz_localize(None)

    filled_dates = set()
    for _, row in outliers.iterrows():
        tw_time = pd.Timestamp(row["twilight_time"])
        # Check if any row within 6h of twilight is filled
        mask = (raw_df["ds"] >= tw_time - pd.Timedelta(hours=6)) & \
               (raw_df["ds"] <= tw_time + pd.Timedelta(hours=1))
        nearby = raw_df[mask]
        if len(nearby) > 0 and nearby["filled"].any():
            filled_dates.add(str(tw_time.date()))

    if filled_dates:
        print(f"  Outliers in imputed periods: {sorted(filled_dates)}")
    else:
        print("  None of the outliers fall in imputed periods.")

    # ── 5. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)
    print("""
  Key finding: The differenced model uses a 12h-old anchor for ALL lead times.
  At short leads (0.5-1h), we know the current temperature but reconstruct from
  T(12h ago) + D_pred. Persistence (T at issue time) beats us at short leads.

  The model's value is at LONGER leads (3h+) where persistence degrades rapidly
  but the differenced approach stays accurate.

  Possible improvements:
  1. Blend with persistence at short leads: T_final = α*T_persist + (1-α)*T_diff
  2. Use a shorter lag for short-lead predictions (multi-horizon approach)
  3. Accept the tradeoff: the 3h lead (the operational target) is what matters
""")


if __name__ == "__main__":
    main()
