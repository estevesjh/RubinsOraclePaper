"""Evaluate cached NBEATSx-Diff model with predictions re-gridded to 5-min real time.

1. Run model on solar grid → get D_pred for full horizon (24 steps)
2. Interpolate D_pred onto a 5-min real-time grid using ds_real
3. Reconstruct T_pred = D_pred_interp + y_anchor_interp at exact twilight time
4. Compute lead = (ds_real_twilight - ds_real_issue) in real hours

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/eval_real_hours.py
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

sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path

from config import (
    DATA_PATH, RESULTS_PATH, MODEL_CACHE_PATH,
    TEST_START_DATE, HALFDAY_LAG_STEPS, SOLAR_GRID_FREQ,
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, STEPS_PER_DAY, SOLAR_GRID_STEP,
)
from run import (
    load_and_prepare, find_twilight_targets, get_hist_futr_exog,
    suppress_stdout, train_nbeats_diff,
)


LEAD_TIMES_HOURS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
REGRID_FREQ_MIN = 5  # re-grid to 5-min real time


def main():
    print("=" * 70)
    print("EVALUATION: Re-gridded to 5-min real time")
    print("=" * 70)

    grid, cfg = load_and_prepare()

    tw_events = find_twilight_targets(grid)
    tw_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE].copy()
    print(f"  Test twilights: {len(tw_test)}")

    print("\nLoading cached model...")
    model = train_nbeats_diff(grid, cfg, target_col="D", cache_name="NBEATSx_diff_v2", use_cache=True)

    hist_exog, futr_exog = get_hist_futr_exog(cfg)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    y_arr = grid["y"].values
    ds_real_arr = grid["ds_real"].values  # datetime64
    ds_real_sec = pd.to_datetime(ds_real_arr).astype("int64") / 1e9  # seconds for interp

    # For each twilight event, issue forecast from multiple lead times
    # and interpolate the full horizon onto 5-min real time
    print("\nRunning forecasts and re-gridding...")

    results = []
    BATCH_SIZE = 64

    # Build requests: one per (twilight, lead_time)
    # But actually we can be smarter: for each twilight, issue from MULTIPLE origins
    # and get the full horizon each time. Then interpolate.
    # Simpler: for each twilight, issue from each lead and get full horizon,
    # then interpolate D_pred onto 5-min grid between issue and target.

    for ev_i, (_, ev) in enumerate(tw_test.iterrows()):
        if ev_i % 50 == 0:
            print(f"  Event {ev_i}/{len(tw_test)}...")

        target_idx = ev["grid_idx"]
        tw_real = pd.Timestamp(ev["ds_real"])
        tw_real_sec = tw_real.timestamp()

        for lead_h in LEAD_TIMES_HOURS:
            # Find issue index by real time
            issue_time_sec = tw_real_sec - lead_h * 3600
            # searchsorted on ds_real_sec
            issue_idx = int(np.searchsorted(ds_real_sec, issue_time_sec))
            issue_idx = min(max(issue_idx, 0), len(grid) - 1)

            if issue_idx < NBEATS_INPUT_SIZE:
                continue
            if (target_idx - issue_idx) >= NBEATS_HORIZON:
                continue
            if issue_idx >= target_idx:
                continue

            # Get history + future for this request
            hist_start = issue_idx - NBEATS_INPUT_SIZE
            futr_end = issue_idx + NBEATS_HORIZON
            if futr_end > len(grid):
                continue

            uid = f"ev{ev_i}_l{lead_h}"
            hist_df = grid.iloc[hist_start:issue_idx][["ds", "D"] + all_exog].copy()
            if hist_df[all_exog].isna().any().any():
                continue
            hist_df["y"] = hist_df["D"]
            hist_df["unique_id"] = uid

            futr_df = grid.iloc[issue_idx:futr_end][["ds"] + futr_exog].copy()
            futr_df["unique_id"] = uid

            # Predict
            with suppress_stdout():
                fc = model.predict(hist_df, futr_df=futr_df)
            fc = fc.reset_index()
            model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]

            # D_pred on solar grid (NBEATS_HORIZON steps)
            D_pred_solar = fc[model_col].values
            # ds_real for the forecast horizon
            ds_real_horizon = ds_real_sec[issue_idx:futr_end]

            if len(D_pred_solar) != len(ds_real_horizon):
                continue

            # Also get y (observed) on the same horizon for anchor reconstruction
            # Anchor: y at (each forecast point - HALFDAY_LAG_STEPS)
            # For the target specifically:
            anchor_idx = target_idx - HALFDAY_LAG_STEPS
            if anchor_idx < 0 or anchor_idx >= len(y_arr):
                continue
            anchor_temp = y_arr[anchor_idx]
            if np.isnan(anchor_temp):
                continue

            # Re-grid D_pred onto 5-min real time between issue and target
            regrid_step_sec = REGRID_FREQ_MIN * 60
            issue_real_sec = ds_real_sec[issue_idx]
            regrid_times = np.arange(issue_real_sec, tw_real_sec + regrid_step_sec, regrid_step_sec)

            # Interpolate D_pred
            D_pred_interp = np.interp(regrid_times, ds_real_horizon, D_pred_solar)

            # Get D_pred at exact twilight time
            D_pred_at_tw = np.interp(tw_real_sec, ds_real_horizon, D_pred_solar)

            # Reconstruct
            T_pred = D_pred_at_tw + anchor_temp

            # Actual lead in real hours
            actual_lead = (tw_real_sec - issue_real_sec) / 3600.0

            # Persistence = T at issue time
            T_persist = y_arr[issue_idx]

            results.append({
                "twilight_time": tw_real,
                "forecast_time": pd.Timestamp(ds_real_arr[issue_idx]),
                "lead_time_hours": actual_lead,
                "lead_nominal": lead_h,
                "actual_temp": ev["y_actual"],
                "forecast_temp": T_pred,
                "persist_temp": T_persist,
                "error": ev["y_actual"] - T_pred,
                "error_persist": ev["y_actual"] - T_persist,
                "anchor_temp": anchor_temp,
                "D_pred": D_pred_at_tw,
            })

    results_df = pd.DataFrame(results)
    print(f"\n  Total results: {len(results_df)}")

    # RMSE by nominal lead time
    print("\n" + "=" * 70)
    print("RMSE BY REAL LEAD TIME (5-min interpolated)")
    print("=" * 70)
    print(f"{'Lead(h)':>8} {'Actual(h)':>10} {'NBEATSx':>8} {'Persist':>8} {'<1C':>6} {'N':>5}")
    print("-" * 55)

    for lead_h in LEAD_TIMES_HOURS:
        sub = results_df[results_df["lead_nominal"] == lead_h]
        if len(sub) > 0:
            rmse = np.sqrt((sub["error"] ** 2).mean())
            rmse_p = np.sqrt((sub["error_persist"] ** 2).mean())
            mean_lead = sub["lead_time_hours"].mean()
            lt1 = (sub["error"].abs() < 1.0).mean() * 100
            print(f"{lead_h:8.1f} {mean_lead:10.2f} {rmse:8.3f} {rmse_p:8.3f} {lt1:5.1f}% {len(sub):5d}")

    # Save
    output_file = RESULTS_PATH / "paper_results_real_hours.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
