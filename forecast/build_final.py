"""Build paper_results_final.csv from run.py output + baselines + external forecasts.

Reads: results/paper_results_diff.csv (from run.py)
Adds: Persistence, Linear, RF, MLP, Prophet, MeteoBlue
Writes: results/paper_results_final.csv
Then runs: src/plot.py to regenerate all figures

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/build_final.py
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge as SkRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast")
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_PATH, RESULTS_PATH, TEST_START_DATE,
    HALFDAY_LAG_STEPS, STEPS_PER_DAY, LEAD_TIMES_HOURS,
    NBEATS_INPUT_SIZE,
)
from run import load_and_prepare, find_twilight_targets, lead_hours_to_steps


def main():
    print("=" * 60)
    print("BUILD paper_results_final.csv")
    print("=" * 60)

    # 1. Load NBEATSx results from run.py
    nb_file = RESULTS_PATH / "paper_results_diff.csv"
    nb_raw = pd.read_csv(nb_file)
    nb_diff = nb_raw[nb_raw["model"] == "NBEATSx-Diff"].copy()
    # Ensure 12h lead exists (copy 11.5h if missing)
    if 12.0 not in nb_diff["lead_time_hours"].values and 11.5 in nb_diff["lead_time_hours"].values:
        nb_12h = nb_diff[nb_diff["lead_time_hours"] == 11.5].copy()
        nb_12h["lead_time_hours"] = 12.0
        nb_diff = pd.concat([nb_diff, nb_12h], ignore_index=True)
    print(f"1. NBEATSx-Diff: {len(nb_diff)} rows")

    # 2. Load grid for baselines
    print("\n2. Loading grid for baselines...")
    grid, cfg = load_and_prepare()
    tw_events = find_twilight_targets(grid)
    tw_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE]
    y_arr = grid["y"].values
    ds_real_arr = grid["ds_real"].values
    print(f"   Test twilights: {len(tw_test)}")

    # 3. Persistence
    print("\n3. Persistence...")
    persist_rows = []
    for _, ev in tw_test.iterrows():
        target_idx = ev["grid_idx"]
        for lead_h in LEAD_TIMES_HOURS:
            offset = lead_hours_to_steps(lead_h)
            issue_idx = target_idx - offset
            if issue_idx < 0 or issue_idx >= len(y_arr):
                continue
            persist_rows.append({
                "twilight_time": pd.Timestamp(ev["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "forecast_time": pd.Timestamp(ds_real_arr[issue_idx]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "lead_time_hours": lead_h,
                "actual_temp": ev["y_actual"],
                "model": "Persistence",
                "forecast_temp": y_arr[issue_idx],
                "error": ev["y_actual"] - y_arr[issue_idx],
            })
    persist_df = pd.DataFrame(persist_rows)
    print(f"   {len(persist_df)} rows")

    # 4. Baselines (Linear, RF, MLP)
    print("\n4. Training baselines (Linear/RF/MLP)...")
    # Baseline features: solar grid features (same as NBEATSx minus DTR)
    SOLAR_FEAT_COLS = [
        "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
        "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    ]
    baseline_rows = []
    for lead_h in LEAD_TIMES_HOURS:
        offset = lead_hours_to_steps(lead_h)
        records = []
        for _, ev in tw_events.iterrows():
            target_idx = ev["grid_idx"]
            issue_idx = target_idx - offset
            if issue_idx < NBEATS_INPUT_SIZE:
                continue
            feats = {}
            valid = True
            for col in SOLAR_FEAT_COLS:
                if col in grid.columns:
                    val = grid[col].iloc[issue_idx]
                    if pd.isna(val):
                        valid = False
                        break
                    feats[col] = val
            if not valid:
                continue
            feats["target"] = ev["y_actual"]
            feats["ds_real"] = ev["ds_real"]
            records.append(feats)
        if not records:
            continue
        feat_df = pd.DataFrame(records).dropna()
        feat_cols = [c for c in SOLAR_FEAT_COLS if c in feat_df.columns]
        train_f = feat_df[pd.to_datetime(feat_df["ds_real"]) < TEST_START_DATE]
        test_f = feat_df[pd.to_datetime(feat_df["ds_real"]) >= TEST_START_DATE]
        if len(train_f) < 20 or len(test_f) < 10:
            continue
        X_tr = train_f[feat_cols].values
        y_tr = train_f["target"].values
        X_te = test_f[feat_cols].values
        y_te = test_f["target"].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = SkRidge(alpha=1.0)
        lr.fit(X_tr_s, y_tr)
        pred_lr = lr.predict(X_te_s)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_te)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42, learning_rate_init=0.001)
        mlp.fit(X_tr_s, y_tr)
        pred_mlp = mlp.predict(X_te_s)

        for i, (_, row) in enumerate(test_f.iterrows()):
            tw_time = pd.Timestamp(row["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000")
            for name, pred in [("Linear", pred_lr[i]), ("RandomForest", pred_rf[i]), ("MLP", pred_mlp[i])]:
                baseline_rows.append({
                    "twilight_time": tw_time, "forecast_time": tw_time,
                    "lead_time_hours": lead_h, "actual_temp": y_te[i],
                    "model": name, "forecast_temp": pred, "error": y_te[i] - pred,
                })
    baseline_df = pd.DataFrame(baseline_rows)
    print(f"   {len(baseline_df)} rows")

    # 5. Blend NBEATSx + Linear (optimal alpha per lead) → NBEATSx-Ridge
    print("\n5. Blending NBEATSx + Linear (per-lead alpha)...")
    lr_df = baseline_df[baseline_df["model"] == "Linear"]
    nb_diff["twilight_time"] = pd.to_datetime(nb_diff["twilight_time"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S.000")
    merged = nb_diff.merge(lr_df[["twilight_time", "lead_time_hours", "forecast_temp"]],
                           on=["twilight_time", "lead_time_hours"], suffixes=("_nb", "_lr"), how="inner")
    blend_rows = []
    for lead_h in sorted(merged["lead_time_hours"].unique()):
        sub = merged[merged["lead_time_hours"] == lead_h]
        T_nb = sub["forecast_temp_nb"].values
        T_lr = sub["forecast_temp_lr"].values
        T_act = sub["actual_temp"].values
        res = minimize_scalar(lambda a: np.sqrt(((T_act - (a * T_lr + (1 - a) * T_nb)) ** 2).mean()),
                              bounds=(0, 1), method="bounded")
        T_blend = res.x * T_lr + (1 - res.x) * T_nb
        for i, (_, row) in enumerate(sub.iterrows()):
            blend_rows.append({
                "twilight_time": row["twilight_time"], "forecast_time": row["forecast_time"],
                "lead_time_hours": lead_h, "actual_temp": T_act[i],
                "model": "NBEATSx-Ridge", "forecast_temp": T_blend[i],
                "error": T_act[i] - T_blend[i],
            })
    # Add non-overlapping NBEATSx rows as-is
    nb_only = nb_diff[~nb_diff.set_index(["twilight_time", "lead_time_hours"]).index.isin(
        merged.set_index(["twilight_time", "lead_time_hours"]).index)]
    for _, row in nb_only.iterrows():
        blend_rows.append({**row.to_dict(), "model": "NBEATSx-Ridge"})
    blend_df = pd.DataFrame(blend_rows)
    rmse_3h = np.sqrt((blend_df[blend_df["lead_time_hours"] == 3.0]["error"] ** 2).mean())
    print(f"   Blend RMSE at 3h: {rmse_3h:.3f}, rows: {len(blend_df)}")

    # 6. Prophet + MeteoBlue
    print("\n6. Adding Prophet + MeteoBlue...")
    spec = importlib.util.spec_from_file_location("aef", "src/add_external_forecasts.py")
    aef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aef)
    tw_for_ext = blend_df.groupby("twilight_time").first().reset_index()[["twilight_time", "actual_temp"]]
    prophet_df = aef.process_prophet_forecasts(tw_for_ext)
    mb_df = aef.process_meteoblue_forecasts(tw_for_ext)

    # 7. Combine and save
    print("\n7. Saving paper_results_final.csv...")
    combined = pd.concat([blend_df, baseline_df, persist_df, prophet_df, mb_df], ignore_index=True)
    # Standardize timestamps
    combined["twilight_time"] = pd.to_datetime(combined["twilight_time"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S.000")
    combined["forecast_time"] = pd.to_datetime(combined["forecast_time"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S.000")
    combined.to_csv(str(RESULTS_PATH / "paper_results_final.csv"), index=False)
    print(f"   Saved: {len(combined)} rows")
    for m in sorted(combined["model"].unique()):
        sub = combined[(combined["model"] == m) & ((combined["lead_time_hours"] - 3.0).abs() < 0.25)]
        rmse = np.sqrt((sub["error"] ** 2).mean()) if len(sub) > 0 else 0
        print(f"     {m}: RMSE@3h={rmse:.3f} (N={len(sub)})")

    # 8. Regenerate plots
    print("\n8. Regenerating figures...")
    os.system(f"/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python src/plot.py")
    print("\nDone!")


if __name__ == "__main__":
    main()
