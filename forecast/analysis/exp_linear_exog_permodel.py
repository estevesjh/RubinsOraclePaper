"""NBEATSx + Linear-as-futr-exog: one model per lead time.

For each target lead L (3h, 6h, 9h, 12h):
  - Train Ridge_L on pre-2025 grid: features at t-L_steps -> y(t)
  - Build static column linear_Lh(t) = Ridge_L(features at t - L_steps)
  - Train NBEATSx with horizon = L_steps, futr_exog includes linear_Lh
  - Evaluate on 2025 twilights at that single lead

Each NBEATSx model has horizon = lead_steps (not 26). Train/predict see the
same signal -> no proxy mismatch.

Usage:
    uv run --no-sync python forecast/exp_linear_exog_permodel.py
"""

import os, sys, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning","lightning.pytorch","lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
from config import (NBEATS_INPUT_SIZE, STEPS_PER_DAY,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS  # 700
TARGET_LEADS_H = [3.0, 6.0, 9.0, 12.0]
BASE_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
DET_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
LINEAR_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]


def build_linear_column(grid, lead_steps, col_name):
    """Train Ridge at lead_steps; build static column linear(t) = Ridge(feat@t-lead_steps)."""
    y = grid["y"].values
    ds_real = pd.to_datetime(grid["ds_real"])
    train_mask = ds_real < TEST_START_DATE
    feat_cols = [c for c in LINEAR_FEATS if c in grid.columns]
    n = len(grid)

    X_rows, y_rows = [], []
    for t in range(lead_steps, n):
        if not train_mask.iloc[t]:
            continue
        iss = t - lead_steps
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)) or np.isnan(y[t]):
            continue
        X_rows.append(feats)
        y_rows.append(y[t])

    X_train = np.array(X_rows); y_train = np.array(y_rows)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, y_train)

    # Populate column for all grid rows
    col = np.full(n, np.nan)
    for t in range(lead_steps, n):
        iss = t - lead_steps
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)):
            continue
        col[t] = ridge.predict(scaler.transform(feats.reshape(1,-1)))[0]
    grid[col_name] = col

    # Report
    valid = ~np.isnan(col) & ~np.isnan(y)
    test_m = valid & ~train_mask.values
    rmse = np.sqrt(np.mean((col[test_m] - y[test_m])**2))
    print(f"  {col_name}: Ridge trained on {len(X_train)} rows, test RMSE={rmse:.3f}")
    return grid


def train_and_eval(grid, tw_test, lead_h, lead_steps, futr_cols):
    """Train NBEATSx with horizon=lead_steps, evaluate at that single lead."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in BASE_HIST if c in grid.columns]
    futr = [c for c in futr_cols if c in grid.columns]
    allx = hist + futr

    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds","y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"

    model = NBEATSx(
        h=lead_steps, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend","seasonality","identity","exogenous"],
        mlp_units=4*[[WIDTH,WIDTH]], n_blocks=[1,1,1,1],
        early_stop_patience_steps=10, val_check_steps=50,
        accelerator="mps", devices=1,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train)*0.1))

    # Evaluate
    errs = []
    BATCH = 64
    requests = []
    for _, ev in tw_test.iterrows():
        tgt = int(ev["grid_idx"])
        iss = tgt - lead_steps
        if iss < NBEATS_INPUT_SIZE: continue
        requests.append((tgt, iss, ev["y_actual"]))

    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (tgt, iss, yact) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + lead_steps
            if hs < 0 or fe > len(grid): continue
            hd = grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd = grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid = f"r_{b0}_{tgt}"
            hd["unique_id"]=uid; fd["unique_id"]=uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, yact))
        if not all_h: continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h,ignore_index=True),
                           futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, yact) in meta:
            u = fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            # Last step = target
            if len(u)==0: continue
            pred = float(u[mc].iloc[-1])
            errs.append(yact - pred)

    e = np.array(errs)
    rmse = float(np.sqrt((e**2).mean()))
    p1 = float((np.abs(e)<1).mean()*100)
    return rmse, p1, len(e)


def main():
    print("="*72)
    print(f"NBEATSx + LINEAR PER-MODEL  |  width={WIDTH} steps={MAX_STEPS}")
    print("="*72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  test twilights: {len(tw_test)}\n")

    # Build linear columns for each lead
    for L in TARGET_LEADS_H:
        ls = run.lead_hours_to_steps(L)
        grid = build_linear_column(grid, ls, f"linear_{int(L)}h")

    # Run each model
    print(f"\n{'lead':>5} {'variant':>14} {'RMSE':>7} {'<1C':>5} {'N':>4}")
    print("-"*40)
    results = []
    for L in TARGET_LEADS_H:
        ls = run.lead_hours_to_steps(L)
        col = f"linear_{int(L)}h"

        # Base (no linear futr)
        rmse_b, p1_b, n_b = train_and_eval(grid, tw_test, L, ls, DET_FUTR)
        print(f"{L:>5} {'base':>14} {rmse_b:>7.3f} {p1_b:>4.0f}% {n_b:>4}")

        # +linear
        rmse_l, p1_l, n_l = train_and_eval(grid, tw_test, L, ls, DET_FUTR + [col])
        d = 100*(rmse_l - rmse_b)/rmse_b
        print(f"{L:>5} {'+linear':>14} {rmse_l:>7.3f} {p1_l:>4.0f}% {n_l:>4}  ({d:+.0f}%)")
        results.append((L, rmse_b, rmse_l, d))

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    for L, rb, rl, d in results:
        gate = "PASS" if d < -7 else "below gate"
        print(f"  {L:>4}h: base={rb:.3f} +linear={rl:.3f} ({d:+.1f}%) -> {gate}")


if __name__ == "__main__":
    main()
