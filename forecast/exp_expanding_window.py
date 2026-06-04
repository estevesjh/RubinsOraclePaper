"""Expanding-window evaluator: retrain monthly, predict the next month.

For each month M in 2025:
  - Train on all data prior to M (pre-2025 + earlier 2025 months)
  - Predict twilights in month M
  - Report RMSE at 3h

This mimics operational deployment with monthly retraining.
No leakage: each twilight is predicted by a model that never saw it.

Usage:
    uv run --no-sync python forecast/exp_expanding_window.py
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

sys.path.insert(0, os.path.dirname(__file__))
import run
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS  # 700
LEAD_H = 3.0
HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]


def train_model(grid, train_end):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in HIST if c in grid.columns]
    futr = [c for c in FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < train_end
    nf_train = grid.loc[tr, ["ds","y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    if len(nf_train) < 1000:
        return None, hist, futr

    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
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
    return nf, hist, futr


def eval_month(nf, grid, tw_month, hist, futr):
    allx = hist + futr
    lead_steps = run.lead_hours_to_steps(LEAD_H)
    errs = []
    BATCH = 64
    requests = [(int(ev["grid_idx"]), ev["y_actual"]) for _, ev in tw_month.iterrows()]
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (tgt, yact) in batch:
            iss = tgt - lead_steps
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid) or iss < NBEATS_INPUT_SIZE:
                continue
            hd = grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd = grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid = f"e_{b0}_{tgt}"
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
            ps = tgt - iss
            if len(u)==0 or ps >= len(u): continue
            errs.append(yact - float(u[mc].iloc[ps]))
    if not errs:
        return np.nan, 0.0, 0
    e = np.array(errs)
    return float(np.sqrt((e**2).mean())), float((np.abs(e)<1).mean()*100), len(e)


def main():
    print("="*72)
    print(f"EXPANDING WINDOW  |  monthly retrain  |  width={WIDTH} steps={MAX_STEPS}")
    print("="*72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    print(f"  total twilights: {len(tw)}\n")

    months_2025 = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    print(f"{'month':>8} {'train_end':>12} {'n_train':>8} {'RMSE@3h':>8} {'<1C':>5} {'N':>4}")
    print("-"*50)

    all_errs = []
    for m_start in months_2025:
        m_end = m_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        train_end = m_start  # train on everything BEFORE this month

        # Twilights in this month
        tw_month = tw[(tw["ds_real"] >= m_start) & (tw["ds_real"] < m_end)]
        if len(tw_month) == 0:
            continue

        # Count training rows
        tr_count = int((pd.to_datetime(grid["ds_real"]) < train_end).sum())

        nf, hist, futr = train_model(grid, train_end)
        if nf is None:
            print(f"{m_start.strftime('%Y-%m'):>8} {str(train_end.date()):>12} {tr_count:>8} {'skip':>8}")
            continue

        rmse, p1, n = eval_month(nf, grid, tw_month, hist, futr)
        print(f"{m_start.strftime('%Y-%m'):>8} {str(train_end.date()):>12} {tr_count:>8} "
              f"{rmse:>8.3f} {p1:>4.0f}% {n:>4}")

    print("\nDone.")


if __name__ == "__main__":
    main()
