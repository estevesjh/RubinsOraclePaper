"""Sample-size sweep: NBEATSx+Ridge, expanding window oldest->newest.

For each window size N (10 log-spaced, pre-2025 only):
  1. Train NBEATSx on the first N rows of pre-2025 grid
  2. Get in-sample preds on pre-2025 twilights within that window
  3. Train Ridge on those preds
  4. Get NBEATSx test preds on 2025 twilights
  5. Apply Ridge -> final NB+Ridge RMSE

Usage:
    uv run --no-sync python forecast/sweep_samplesize.py
"""
import os, sys, warnings, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning","lightning.pytorch","lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import run
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON, SOLAR_GRID_FREQ,
                    TEST_START_DATE, NBEATS_MAX_STEPS, RESULTS_PATH, FIGURES_PATH)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
N_SIZES = 10
SIZE_MIN = 3200  # ~Jan 2024 start
KEY_LEADS = [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
]


def get_preds_batch(nf, grid, tw_set, hist, futr, leads):
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in leads:
            ls = run.lead_hours_to_steps(L)
            iss = tgt - ls
            if iss < NBEATS_INPUT_SIZE or (tgt-iss) >= NBEATS_HORIZON: continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact, ds_r) in batch:
            hs=iss-NBEATS_INPUT_SIZE; fe=iss+NBEATS_HORIZON
            if hs<0 or fe>len(grid): continue
            hd=grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd=grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid=f"p_{b0}_{ev_i}_{L}"
            hd["unique_id"]=uid; fd["unique_id"]=uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, L, yact, ev_i))
        if not all_h: continue
        with run.suppress_stdout():
            fc=nf.predict(pd.concat(all_h,ignore_index=True),
                         futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc=[c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, L, yact, ev_i) in meta:
            u=fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            ps=tgt-iss
            if len(u)==0 or ps>=len(u): continue
            preds[(ev_i, L)] = {"T_nb": float(u[mc].iloc[ps]), "y_actual": yact, "iss": iss}
    return preds


def ridge_eval(grid, train_preds, test_preds, leads):
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    results = {}
    for L in leads:
        X_tr, y_tr, X_te, y_te = [], [], [], []
        for (ei, lead), p in train_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            X_tr.append(np.concatenate([feats, [p["T_nb"]]])); y_tr.append(p["y_actual"])
        for (ei, lead), p in test_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            X_te.append(np.concatenate([feats, [p["T_nb"]]])); y_te.append(p["y_actual"])
        if len(X_tr) < 20 or len(X_te) < 20:
            results[L] = np.nan; continue
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)
        scaler = StandardScaler()
        ridge = SkRidge(alpha=1.0)
        ridge.fit(scaler.fit_transform(X_tr), y_tr)
        results[L] = float(np.sqrt(((y_te - ridge.predict(scaler.transform(X_te)))**2).mean()))
    return results


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    print("="*72)
    print(f"SAMPLE-SIZE SWEEP (NBEATSx+Ridge)  |  width={WIDTH} steps={MAX_STEPS}")
    print("="*72)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)

    # Full pre-2025 training frame
    ds_real = pd.to_datetime(grid["ds_real"])
    pre_mask = ds_real < TEST_START_DATE
    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    allx = hist + futr
    nf_full = grid.loc[pre_mask, ["ds","y","ds_real"]+allx].dropna()
    n_full = len(nf_full)
    print(f"  full pre-2025: {n_full} rows, test: {len(tw_test)} twilights")

    sizes = np.unique(np.round(np.linspace(SIZE_MIN, n_full, N_SIZES)).astype(int))
    sizes[-1] = n_full

    # Pre-2025 twilights (for Ridge training)
    tw_train_all = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)

    rows = []
    for n in sizes:
        window = nf_full.iloc[:n]
        window_end = pd.to_datetime(window["ds_real"].iloc[-1])
        # Train NBEATSx on this window
        nf_train = window[["ds","y"]+allx].copy()
        nf_train["unique_id"] = "temp"
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

        # In-sample preds on pre-2025 twilights (for Ridge training)
        # Only twilights whose grid_idx falls within this window
        tw_in_window = tw_train_all[tw_train_all["grid_idx"] < (nf_full.index[0] + n)].copy().reset_index(drop=True)
        train_preds = get_preds_batch(nf, grid, tw_in_window, hist, futr, KEY_LEADS)
        # Test preds on 2025
        test_preds = get_preds_batch(nf, grid, tw_test, hist, futr, KEY_LEADS)
        # Ridge
        ridge_rmses = ridge_eval(grid, train_preds, test_preds, KEY_LEADS)

        row = {"n_train": int(n), "window_end": window_end.isoformat()}
        for L in KEY_LEADS:
            row[f"rmse_{L}h"] = ridge_rmses.get(L, np.nan)
        rows.append(row)
        print(f"  n={n:>6} end={window_end.strftime('%Y-%m')} "
              f"R@3h={row.get('rmse_3.0h', np.nan):.3f} "
              f"R@12h={row.get('rmse_12.0h', np.nan):.3f}")

    df = pd.DataFrame(rows)
    out = RESULTS_PATH / "sweep_samplesize.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
