"""NBEATSx + Ridge: use NBEATSx residual as a feature for Ridge.

Approach (fully causal):
1. Train NBEATSx (direct-T, width=16, 700 steps) on pre-2025.
2. For each twilight event at lead L, get NBEATSx prediction T_nb.
3. Compute residual r = T_nb - T_actual (on training twilights only).
4. Train Ridge on [Linear features at issue time, T_nb, r_lag1, r_lag2]
   to predict T_actual. Only uses training-period twilights.
5. At test time: get T_nb from NBEATSx, compute Ridge prediction.

The key insight: NBEATSx's residual (its approximation error) carries
information that the Linear model's features can correct — e.g. systematic
biases in specific temperature regimes.

CAUSAL: Ridge only sees T_nb (the NBEATSx forecast, issued before target)
and features at issue time. No future information.

Usage:
    uv run --no-sync python forecast/exp_nbeats_ridge.py
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

sys.path.insert(0, os.path.dirname(__file__))
import run
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS  # 700
LEADS = [3.0, 6.0, 9.0, 12.0]
HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
LINEAR_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]


def train_nbeats(grid):
    """Train NBEATSx (direct-T) on pre-2025."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in HIST if c in grid.columns]
    futr = [c for c in FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds","y"]+allx].dropna().copy()
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
    return nf, hist, futr


def get_nbeats_predictions(nf, grid, tw_events, hist, futr):
    """Get NBEATSx predictions for a set of twilight events at all leads.
    Returns dict: (tw_idx, lead_h) -> T_nb prediction."""
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_events.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in LEADS:
            ls = run.lead_hours_to_steps(L)
            iss = tgt - ls
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))

    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid): continue
            hd = grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd = grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid = f"p_{b0}_{ev_i}_{L}"
            hd["unique_id"]=uid; fd["unique_id"]=uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, L, yact, ev_i))
        if not all_h: continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h,ignore_index=True),
                           futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, L, yact, ev_i) in meta:
            u = fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u)==0 or ps >= len(u): continue
            preds[(ev_i, L)] = {
                "T_nb": float(u[mc].iloc[ps]),
                "y_actual": yact,
                "issue_idx": iss,
                "target_idx": tgt,
            }
    return preds


def main():
    print("="*72)
    print(f"NBEATSx + RIDGE  |  width={WIDTH} steps={MAX_STEPS}")
    print("="*72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}\n")

    feat_cols = [c for c in LINEAR_FEATS if c in grid.columns]

    # Step 1: Train NBEATSx
    print("Training NBEATSx...")
    nf, hist, futr = train_nbeats(grid)

    # Step 2: Get in-sample predictions (training twilights) for Ridge training
    print("Getting NBEATSx predictions on TRAINING twilights (in-sample)...")
    train_preds = get_nbeats_predictions(nf, grid, tw_train, hist, futr)
    print(f"  got {len(train_preds)} predictions")

    # Step 3: Get test predictions
    print("Getting NBEATSx predictions on TEST twilights...")
    test_preds = get_nbeats_predictions(nf, grid, tw_test, hist, futr)
    print(f"  got {len(test_preds)} predictions")

    # Step 4: Per-lead Ridge
    print(f"\n{'lead':>5} {'NBEATSx':>8} {'Linear':>8} {'NB+Ridge':>9} {'Δ vs Lin':>9}")
    print("-"*45)

    for L in LEADS:
        ls = run.lead_hours_to_steps(L)

        # Build Ridge training data
        X_train_rows, y_train_rows = [], []
        for ev_i in range(len(tw_train)):
            key = (ev_i, L)
            if key not in train_preds:
                continue
            p = train_preds[key]
            iss = p["issue_idx"]
            # Linear features at issue time
            feats = grid.iloc[iss][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            # NBEATSx prediction
            T_nb = p["T_nb"]
            row = np.concatenate([feats, [T_nb]])
            X_train_rows.append(row)
            y_train_rows.append(p["y_actual"])

        if len(X_train_rows) < 50:
            print(f"{L:>5}h  insufficient training data ({len(X_train_rows)})")
            continue

        X_tr = np.array(X_train_rows)
        y_tr = np.array(y_train_rows)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)

        # Build Ridge test data
        nb_errs, lin_errs, ridge_errs = [], [], []
        for ev_i in range(len(tw_test)):
            key = (ev_i, L)
            if key not in test_preds:
                continue
            p = test_preds[key]
            iss = p["issue_idx"]
            feats = grid.iloc[iss][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            T_nb = p["T_nb"]
            yact = p["y_actual"]

            # Ridge prediction
            row = np.concatenate([feats, [T_nb]])
            T_ridge = ridge.predict(scaler.transform(row.reshape(1,-1)))[0]

            # Linear-only prediction (same Ridge but without T_nb)
            row_lin = np.concatenate([feats, [0.0]])  # zero-out T_nb
            # Actually train a separate linear-only model for fair comparison
            nb_errs.append(yact - T_nb)
            ridge_errs.append(yact - T_ridge)

        # Also train pure Linear (no T_nb) for comparison
        X_lin_tr = np.array([r[:-1] for r in X_train_rows])  # drop T_nb col
        scaler_lin = StandardScaler()
        X_lin_s = scaler_lin.fit_transform(X_lin_tr)
        ridge_lin = Ridge(alpha=1.0)
        ridge_lin.fit(X_lin_s, y_tr)

        lin_errs = []
        for ev_i in range(len(tw_test)):
            key = (ev_i, L)
            if key not in test_preds:
                continue
            p = test_preds[key]
            iss = p["issue_idx"]
            feats = grid.iloc[iss][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            T_lin = ridge_lin.predict(scaler_lin.transform(feats.reshape(1,-1)))[0]
            lin_errs.append(p["y_actual"] - T_lin)

        rmse_nb = np.sqrt(np.mean(np.array(nb_errs)**2))
        rmse_lin = np.sqrt(np.mean(np.array(lin_errs)**2))
        rmse_ridge = np.sqrt(np.mean(np.array(ridge_errs)**2))
        d_lin = 100*(rmse_ridge - rmse_lin)/rmse_lin

        print(f"{L:>5} {rmse_nb:>8.3f} {rmse_lin:>8.3f} {rmse_ridge:>9.3f} {d_lin:>8.1f}%")

    print("\nNB+Ridge = Ridge(linear_features + T_nbeats) -> predict T_actual")
    print("Δ vs Lin: negative = NB+Ridge beats pure Linear")


if __name__ == "__main__":
    main()
