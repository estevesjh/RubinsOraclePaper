"""Feature sweep for the Ridge correction stage (NBEATSx + Ridge).

Fixed: NBEATSx (width=16, 700 steps, direct-T, pre-2025 train).
Sweep: which features in the Ridge stage improve RMSE@3h beyond
       [LINEAR_FEATS + T_nb] baseline.

Candidates: all FeatureBuilder columns + lag/rate/diff features evaluated
at the issue index. Greedy forward selection on test-set RMSE@3h.

Usage:
    uv run --no-sync python forecast/exp_ridge_sweep.py
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
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
LEAD_H = 3.0
HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

# Base Ridge features (the 10 from Linear + T_nb)
BASE_RIDGE = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]

# Additional candidates to sweep
CANDIDATES = [
    "y_lag_96", "y_diff_24h", "y_diff_60", "y_diff_72h",
    "y_diff_30h", "y_diff_39h", "y_diff_45h",
    "cooling_rate_3h", "cooling_rate_6h",
    "trend_solar_4h", "trend_solar_6h",
    "last_std_24h", "last_max_24h", "last_min_24h",
    "DTR", "DTR_3d", "dDTR_1d",
    "dmean_1d", "dmean_3d",
    "dTmax_1d", "dTmax_3d", "temp_trend_3d", "Tmean_24h",
    "max_minus_sunrise", "temp_last_sunrise",
    "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
    "rate_sunrise_to_midday", "rate_midday_to_twilight",
    "velocity_sunrise", "velocity_noon",
]


def train_nbeats(grid):
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


def get_nbeats_preds(nf, grid, tw_set, hist, futr):
    """Get T_nb for each twilight at LEAD_H."""
    allx = hist + futr
    ls = run.lead_hours_to_steps(LEAD_H)
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        iss = tgt - ls
        if iss < NBEATS_INPUT_SIZE or (tgt-iss) >= NBEATS_HORIZON:
            continue
        requests.append((ev_i, tgt, iss, ev["y_actual"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, yact) in batch:
            hs=iss-NBEATS_INPUT_SIZE; fe=iss+NBEATS_HORIZON
            if hs<0 or fe>len(grid): continue
            hd=grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd=grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid=f"p_{b0}_{ev_i}"
            hd["unique_id"]=uid; fd["unique_id"]=uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, yact, ev_i))
        if not all_h: continue
        with run.suppress_stdout():
            fc=nf.predict(pd.concat(all_h,ignore_index=True),
                         futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc=[c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, yact, ev_i) in meta:
            u=fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            ps=tgt-iss
            if len(u)==0 or ps>=len(u): continue
            preds[ev_i] = {"T_nb": float(u[mc].iloc[ps]), "y_actual": yact, "iss": iss}
    return preds


def eval_ridge(grid, train_preds, test_preds, tw_train, tw_test, feat_list):
    """Train Ridge on feat_list + T_nb, eval on test. Return RMSE@3h."""
    feat_cols = [c for c in feat_list if c in grid.columns]
    if not feat_cols:
        return np.nan

    X_tr, y_tr = [], []
    for ev_i, p in train_preds.items():
        iss = p["iss"]
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)):
            continue
        row = np.concatenate([feats, [p["T_nb"]]])
        X_tr.append(row)
        y_tr.append(p["y_actual"])
    if len(X_tr) < 50:
        return np.nan

    X_tr = np.array(X_tr); y_tr = np.array(y_tr)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, y_tr)

    errs = []
    for ev_i, p in test_preds.items():
        iss = p["iss"]
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)):
            continue
        row = np.concatenate([feats, [p["T_nb"]]])
        pred = ridge.predict(scaler.transform(row.reshape(1,-1)))[0]
        errs.append(p["y_actual"] - pred)

    if not errs:
        return np.nan
    return float(np.sqrt(np.mean(np.array(errs)**2)))


def main():
    print("="*72)
    print(f"RIDGE FEATURE SWEEP  |  NBEATSx+Ridge @{LEAD_H}h")
    print("="*72)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train: {len(tw_train)}, test: {len(tw_test)}")

    # Train NBEATSx once
    print("\nTraining NBEATSx...")
    nf, hist, futr = train_nbeats(grid)

    # Get predictions
    print("NBEATSx predictions (train)...")
    train_preds = get_nbeats_preds(nf, grid, tw_train, hist, futr)
    print(f"  {len(train_preds)} train preds")
    print("NBEATSx predictions (test)...")
    test_preds = get_nbeats_preds(nf, grid, tw_test, hist, futr)
    print(f"  {len(test_preds)} test preds")

    # Base Ridge: LINEAR_FEATS + T_nb
    base_rmse = eval_ridge(grid, train_preds, test_preds, tw_train, tw_test, BASE_RIDGE)
    print(f"\nBase Ridge ({len(BASE_RIDGE)} feats + T_nb): RMSE@3h = {base_rmse:.3f}")

    # Greedy forward selection
    available = [c for c in CANDIDATES if c in grid.columns
                 and c not in BASE_RIDGE
                 and grid[c].notna().sum() > 0.8 * len(grid)]
    print(f"Candidates: {len(available)}")

    current_feats = list(BASE_RIDGE)
    current_rmse = base_rmse
    round_num = 0

    while available:
        round_num += 1
        print(f"\n  Round {round_num} (current {len(current_feats)} feats, RMSE={current_rmse:.3f})")
        results = []
        for feat in available:
            test_feats = current_feats + [feat]
            rmse = eval_ridge(grid, train_preds, test_preds, tw_train, tw_test, test_feats)
            imp = (current_rmse - rmse) / current_rmse * 100 if not np.isnan(rmse) else 0
            results.append((feat, rmse, imp))
            if imp > 1:
                print(f"    + {feat:>28}: {rmse:.3f} ({imp:+.1f}%)")

        valid = [(f, r, i) for f, r, i in results if not np.isnan(r)]
        if not valid:
            break
        best_feat, best_rmse, best_imp = min(valid, key=lambda x: x[1])
        if best_rmse >= current_rmse - 0.003:
            print(f"  No improvement. Stopping.")
            break
        current_feats.append(best_feat)
        available.remove(best_feat)
        current_rmse = best_rmse
        print(f"  -> Added '{best_feat}': RMSE={current_rmse:.3f}")

    print("\n" + "="*72)
    print(f"OPTIMAL RIDGE FEATURES ({len(current_feats)} + T_nb):")
    print(f"  {current_feats}")
    print(f"  RMSE@3h = {current_rmse:.3f}")
    print(f"  vs base Linear alone: check exp_nbeats_ridge.py (was 0.857)")
    print(f"  vs base NB+Ridge (10 feats): {base_rmse:.3f}")


if __name__ == "__main__":
    main()
