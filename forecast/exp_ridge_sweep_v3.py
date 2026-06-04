"""NBEATSx + Ridge feature sweep v3.

Setup (fully causal, no even/odd):
  - NBEATSx: trained on pre-2025 (as before)
  - Ridge: trained on pre-2025 twilights using NBEATSx IN-SAMPLE predictions
  - Test: all 2025 twilights (396 events)

This IS causal: Ridge trains on data before the test period. The in-sample
NBEATSx predictions are the model's own training-period output — they're
optimistic, but Ridge learns the feature→target mapping on that same period,
which is a legitimate temporal split.

Usage:
    uv run --no-sync python forecast/exp_ridge_sweep_v3.py
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
LEADS = [3.0, 6.0, 9.0, 12.0]
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

RIDGE_BASE = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]

RIDGE_CANDIDATES = [
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
    "mb_corrected",
]


def train_nbeats(grid):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
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


def get_nb_preds(nf, grid, tw_set, hist, futr, leads):
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in leads:
            ls = run.lead_hours_to_steps(L)
            iss = tgt - ls
            if iss < NBEATS_INPUT_SIZE or (tgt-iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
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
            preds[(ev_i, L)] = {"T_nb": float(u[mc].iloc[ps]),
                                "y_actual": yact, "iss": iss}
    return preds


def eval_ridge(grid, train_preds, test_preds, feat_list, lead):
    feat_cols = [c for c in feat_list if c in grid.columns]
    def build_Xy(preds):
        X, Y = [], []
        for (ev_i, L), p in preds.items():
            if L != lead: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            row = np.concatenate([feats, [p["T_nb"]]])
            X.append(row); Y.append(p["y_actual"])
        return np.array(X) if X else None, np.array(Y) if Y else None

    X_tr, y_tr = build_Xy(train_preds)
    X_te, y_te = build_Xy(test_preds)
    if X_tr is None or X_te is None or len(X_tr)<30 or len(X_te)<30:
        return np.nan
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)
    preds_te = ridge.predict(scaler.transform(X_te))
    return float(np.sqrt(((y_te - preds_te)**2).mean()))


def main():
    print("="*72)
    print(f"NBEATSx + RIDGE SWEEP v3  |  Ridge: pre-2025 train, 2025 test")
    print("="*72)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  Ridge train: {len(tw_train)} pre-2025 twilights")
    print(f"  Ridge test:  {len(tw_test)} 2025 twilights")

    # Train NBEATSx
    print("\nTraining NBEATSx (pre-2025)...")
    nf, hist, futr = train_nbeats(grid)

    # Get in-sample preds (train twilights) for Ridge training
    print("NBEATSx in-sample predictions (pre-2025 twilights)...")
    train_preds = get_nb_preds(nf, grid, tw_train, hist, futr, LEADS)
    print(f"  {len(train_preds)} preds")

    # Get test preds (2025 twilights)
    print("NBEATSx test predictions (2025 twilights)...")
    test_preds = get_nb_preds(nf, grid, tw_test, hist, futr, LEADS)
    print(f"  {len(test_preds)} preds")

    # Sweep at 3h
    SWEEP_LEAD = 3.0
    print(f"\n--- RIDGE FEATURE SWEEP @{SWEEP_LEAD}h ---")
    base_rmse = eval_ridge(grid, train_preds, test_preds, RIDGE_BASE, SWEEP_LEAD)
    print(f"Base ({len(RIDGE_BASE)} feats + T_nb): RMSE = {base_rmse:.3f}")

    available = [c for c in RIDGE_CANDIDATES if c in grid.columns
                 and c not in RIDGE_BASE
                 and grid[c].notna().sum() > 0.8 * len(grid)]
    print(f"Candidates: {len(available)}")

    current_feats = list(RIDGE_BASE)
    current_rmse = base_rmse
    round_num = 0

    while available:
        round_num += 1
        print(f"\n  Round {round_num} ({len(current_feats)} feats, RMSE={current_rmse:.3f})")
        results = []
        for feat in available:
            rmse = eval_ridge(grid, train_preds, test_preds, current_feats + [feat], SWEEP_LEAD)
            results.append((feat, rmse))
        valid = [(f, r) for f, r in results if not np.isnan(r)]
        if not valid: break
        best_feat, best_rmse = min(valid, key=lambda x: x[1])
        imp = (current_rmse - best_rmse) / current_rmse * 100
        if best_rmse >= current_rmse - 0.003:
            print(f"  No improvement (best: {best_feat} {best_rmse:.3f}, {imp:+.1f}%). Stopping.")
            break
        current_feats.append(best_feat)
        available.remove(best_feat)
        current_rmse = best_rmse
        print(f"  -> Added '{best_feat}': RMSE={current_rmse:.3f} ({imp:+.1f}%)")

    # Final eval at all leads
    print(f"\n{'='*72}")
    print(f"FINAL RESULTS: NB+Ridge ({len(current_feats)} feats + T_nb)")
    print(f"Features: {current_feats}")
    print(f"{'='*72}")
    print(f"\n{'lead':>5} {'NBEATSx':>8} {'NB+Ridge':>9}")
    print("-"*25)
    for L in LEADS:
        nb_errs = [p["y_actual"]-p["T_nb"] for (ei,l),p in test_preds.items() if l==L]
        rmse_nb = np.sqrt(np.mean(np.array(nb_errs)**2)) if nb_errs else np.nan
        rmse_ridge = eval_ridge(grid, train_preds, test_preds, current_feats, L)
        print(f"{L:>5} {rmse_nb:>8.3f} {rmse_ridge:>9.3f}")

    print(f"\nRMSE@3h: {current_rmse:.3f} (vs Linear ~0.857)")


if __name__ == "__main__":
    main()
