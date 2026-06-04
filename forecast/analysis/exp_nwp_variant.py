"""Compare NBEATSx+Ridge: local-only vs local+NWP (mb_corrected as futr_exog).

Two NBEATSx models:
  - local:  futr = [solar_sin, solar_cos, doy_sin, doy_cos]
  - +NWP:   futr = [solar_sin, solar_cos, doy_sin, doy_cos, mb_corrected]

Both: direct-T, width=16, 700 steps, pre-2025 train.
Ridge stage: same 13 features + T_nb, pre-2025 train, 2025 test.

Usage:
    uv run --no-sync python forecast/exp_nwp_variant.py
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
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
LEADS = [0.5, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0]
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
DET_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
]


def train_nbeats(grid, futr_list):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in futr_list if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds","y"]+allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    print(f"    train rows: {len(nf_train)}, futr: {futr}")
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


def get_preds(nf, grid, tw_set, hist, futr):
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in LEADS:
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


def run_ridge(grid, train_preds, test_preds):
    """Train + eval Ridge per lead. Returns dict[lead] -> rmse."""
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    results = {}
    for L in LEADS:
        X_tr, y_tr, X_te, y_te = [], [], [], []
        for (ev_i, lead), p in train_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            X_tr.append(np.concatenate([feats, [p["T_nb"]]]))
            y_tr.append(p["y_actual"])
        for (ev_i, lead), p in test_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            X_te.append(np.concatenate([feats, [p["T_nb"]]]))
            y_te.append(p["y_actual"])
        if len(X_tr) < 30 or len(X_te) < 30:
            results[L] = np.nan; continue
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)
        scaler = StandardScaler()
        ridge = SkRidge(alpha=1.0)
        ridge.fit(scaler.fit_transform(X_tr), y_tr)
        preds = ridge.predict(scaler.transform(X_te))
        results[L] = float(np.sqrt(((y_te - preds)**2).mean()))
    return results


def main():
    print("="*72)
    print("NBEATSx+Ridge: LOCAL vs LOCAL+NWP (mb_corrected as futr_exog)")
    print("="*72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train: {len(tw_train)}, test: {len(tw_test)}")
    print(f"  mb_corrected available: {'mb_corrected' in grid.columns}")

    # --- LOCAL variant ---
    print("\n>>> LOCAL (no mb_corrected in NBEATSx futr)")
    nf_loc, h_loc, f_loc = train_nbeats(grid, DET_FUTR)
    print("  in-sample preds...")
    tr_preds_loc = get_preds(nf_loc, grid, tw_train, h_loc, f_loc)
    print("  test preds...")
    te_preds_loc = get_preds(nf_loc, grid, tw_test, h_loc, f_loc)
    r_loc = run_ridge(grid, tr_preds_loc, te_preds_loc)

    # --- +NWP variant ---
    print("\n>>> +NWP (mb_corrected as futr_exog)")
    nf_nwp, h_nwp, f_nwp = train_nbeats(grid, DET_FUTR + ["mb_corrected"])
    print("  in-sample preds...")
    tr_preds_nwp = get_preds(nf_nwp, grid, tw_train, h_nwp, f_nwp)
    print("  test preds...")
    te_preds_nwp = get_preds(nf_nwp, grid, tw_test, h_nwp, f_nwp)
    r_nwp = run_ridge(grid, tr_preds_nwp, te_preds_nwp)

    # --- Summary ---
    print(f"\n{'='*72}")
    print(f"{'lead':>5} {'Local':>8} {'+NWP':>8} {'Δ%':>7}")
    print("-"*30)
    for L in LEADS:
        loc = r_loc.get(L, np.nan)
        nwp = r_nwp.get(L, np.nan)
        d = 100*(nwp - loc)/loc if not (np.isnan(loc) or np.isnan(nwp)) else np.nan
        print(f"{L:>5} {loc:>8.3f} {nwp:>8.3f} {d:>+6.1f}%")


if __name__ == "__main__":
    main()
