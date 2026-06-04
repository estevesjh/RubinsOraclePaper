"""Final paper model: direct-T NBEATSx + Ridge (two-stage).

Produces results/paper_results_diff.csv in the same format as run.py,
compatible with build_final.py and src/plot.py.

Architecture:
  NBEATSx: direct-T, width=16, 700 steps, pre-2025 train
  Ridge:   13 features + T_nb, pre-2025 twilight train, 2025 test

Usage:
    uv run --no-sync python forecast/run_final_model.py
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

sys.path.insert(0, os.path.dirname(__file__))
import run
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON, RESULTS_PATH,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS,
                    LEAD_TIMES_HOURS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
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
    print(f"  NBEATSx: {len(nf_train)} pre-2025 rows, width={WIDTH}, steps={MAX_STEPS}")
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=True,
        enable_model_summary=False,
        stack_types=["trend","seasonality","identity","exogenous"],
        mlp_units=4*[[WIDTH,WIDTH]], n_blocks=[1,1,1,1],
        early_stop_patience_steps=10, val_check_steps=50,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train)*0.1))
    return nf, hist, futr


def get_all_preds(nf, grid, tw_set, hist, futr, leads):
    """Get NBEATSx predictions for all twilights at all leads."""
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
            requests.append((ev_i, tgt, iss, L, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(requests), BATCH):
        if b0 % (BATCH*20) == 0 and b0 > 0:
            print(f"    batch {b0//BATCH}/{len(requests)//BATCH}...")
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
            meta.append((uid, tgt, iss, L, yact, ev_i, ds_r))
        if not all_h: continue
        with run.suppress_stdout():
            fc=nf.predict(pd.concat(all_h,ignore_index=True),
                         futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc=[c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, L, yact, ev_i, ds_r) in meta:
            u=fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            ps=tgt-iss
            if len(u)==0 or ps>=len(u): continue
            preds[(ev_i, L)] = {"T_nb": float(u[mc].iloc[ps]),
                                "y_actual": yact, "iss": iss, "ds_real": ds_r}
    return preds


def train_and_apply_ridge(grid, train_preds, test_preds, leads):
    """Train Ridge per-lead on train, apply to test. Returns list of result rows."""
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    rows = []
    for L in leads:
        # Build train
        X_tr, y_tr = [], []
        for (ev_i, lead), p in train_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            X_tr.append(np.concatenate([feats, [p["T_nb"]]]))
            y_tr.append(p["y_actual"])
        if len(X_tr) < 30:
            continue
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        scaler = StandardScaler()
        ridge = SkRidge(alpha=1.0)
        ridge.fit(scaler.fit_transform(X_tr), y_tr)

        # Apply to test
        for (ev_i, lead), p in test_preds.items():
            if lead != L: continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)): continue
            row_x = np.concatenate([feats, [p["T_nb"]]])
            T_pred = ridge.predict(scaler.transform(row_x.reshape(1,-1)))[0]
            rows.append({
                "twilight_time": pd.Timestamp(p["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "forecast_time": pd.Timestamp(p["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "lead_time_hours": L,
                "actual_temp": p["y_actual"],
                "model": "NBEATSx-Diff",
                "forecast_temp": T_pred,
                "error": p["y_actual"] - T_pred,
            })
    return rows


def main():
    print("="*72)
    print("FINAL MODEL: direct-T NBEATSx + Ridge (two-stage)")
    print("="*72)

    grid, _ = run.load_and_prepare()
    # Add extra features needed by Ridge (not in run.py's standard load)
    import feature_sweep as fs
    grid = fs.compute_all_features(grid)

    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}")

    # Train NBEATSx
    print("\nTraining NBEATSx...")
    nf, hist, futr = train_nbeats(grid)

    # Get predictions at all leads
    print("\nNBEATSx predictions (train)...")
    train_preds = get_all_preds(nf, grid, tw_train, hist, futr, LEAD_TIMES_HOURS)
    print(f"  {len(train_preds)} preds")
    print("NBEATSx predictions (test)...")
    test_preds = get_all_preds(nf, grid, tw_test, hist, futr, LEAD_TIMES_HOURS)
    print(f"  {len(test_preds)} preds")

    # Ridge correction
    print("\nTraining + applying Ridge...")
    result_rows = train_and_apply_ridge(grid, train_preds, test_preds, LEAD_TIMES_HOURS)
    results = pd.DataFrame(result_rows)
    print(f"  {len(results)} result rows")

    # Save
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    out = RESULTS_PATH / "paper_results_diff.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Summary
    print(f"\n{'lead':>6} {'RMSE':>7} {'<1C':>5} {'N':>4}")
    print("-"*25)
    for L in [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]:
        s = results[results["lead_time_hours"]==L]
        if len(s)==0: continue
        e = s["error"].values
        rmse = np.sqrt((e**2).mean()); p1 = (np.abs(e)<1).mean()*100
        print(f"{L:>6} {rmse:>7.3f} {p1:>4.0f}% {len(s):>4}")


if __name__ == "__main__":
    main()
