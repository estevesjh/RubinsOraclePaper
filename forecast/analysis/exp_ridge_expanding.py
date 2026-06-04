"""Expanding-window Ridge correction: retrain Ridge monthly.

NBEATSx is trained ONCE on pre-2025 (fixed). For each month M in 2025:
  - Ridge trains on all available data before M:
    * pre-2025 twilights (in-sample NBEATSx preds) PLUS
    * earlier-2025 months (out-of-sample NBEATSx preds — real residuals)
  - Test on month M twilights.

This tests whether Ridge benefits from seeing real 2025 extrapolation
residuals as they accumulate (online Ridge correction).

Comparison:
  - NBEATSx alone (no Ridge, fixed)
  - Ridge (pre-2025 only, fixed — the current paper model)
  - Ridge (expanding — retrains monthly with accumulating 2025 data)

Usage:
    uv run --no-sync python forecast/exp_ridge_expanding.py
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
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS,
                    LEAD_TIMES_HOURS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
LEAD_H = 3.0
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
    lead_steps = run.lead_hours_to_steps(LEAD_H)
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        iss = tgt - lead_steps
        if iss < NBEATS_INPUT_SIZE or (tgt-iss) >= NBEATS_HORIZON:
            continue
        requests.append((ev_i, tgt, iss, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, yact, ds_r) in batch:
            hs=iss-NBEATS_INPUT_SIZE; fe=iss+NBEATS_HORIZON
            if hs<0 or fe>len(grid): continue
            hd=grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd=grid.iloc[iss:fe][["ds"]+futr].copy()
            if fd[futr].isna().any().any(): continue
            uid=f"p_{b0}_{ev_i}"
            hd["unique_id"]=uid; fd["unique_id"]=uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, yact, ev_i, ds_r))
        if not all_h: continue
        with run.suppress_stdout():
            fc=nf.predict(pd.concat(all_h,ignore_index=True),
                         futr_df=pd.concat(all_f,ignore_index=True)).reset_index()
        mc=[c for c in fc.columns if c not in ["unique_id","ds","index"]][0]
        for (uid, tgt, iss, yact, ev_i, ds_r) in meta:
            u=fc[fc["unique_id"]==uid].sort_values("ds").reset_index(drop=True)
            ps=tgt-iss
            if len(u)==0 or ps>=len(u): continue
            preds[ev_i] = {"T_nb": float(u[mc].iloc[ps]),
                           "y_actual": yact, "iss": iss, "ds_real": ds_r}
    return preds


def ridge_predict(grid, train_preds, test_preds):
    """Train Ridge on train_preds, predict test_preds. Return errors."""
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    X_tr, y_tr = [], []
    for p in train_preds.values():
        feats = grid.iloc[p["iss"]][feat_cols].values
        if np.any(pd.isna(feats)): continue
        X_tr.append(np.concatenate([feats, [p["T_nb"]]]))
        y_tr.append(p["y_actual"])
    if len(X_tr) < 30:
        return None
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    scaler = StandardScaler()
    ridge = SkRidge(alpha=1.0)
    ridge.fit(scaler.fit_transform(X_tr), y_tr)

    errs = []
    for p in test_preds.values():
        feats = grid.iloc[p["iss"]][feat_cols].values
        if np.any(pd.isna(feats)): continue
        row = np.concatenate([feats, [p["T_nb"]]])
        pred = ridge.predict(scaler.transform(row.reshape(1,-1)))[0]
        errs.append(p["y_actual"] - pred)
    return np.array(errs) if errs else None


def main():
    print("="*72)
    print("EXPANDING-WINDOW RIDGE  |  monthly retrain @3h")
    print("="*72)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_2025 = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  pre-2025 twilights: {len(tw_train)}")
    print(f"  2025 twilights: {len(tw_2025)}")

    # Train NBEATSx once
    print("\nTraining NBEATSx (fixed, pre-2025)...")
    nf, hist, futr = train_nbeats(grid)

    # Get ALL predictions (pre-2025 + 2025)
    print("Predictions (pre-2025)...")
    pre_preds = get_preds(nf, grid, tw_train, hist, futr)
    print(f"  {len(pre_preds)}")
    print("Predictions (2025)...")
    preds_2025 = get_preds(nf, grid, tw_2025, hist, futr)
    print(f"  {len(preds_2025)}")

    # Monthly expanding window
    months = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    print(f"\n{'month':>8} {'Ridge_train_N':>13} {'NB_only':>8} {'Ridge_fixed':>11} {'Ridge_exp':>10}")
    print("-"*55)

    # Fixed Ridge (pre-2025 only) for comparison
    fixed_errs_all = []
    expanding_errs_all = []
    nb_errs_all = []

    accumulated_2025_preds = {}

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)

        # This month's preds
        month_preds = {k: v for k, v in preds_2025.items()
                       if pd.Timestamp(v["ds_real"]) >= m_start
                       and pd.Timestamp(v["ds_real"]) < m_end}
        if not month_preds:
            continue

        # NBEATSx alone
        nb_e = np.array([v["y_actual"] - v["T_nb"] for v in month_preds.values()])
        rmse_nb = np.sqrt((nb_e**2).mean())
        nb_errs_all.extend(nb_e)

        # Ridge fixed (pre-2025 only)
        fixed_e = ridge_predict(grid, pre_preds, month_preds)
        rmse_fixed = np.sqrt((fixed_e**2).mean()) if fixed_e is not None else np.nan
        if fixed_e is not None:
            fixed_errs_all.extend(fixed_e)

        # Ridge expanding (pre-2025 + accumulated earlier 2025 months)
        expanding_train = dict(pre_preds)
        expanding_train.update(accumulated_2025_preds)
        exp_e = ridge_predict(grid, expanding_train, month_preds)
        rmse_exp = np.sqrt((exp_e**2).mean()) if exp_e is not None else np.nan
        if exp_e is not None:
            expanding_errs_all.extend(exp_e)

        n_ridge_train = len(pre_preds) + len(accumulated_2025_preds)
        print(f"{m_start.strftime('%Y-%m'):>8} {n_ridge_train:>13} "
              f"{rmse_nb:>8.3f} {rmse_fixed:>11.3f} {rmse_exp:>10.3f}")

        # Accumulate this month for future Ridge training
        accumulated_2025_preds.update(month_preds)

    # Overall
    print(f"\n{'OVERALL':>8} {'':>13} "
          f"{np.sqrt(np.mean(np.array(nb_errs_all)**2)):>8.3f} "
          f"{np.sqrt(np.mean(np.array(fixed_errs_all)**2)):>11.3f} "
          f"{np.sqrt(np.mean(np.array(expanding_errs_all)**2)):>10.3f}")
    print(f"\n  Δ expanding vs fixed: "
          f"{100*(np.sqrt(np.mean(np.array(expanding_errs_all)**2)) - np.sqrt(np.mean(np.array(fixed_errs_all)**2)))/np.sqrt(np.mean(np.array(fixed_errs_all)**2)):+.1f}%")


if __name__ == "__main__":
    main()
