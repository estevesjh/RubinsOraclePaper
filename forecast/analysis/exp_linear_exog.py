"""NBEATSx + per-horizon Linear forecast as future exogenous.

For each horizon step k (k=0..HORIZON-1), we train a separate Ridge model:
    Ridge_k: features at issue_time -> T at issue_time + k steps

At predict time, for a window with issue at `iss`, the futr_df column is:
    linear_futr[k] = Ridge_k(features at iss)

This is causal at every horizon step: all Ridge_k use features evaluated at
the single issue time `iss`, which is in the past of all horizon steps.

For TRAINING, the futr_exog column is built per-row as if that row is the
target of a Ridge whose issue was k steps earlier. Since NeuralForecast
slides windows, the row at position `iss+k` in a window issued at `iss`
should hold Ridge_k's prediction. We approximate by storing the Ridge_0
prediction (0-step-ahead = "nowcast") for training — not ideal, but:
- Alternative: pre-compute all 26 Ridge outputs per row → pick the right
  k per window. NeuralForecast doesn't support per-window futr construction
  during training.
- Better alternative: use the STATIC column `linear_3h(t) = Ridge_6(feat@t-6)`
  for training (the signal shape is similar), then at predict time fill in
  the correct per-k trajectory. This is what we do.

Usage:
    uv run --no-sync python forecast/exp_linear_exog.py
"""

import os
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import logging
for _n in ["pytorch_lightning", "lightning.pytorch", "lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run  # noqa: E402
from config import (  # noqa: E402
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, STEPS_PER_DAY,
    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS,
)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS  # 700
LEADS = [3.0, 6.0, 12.0]
BASE_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
DET_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

LINEAR_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]


def train_ridge_per_horizon(grid):
    """Train one Ridge per horizon step k=0..HORIZON-1.
    Ridge_k: features at iss -> y at iss+k.
    Returns list of (scaler, ridge) tuples and a training-proxy static column.
    """
    y = grid["y"].values
    ds_real = pd.to_datetime(grid["ds_real"])
    train_mask = ds_real < TEST_START_DATE
    feat_cols = [c for c in LINEAR_FEATS if c in grid.columns]
    n = len(grid)

    # Gather valid issue indices in pre-2025 where all features + targets exist
    valid_iss = []
    for iss in range(NBEATS_INPUT_SIZE, n - NBEATS_HORIZON):
        if not train_mask.iloc[iss]:
            continue
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)):
            continue
        targets = y[iss:iss + NBEATS_HORIZON]
        if np.any(np.isnan(targets)):
            continue
        valid_iss.append(iss)
    valid_iss = np.array(valid_iss)
    print(f"  Ridge per-horizon: {len(valid_iss)} valid issue points")

    X = np.array([grid.iloc[i][feat_cols].values for i in valid_iss])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    ridges = []
    for k in range(NBEATS_HORIZON):
        Y_k = y[valid_iss + k]
        r = Ridge(alpha=1.0)
        r.fit(X_s, Y_k)
        ridges.append(r)

    # Training proxy column: use Ridge_6 (3h-ahead) prediction for each row.
    # This gives the network the right "shape" of signal during training.
    proxy_lead = 6
    linear_proxy = np.full(n, np.nan)
    for t in range(proxy_lead, n):
        iss = t - proxy_lead
        feats = grid.iloc[iss][feat_cols].values
        if np.any(pd.isna(feats)):
            continue
        linear_proxy[t] = ridges[proxy_lead].predict(
            scaler.transform(feats.reshape(1, -1)))[0]
    grid["linear_futr"] = linear_proxy

    # Report proxy quality
    valid = ~np.isnan(linear_proxy) & ~np.isnan(y)
    test_m = valid & ~train_mask.values
    rmse_test = np.sqrt(np.mean((linear_proxy[test_m] - y[test_m]) ** 2))
    print(f"  Proxy column (Ridge_6) test RMSE: {rmse_test:.3f}")

    return ridges, scaler, feat_cols


def build_linear_futr_traj(ridges, scaler, grid, feat_cols, iss):
    """Per-request: Ridge_k(features at iss) for k=0..HORIZON-1."""
    feats = grid.iloc[iss][feat_cols].values
    if np.any(pd.isna(feats)):
        return None
    X = scaler.transform(feats.reshape(1, -1))
    traj = np.array([ridges[k].predict(X)[0] for k in range(NBEATS_HORIZON)])
    return traj


def train_nbeats(grid, futr_exog):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in BASE_HIST if c in grid.columns]
    futr = [c for c in futr_exog if c in grid.columns]
    allx = hist + futr

    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds", "y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"

    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[WIDTH, WIDTH]], n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10, val_check_steps=50,
        accelerator="mps", devices=1,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train) * 0.1))
    return nf, hist, futr


def eval_nbeats_with_linear_traj(nf, grid, tw_test, hist, futr,
                                  ridges, scaler, feat_cols):
    """Predict with per-request Linear trajectory in futr_df."""
    allx = hist + futr
    requests = []
    for ev_i, (_, ev) in enumerate(tw_test.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in LEADS:
            off = run.lead_hours_to_steps(L)
            iss = tgt - off
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))

    errs = {L: [] for L in LEADS}
    BATCH = 64
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0 + BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid): continue
            hd = grid.iloc[hs:iss][["ds", "y"] + allx].copy()
            if hd[allx].isna().any().any(): continue
            # Build per-request linear trajectory
            fd = grid.iloc[iss:fe][["ds"] + [c for c in DET_FUTR if c in grid.columns]].copy()
            traj = build_linear_futr_traj(ridges, scaler, grid, feat_cols, iss)
            if traj is None: continue
            fd["linear_futr"] = traj
            if fd.isna().any().any(): continue
            uid = f"r_{b0}_{ev_i}_{L}"
            hd["unique_id"] = uid; fd["unique_id"] = uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, L, yact))
        if not all_h: continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h, ignore_index=True),
                            futr_df=pd.concat(all_f, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, tgt, iss, L, yact) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u) == 0 or ps >= len(u): continue
            errs[L].append(yact - float(u[mc].iloc[ps]))

    out = {}
    for L in LEADS:
        e = np.array(errs[L])
        out[L] = (float(np.sqrt((e ** 2).mean())), float((np.abs(e) < 1).mean() * 100), len(e))
    return out


def eval_base(nf, grid, tw_test, hist, futr):
    """Base eval (static futr only, no linear traj at predict time)."""
    allx = hist + futr
    requests = []
    for ev_i, (_, ev) in enumerate(tw_test.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in LEADS:
            off = run.lead_hours_to_steps(L)
            iss = tgt - off
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))
    errs = {L: [] for L in LEADS}
    BATCH = 64
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0 + BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid): continue
            hd = grid.iloc[hs:iss][["ds", "y"] + allx].copy()
            if hd[allx].isna().any().any(): continue
            fd = grid.iloc[iss:fe][["ds"] + futr].copy()
            if fd[futr].isna().any().any(): continue
            uid = f"b_{b0}_{ev_i}_{L}"
            hd["unique_id"] = uid; fd["unique_id"] = uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, L, yact))
        if not all_h: continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h, ignore_index=True),
                            futr_df=pd.concat(all_f, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, tgt, iss, L, yact) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u) == 0 or ps >= len(u): continue
            errs[L].append(yact - float(u[mc].iloc[ps]))
    out = {}
    for L in LEADS:
        e = np.array(errs[L])
        out[L] = (float(np.sqrt((e ** 2).mean())), float((np.abs(e) < 1).mean() * 100), len(e))
    return out


def main():
    print("=" * 72)
    print(f"LINEAR-PER-HORIZON AS FUTR-EXOG  |  width={WIDTH} steps={MAX_STEPS}")
    print("=" * 72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  test twilights: {len(tw_test)}\n")

    # Train 26 Ridge models (one per horizon step)
    ridges, scaler, feat_cols = train_ridge_per_horizon(grid)

    # --- Base NBEATSx (no linear futr) ---
    print("\n>>> base (no linear futr_exog)")
    nf_b, hist_b, futr_b = train_nbeats(grid, DET_FUTR)
    r_base = eval_base(nf_b, grid, tw_test, hist_b, futr_b)
    for L in LEADS:
        print(f"    {L}h: RMSE={r_base[L][0]:.3f}  <1C={r_base[L][1]:.0f}%")

    # --- +linear per-horizon trajectory ---
    print("\n>>> +linear_futr (per-request Ridge_k trajectory)")
    nf_l, hist_l, futr_l = train_nbeats(grid, DET_FUTR + ["linear_futr"])
    r_lin = eval_nbeats_with_linear_traj(nf_l, grid, tw_test, hist_l, futr_l,
                                          ridges, scaler, feat_cols)
    for L in LEADS:
        print(f"    {L}h: RMSE={r_lin[L][0]:.3f}  <1C={r_lin[L][1]:.0f}%")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'variant':>16}  {'3h':>12}  {'6h':>12}  {'12h':>12}")
    for name, r in [("base", r_base), ("+linear_futr", r_lin)]:
        parts = []
        for L in LEADS:
            rmse = r[L][0]; d = 100 * (rmse - r_base[L][0]) / r_base[L][0]
            parts.append(f"{rmse:.3f}({d:+.0f}%)")
        print(f"{name:>16}  {'  '.join(parts)}")

    print("\nDecision gate (>7% improvement over base):")
    for L in LEADS:
        imp = 100 * (r_base[L][0] - r_lin[L][0]) / r_base[L][0]
        print(f"  {L}h: {imp:+.1f}%  {'PASS' if imp > 7 else 'below gate'}")


if __name__ == "__main__":
    main()
