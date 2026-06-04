"""Why does Ridge beat NBEATSx? Capacity / regularization / generalization-gap study.

Hypothesis: at the twilight horizon the input->target map is near-linear, so the
~14-coefficient Ridge is a low-variance fit, while NBEATSx has enough capacity to
fit pre-2025 idiosyncrasies that do not transfer to the 2025 synoptic regime.
The diagnostic is the TRAIN vs TEST gap: if bigger nets push train RMSE down while
test RMSE rises, that is overfitting, and regularization (smaller net / dropout)
should help test even as it hurts train.

This sweeps NBEATSx (width, n_blocks, batch_size, dropout) on MPS, reporting BOTH
train and test 3h/9h RMSE for the raw NBEATSx prediction (no Ridge), plus the
Ridge-on-features baseline and the NBEATSx+Ridge blend, all on the same twilights.

Usage:
    EXP_ACCEL=mps uv run --no-sync python forecast/analysis/exp_capacity.py
"""

import os, sys, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning", "lightning.pytorch", "lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON, SOLAR_GRID_FREQ,
                    TEST_START_DATE, NBEATS_MAX_STEPS, LEAD_TIMES_HOURS)

ACCEL = os.environ.get("EXP_ACCEL", "mps")
MAX_STEPS = NBEATS_MAX_STEPS
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
]
LEADS = [3.0, 9.0]

# (label, width, n_blocks, batch_size, dropout)
CONFIGS = [
    ("paper w16 b1 bs48 d0",      16, 1, 48, 0.0),
    ("user  w16 b1 bs32 d0",      16, 1, 32, 0.0),
    ("bigger w64 b2 bs128 d0",    64, 2, 128, 0.0),
    ("bigger w64 b2 bs128 d0.2",  64, 2, 128, 0.2),
    ("big   w128 b2 bs256 d0.2", 128, 2, 256, 0.2),
    ("big   w128 b2 bs256 d0.3", 128, 2, 256, 0.3),
    ("huge  w256 b3 bs256 d0.3", 256, 3, 256, 0.3),
]


def train_nbeats(grid, width, n_blocks, batch_size, dropout):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx
    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds", "y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=batch_size, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[width, width]], n_blocks=[n_blocks] * 4,
        dropout_prob_theta=dropout,
        early_stop_patience_steps=10, val_check_steps=50,
        logger=False, enable_checkpointing=False,
        accelerator=ACCEL, devices=1,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train) * 0.1))
    return nf, hist, futr


def get_preds(nf, grid, tw_set, hist, futr, leads):
    allx = hist + futr
    preds = {}
    BATCH = 64
    reqs = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in leads:
            iss = tgt - run.lead_hours_to_steps(L)
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            reqs.append((ev_i, tgt, iss, L, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(reqs), BATCH):
        batch = reqs[b0:b0 + BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact, ds_r) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid):
                continue
            hd = grid.iloc[hs:iss][["ds", "y"] + allx].copy()
            if hd[allx].isna().any().any():
                continue
            fd = grid.iloc[iss:fe][["ds"] + futr].copy()
            if fd[futr].isna().any().any():
                continue
            uid = f"p_{b0}_{ev_i}_{L}"
            hd["unique_id"] = uid; fd["unique_id"] = uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, tgt, iss, L, yact, ev_i, ds_r))
        if not all_h:
            continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h, ignore_index=True),
                            futr_df=pd.concat(all_f, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, tgt, iss, L, yact, ev_i, ds_r) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u) == 0 or ps >= len(u):
                continue
            preds[(ev_i, L)] = {"T_nb": float(u[mc].iloc[ps]), "y": yact,
                                "iss": iss, "ds": ds_r}
    return preds


def rmse_by_lead(preds, leads):
    out = {}
    for L in leads:
        e = [p["y"] - p["T_nb"] for (k, l), p in preds.items() if l == L]
        out[L] = float(np.sqrt(np.mean(np.square(e)))) if e else np.nan
    return out


def ridge_blend_rmse(grid, tr_preds, te_preds, leads, feats, use_nb=True):
    """Train per-lead Ridge on (features [+ T_nb]); return test RMSE per lead."""
    feats = [c for c in feats if c in grid.columns]
    out = {}
    for L in leads:
        Xtr, ytr = [], []
        for (ev_i, l), p in tr_preds.items():
            if l != L: continue
            fv = grid.iloc[p["iss"]][feats].values
            if np.any(pd.isna(fv)): continue
            row = np.concatenate([fv, [p["T_nb"]]]) if use_nb else fv
            Xtr.append(row); ytr.append(p["y"])
        if len(Xtr) < 30:
            out[L] = np.nan; continue
        sc = StandardScaler(); rg = SkRidge(alpha=1.0)
        rg.fit(sc.fit_transform(np.array(Xtr)), np.array(ytr))
        e = []
        for (ev_i, l), p in te_preds.items():
            if l != L: continue
            fv = grid.iloc[p["iss"]][feats].values
            if np.any(pd.isna(fv)): continue
            row = np.concatenate([fv, [p["T_nb"]]]) if use_nb else fv
            e.append(p["y"] - rg.predict(sc.transform(row.reshape(1, -1)))[0])
        out[L] = float(np.sqrt(np.mean(np.square(e)))) if e else np.nan
    return out


def main():
    print("=" * 96)
    print(f"CAPACITY / REGULARIZATION STUDY  (accel={ACCEL}, max_steps={MAX_STEPS})")
    print("=" * 96)
    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_tr = tw[tw["ds_real"] < TEST_START_DATE].reset_index(drop=True)
    tw_te = tw[tw["ds_real"] >= TEST_START_DATE].reset_index(drop=True)

    # Ridge-only baseline (features, no NBEATSx) — the thing to beat
    # Build once using a throwaway preds dict keyed to issue rows via a dummy T_nb.
    print(f"\nRidge-only (features, no NBEATSx): the linear baseline")
    # reuse ridge_blend with use_nb=False; needs preds for the issue rows -> build minimal
    def minimal_preds(tw_set):
        d = {}
        for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
            tgt = int(ev["grid_idx"])
            for L in LEADS:
                iss = tgt - run.lead_hours_to_steps(L)
                if iss < 0 or iss >= len(grid): continue
                d[(ev_i, L)] = {"T_nb": 0.0, "y": ev["y_actual"], "iss": iss, "ds": ev["ds_real"]}
        return d
    mtr, mte = minimal_preds(tw_tr), minimal_preds(tw_te)
    r_ridge = ridge_blend_rmse(grid, mtr, mte, LEADS, RIDGE_FEATS, use_nb=False)
    print(f"  Ridge-only TEST: 3h={r_ridge[3.0]:.3f}  9h={r_ridge[9.0]:.3f}")

    hdr = f"\n{'config':26s} | {'NB train 3h/9h':>16} | {'NB test 3h/9h':>15} | {'gap 3h/9h':>13} | {'+Ridge test 3h/9h':>17}"
    print(hdr); print("-" * len(hdr))
    for (label, w, nb, bs, dp) in CONFIGS:
        nf, hist, futr = train_nbeats(grid, w, nb, bs, dp)
        tr_p = get_preds(nf, grid, tw_tr, hist, futr, LEADS)
        te_p = get_preds(nf, grid, tw_te, hist, futr, LEADS)
        rtr = rmse_by_lead(tr_p, LEADS)
        rte = rmse_by_lead(te_p, LEADS)
        rbl = ridge_blend_rmse(grid, tr_p, te_p, LEADS, RIDGE_FEATS, use_nb=True)
        g3 = rte[3.0] - rtr[3.0]; g9 = rte[9.0] - rtr[9.0]
        print(f"{label:26s} | {rtr[3.0]:6.3f}/{rtr[9.0]:6.3f}   | "
              f"{rte[3.0]:6.3f}/{rte[9.0]:6.3f} | {g3:+5.2f}/{g9:+5.2f}  | "
              f"{rbl[3.0]:6.3f}/{rbl[9.0]:6.3f}", flush=True)
    print(f"\nRidge-only baseline (repeat): 3h={r_ridge[3.0]:.3f}  9h={r_ridge[9.0]:.3f}")
    print("\nRead: large NB train<<test gap => overfitting; if +Ridge≈Ridge-only, NB adds little.")


if __name__ == "__main__":
    main()
