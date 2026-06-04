"""NBEATSx + persistence trajectory as future exogenous.

Per-request persistence trajectory (Section 3 of plan):
    For each request (issue_idx, horizon), build:
        persist_traj[k] = y(iss) + ( y_src(iss + k - 48) - y_src(iss - 48) )
    for k = 0, 1, ..., NBEATS_HORIZON-1.

    persist_traj is added as a column to futr_df for that request.
    It is a FUTR exog (never hist).

Two variants: y_src = y (raw) or y_src = y_smooth3h.
NBEATSx predicts absolute T directly.

Usage:
    uv run --no-sync python forecast/exp_persist_exog.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run  # noqa: E402
from config import (  # noqa: E402
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, STEPS_PER_DAY,
    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS,
)

WIDTH = 64
MAX_STEPS = NBEATS_MAX_STEPS  # 700
LEADS = [3.0, 6.0, 12.0]
BASE_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
DET_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]


def build_persist_traj(y_src, iss, horizon):
    """Build the persistence trajectory for a single request.
    persist_traj[k] = y(iss) + ( y_src(iss + k - 48) - y_src(iss - 48) )
    """
    level = y_src[iss]  # NOTE: we use y (not y_src) for the level? see below
    yesterday_now = y_src[iss - STEPS_PER_DAY]
    ks = np.arange(horizon)
    yesterday_future = y_src[iss + ks - STEPS_PER_DAY]
    return level + (yesterday_future - yesterday_now)


def train_with_persist(grid, y_src, persist_col_name):
    """Train NBEATSx with per-step persistence trajectory as futr_exog.

    Since the persist trajectory depends on the issue index (it's anchored on
    y(iss)), we can't pre-compute a single static column. Instead we:
    1. For TRAINING: use every row as a potential issue point and build the
       persist traj as a column on the grid. This is an approximation — for
       training, we treat each row as if it IS the issue row. NeuralForecast
       slides windows over the series, so each window's "issue" is its last row.
       The persist column at row t is: y(t) + (y_src(t+k-48) - y_src(t-48))
       evaluated for k=0 (the issue step). For the future steps k=1..H-1,
       the series naturally has the right values because we define:
           persist_col(t) = y_src(t - 48) + [y(iss_of_window) - y_src(iss_of_window - 48)]
       But we don't know iss_of_window at column-build time.

    Actually the clean solution: persist_col(t) = y_src(t - 48). Then the
    network learns to add the offset (y(iss) - y_src(iss-48)) from the hist
    features.  <-- THIS IS THE STATIC y_prevday APPROACH (Section 2).

    For the CORRECT Section 3 approach (per-request traj), we must build
    custom futr_df per request at PREDICT TIME. For training, we use the
    static y_prevday as a proxy (the network sees the same shape).
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    # Add static y_prevday column for training (proxy for the per-request traj)
    grid["persist_futr"] = np.roll(y_src, STEPS_PER_DAY)
    grid.iloc[:STEPS_PER_DAY, grid.columns.get_loc("persist_futr")] = np.nan

    hist = [c for c in BASE_HIST if c in grid.columns]
    futr = DET_FUTR + ["persist_futr"]
    futr = [c for c in futr if c in grid.columns]
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


def eval_with_persist(nf, grid, tw_test, y_src, hist, futr):
    """Predict with per-request persistence trajectory in futr_df (Section 3)."""
    y = grid["y"].values
    requests = []
    for ev_i, (_, ev) in enumerate(tw_test.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in LEADS:
            off = run.lead_hours_to_steps(L)
            iss = tgt - off
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            if iss - STEPS_PER_DAY < 0 or iss + NBEATS_HORIZON - STEPS_PER_DAY >= len(y_src):
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))

    allx = hist + futr
    errs = {L: [] for L in LEADS}
    BATCH = 64
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0 + BATCH]
        all_hist, all_futr, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
            hist_start = iss - NBEATS_INPUT_SIZE
            futr_end = iss + NBEATS_HORIZON
            if hist_start < 0 or futr_end > len(grid):
                continue
            # hist: use static persist_futr column (= y_src[t-48])
            hd = grid.iloc[hist_start:iss][["ds", "y"] + allx].copy()
            if hd[allx].isna().any().any():
                continue
            # futr: build per-request persist trajectory
            fd = grid.iloc[iss:futr_end][["ds"] + DET_FUTR].copy()
            # Section 3 trajectory:
            persist_traj = build_persist_traj(y_src, iss, NBEATS_HORIZON)
            fd["persist_futr"] = persist_traj
            if fd.isna().any().any():
                continue
            uid = f"r_{b0}_{ev_i}_{L}"
            hd["unique_id"] = uid
            fd["unique_id"] = uid
            all_hist.append(hd)
            all_futr.append(fd)
            meta.append((uid, tgt, iss, L, yact))
        if not all_hist:
            continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_hist, ignore_index=True),
                            futr_df=pd.concat(all_futr, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, tgt, iss, L, yact) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u) == 0 or ps >= len(u):
                continue
            errs[L].append(yact - float(u[mc].iloc[ps]))

    out = {}
    for L in LEADS:
        e = np.array(errs[L])
        out[L] = (float(np.sqrt((e ** 2).mean())), float((np.abs(e) < 1).mean() * 100), len(e))
    return out


def eval_base(grid, tw_test):
    """Base model: direct-T, no persist, same as earlier experiment."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in BASE_HIST if c in grid.columns]
    futr = [c for c in DET_FUTR if c in grid.columns]
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

    # Evaluate
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
            uid = f"b_{b0}_{ev_i}_{L}"
            hd["unique_id"] = uid; fd["unique_id"] = uid
            all_h.append(hd); all_f.append(fd); meta.append((uid, tgt, iss, L, yact))
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
    print(f"PERSIST-AS-FUTR-EXOG (Section 3)  |  width={WIDTH} steps={MAX_STEPS}")
    print("=" * 72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  test twilights: {len(tw_test)}\n")

    y = grid["y"].values
    y_sm = pd.Series(y).rolling(6, min_periods=3, center=True).mean().values

    # --- Base (no persist) ---
    print(">>> base (no persistence futr_exog)")
    r_base = eval_base(grid, tw_test)
    for L in LEADS:
        print(f"    {L}h: RMSE={r_base[L][0]:.3f}  <1C={r_base[L][1]:.0f}%  N={r_base[L][2]}")

    # --- +P_raw ---
    print("\n>>> +P_raw (per-request Pers traj, y_src = y)")
    nf_raw, hist_r, futr_r = train_with_persist(grid, y, "persist_raw")
    r_praw = eval_with_persist(nf_raw, grid, tw_test, y, hist_r, futr_r)
    for L in LEADS:
        print(f"    {L}h: RMSE={r_praw[L][0]:.3f}  <1C={r_praw[L][1]:.0f}%  N={r_praw[L][2]}")

    # --- +P_smooth3h ---
    print("\n>>> +P_smooth3h (per-request Pers traj, y_src = 3h smooth)")
    nf_sm, hist_s, futr_s = train_with_persist(grid, y_sm, "persist_sm3h")
    r_psm = eval_with_persist(nf_sm, grid, tw_test, y_sm, hist_s, futr_s)
    for L in LEADS:
        print(f"    {L}h: RMSE={r_psm[L][0]:.3f}  <1C={r_psm[L][1]:.0f}%  N={r_psm[L][2]}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY  (RMSE; Δ% vs base, negative = better)")
    print("=" * 72)
    print(f"{'variant':>16}  {'3h':>12}  {'6h':>12}  {'12h':>12}")
    for name, r in [("base", r_base), ("+P_raw", r_praw), ("+P_sm3h", r_psm)]:
        parts = []
        for L in LEADS:
            rmse = r[L][0]
            d = 100 * (rmse - r_base[L][0]) / r_base[L][0]
            parts.append(f"{rmse:.3f}({d:+.0f}%)")
        print(f"{name:>16}  {'  '.join(parts)}")

    print("\nDecision gate (3h, >7% improvement over base):")
    base3 = r_base[3.0][0]
    for name, r in [("+P_raw", r_praw), ("+P_sm3h", r_psm)]:
        imp = 100 * (base3 - r[3.0][0]) / base3
        verdict = "PASS -> consider linear-as-exog" if imp > 7 else "below 7% gate"
        print(f"  {name}: {imp:+.1f}%  -> {verdict}")


if __name__ == "__main__":
    main()
