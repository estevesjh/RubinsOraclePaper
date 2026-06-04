"""Derivative-aware NBEATSx loss: does penalizing trajectory SLOPE help?

The Stage-1 NBEATSx predicts a full 26-step forward trajectory but is trained
(HuberLoss) and read only at the single twilight point. The paper's thermal
control (Sec 5, Eq 7-8) consumes the trajectory's derivative dT/dt:
    T_setpoint(t) = Tbar_pred(t) + 0.5*tau*dT/dt - 0.3 degC
so slope fidelity matters operationally but is never optimized.

This sweeps lambda in the derivative-aware loss
    L = Huber(T_hat - T) + lambda * Huber( diff_H(T_hat) - diff_H(T) )
(traj_loss.TrajHuberLoss) and reports, per lambda:
  (a) twilight-POINT RMSE per lead   (regression gate: 3h must stay ~0.80)
  (b) trajectory-SLOPE RMSE          (new metric; should drop as lambda rises)

The full two-stage pipeline (NBEATSx + lead-gated Ridge) is reused from
run_final_model so the point metric is apples-to-apples with the paper.
Strict pre-2025 train / 2025 test.

Slope actual uses RAW grid["y"] (the operational trajectory), NOT the
3h-Gaussian-smoothed twilight actual used for the single point target.

Usage:
    uv run --no-sync python forecast/exp_traj_loss.py
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
import exp_humidity as eh
import run_final_model as rfm
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON, SOLAR_GRID_FREQ,
                    TEST_START_DATE, NBEATS_MAX_STEPS, LEAD_TIMES_HOURS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
NB_HIST = rfm.NB_HIST
NB_FUTR = rfm.NB_FUTR
LAMBDAS = [0.0, 0.25, 0.5, 1.0]


def train_nbeats(grid, lam):
    """Identical to rfm.train_nbeats but with the derivative-aware loss (lam)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from traj_loss import TrajHuberLoss

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds", "y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    print(f"  NBEATSx: {len(nf_train)} pre-2025 rows, lam={lam}", flush=True)
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=TrajHuberLoss(lam=lam), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[WIDTH, WIDTH]], n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10, val_check_steps=50,
        logger=False, enable_checkpointing=False,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train) * 0.1))
    return nf, hist, futr


def point_rmse_by_lead(rows, leads=(0.5, 1.0, 3.0, 6.0, 9.0, 12.0)):
    """Per-lead twilight-point RMSE from the two-stage Ridge result rows."""
    df = pd.DataFrame(rows)
    out = {}
    for L in leads:
        e = df.loc[df["lead_time_hours"] == L, "error"].to_numpy()
        out[L] = float(np.sqrt((e ** 2).mean())) if len(e) else np.nan
    return out


def traj_slope_rmse(grid, preds):
    """RMSE of predicted step-to-step dT/dt vs actual, over the forward window.

    yhat_traj covers grid steps [iss : iss+H]; the actual trajectory is the raw
    grid["y"] over the same window. Slope = first difference along the horizon.
    """
    y = grid["y"].to_numpy()
    sq = 0.0
    n = 0
    for (_, _), p in preds.items():
        yhat = np.asarray(p["yhat_traj"], dtype=float)
        m = len(yhat)
        ytrue = y[p["iss"]:p["iss"] + m]
        if len(ytrue) < m or np.any(np.isnan(ytrue)):
            continue
        d = np.diff(ytrue) - np.diff(yhat)
        sq += float((d ** 2).sum())
        n += len(d)
    return float(np.sqrt(sq / n)) if n else float("nan")


def night_traj_rmse(nf, grid, hist, futr):
    """Whole-night NBEATSx-ONLY trajectory RMSE (no Ridge second stage).

    For each 2025 solar night we issue ONE forecast at sunset (solar phase
    phi=0.5), predict the full NBEATS_HORIZON, and score the predicted T against
    the actual grid["y"] at EVERY night step (phi in [0.5, 1.0), i.e. sunset ->
    next sunrise). Residuals are pooled over all night steps of all 2025 nights.
    This tests how well NBEATSx alone reconstructs the temperature THROUGHOUT the
    night, not just at the twilight point.

    Errors are accumulated BOTH pooled and PER HORIZON STEP k (= k*0.5 h ahead of
    the sunset issue, since 1 solar-grid step = 30 min), so we can report RMSE/MAE
    vs horizon hours.

    Returns (rmse, mae, n_steps, n_nights, by_horizon) where by_horizon maps
    horizon_hours -> (rmse, mae, n) for each step k within the night window.
    """
    allx = hist + futr
    y = grid["y"].to_numpy()
    phi = grid["SolarTime"].to_numpy()
    day = grid["DayCount"].to_numpy()
    ds_real = pd.to_datetime(grid["ds_real"]).to_numpy()

    # Issue index per 2025 night = first grid row with phi>=0.5 in that DayCount.
    test_mask = ds_real >= np.datetime64(TEST_START_DATE)
    issues = []  # (iss, day_id)
    for d in np.unique(day[test_mask]):
        rows = np.where((day == d) & (phi >= 0.5))[0]
        if len(rows):
            issues.append((int(rows[0]), int(d)))

    BATCH = 64
    sq = 0.0; abserr = 0.0; n = 0; n_nights = 0
    # per-horizon-step accumulators (index k = step ahead of issue)
    h_sq = np.zeros(NBEATS_HORIZON); h_abs = np.zeros(NBEATS_HORIZON)
    h_n = np.zeros(NBEATS_HORIZON, dtype=int)
    for b0 in range(0, len(issues), BATCH):
        batch = issues[b0:b0 + BATCH]
        all_h, all_f, meta = [], [], []
        for (iss, d) in batch:
            hs = iss - NBEATS_INPUT_SIZE; fe = iss + NBEATS_HORIZON
            if hs < 0 or fe > len(grid):
                continue
            hd = grid.iloc[hs:iss][["ds", "y"] + allx].copy()
            if hd[allx].isna().any().any():
                continue
            fd = grid.iloc[iss:fe][["ds"] + futr].copy()
            if fd[futr].isna().any().any():
                continue
            uid = f"n_{b0}_{d}"
            hd["unique_id"] = uid; fd["unique_id"] = uid
            all_h.append(hd); all_f.append(fd)
            meta.append((uid, iss, d))
        if not all_h:
            continue
        with run.suppress_stdout():
            fc = nf.predict(pd.concat(all_h, ignore_index=True),
                            futr_df=pd.concat(all_f, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, iss, d) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            yhat = u[mc].to_numpy()
            counted = 0
            for k in range(len(yhat)):
                gi = iss + k
                if gi >= len(grid):
                    break
                # night step: same solar day, phi in [0.5, 1.0)
                if day[gi] == d and 0.5 <= phi[gi] < 1.0 and not np.isnan(y[gi]):
                    e = yhat[k] - y[gi]
                    sq += float(e * e); abserr += abs(float(e)); n += 1; counted += 1
                    h_sq[k] += float(e * e); h_abs[k] += abs(float(e)); h_n[k] += 1
            if counted:
                n_nights += 1
    rmse = float(np.sqrt(sq / n)) if n else float("nan")
    mae = float(abserr / n) if n else float("nan")
    by_horizon = {}
    for k in range(NBEATS_HORIZON):
        if h_n[k]:
            by_horizon[round(k * 0.5, 1)] = (
                float(np.sqrt(h_sq[k] / h_n[k])), float(h_abs[k] / h_n[k]), int(h_n[k]))
    return rmse, mae, n, n_nights, by_horizon


# Same 10 solar-grid features as the paper's Linear baseline (build_final.py).
LINEAR_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]


def night_traj_linear(grid):
    """Whole-night LINEAR baseline (per-horizon-step Ridge), no NBEATSx.

    The paper's Linear baseline fits a per-LEAD Ridge on solar-grid features
    (LINEAR_FEATS) read at the issue row, predicting the absolute target temp.
    Here we apply the SAME idea per NIGHT-HORIZON STEP: for each step k ahead of
    the sunset issue (phi=0.5), fit a Ridge mapping the issue-row features -> y at
    issue+k, then score over the same night window as night_traj_rmse.

    SPLIT: EVEN/ODD nights (train on even DayCount, test on odd) across ALL data,
    per the user's request. This interleaves every season into both folds and so
    removes the temporal pre-2025/2025 split artifact (cf. the April-bias note in
    project memory). NOTE: this is a DIFFERENT, linear-favorable protocol than the
    NBEATSx night metric (strict pre-2025 train / 2025 test), so treat the
    comparison as indicative of linear's best case, not a strict head-to-head.

    Returns (rmse, mae, n_steps, n_nights, by_horizon).
    """
    feat_cols = [c for c in LINEAR_FEATS if c in grid.columns]
    Xall = grid[feat_cols].to_numpy()
    y = grid["y"].to_numpy()
    phi = grid["SolarTime"].to_numpy()
    day = grid["DayCount"].to_numpy()

    # Sunset issue index per solar night; even DayCount -> train, odd -> test.
    issues = []  # (iss, day_id, is_test)
    for d in np.unique(day):
        rows = np.where((day == d) & (phi >= 0.5))[0]
        if len(rows):
            iss = int(rows[0])
            issues.append((iss, int(d), bool(int(d) % 2 == 1)))

    h_sq = np.zeros(NBEATS_HORIZON); h_abs = np.zeros(NBEATS_HORIZON)
    h_n = np.zeros(NBEATS_HORIZON, dtype=int)
    sq = abserr = 0.0; n = 0
    nights_hit = set()
    for k in range(NBEATS_HORIZON):
        Xtr, ytr, Xte, yte, te_days = [], [], [], [], []
        for (iss, d, is_test) in issues:
            gi = iss + k
            if gi >= len(grid):
                continue
            # only score genuine night steps (same day, phi in [0.5,1))
            if not (day[gi] == d and 0.5 <= phi[gi] < 1.0):
                continue
            xf = Xall[iss]
            if np.any(np.isnan(xf)) or np.isnan(y[gi]):
                continue
            if is_test:
                Xte.append(xf); yte.append(y[gi]); te_days.append(d)
            else:
                Xtr.append(xf); ytr.append(y[gi])
        if len(Xtr) < 20 or len(Xte) < 5:
            continue
        Xtr = np.array(Xtr); ytr = np.array(ytr)
        Xte = np.array(Xte); yte = np.array(yte)
        scaler = StandardScaler()
        ridge = SkRidge(alpha=1.0)
        ridge.fit(scaler.fit_transform(Xtr), ytr)
        pred = ridge.predict(scaler.transform(Xte))
        e = pred - yte
        h_sq[k] = float((e ** 2).sum()); h_abs[k] = float(np.abs(e).sum()); h_n[k] = len(e)
        sq += h_sq[k]; abserr += h_abs[k]; n += len(e)
        nights_hit.update(te_days)
    rmse = float(np.sqrt(sq / n)) if n else float("nan")
    mae = float(abserr / n) if n else float("nan")
    by_horizon = {}
    for k in range(NBEATS_HORIZON):
        if h_n[k]:
            by_horizon[round(k * 0.5, 1)] = (
                float(np.sqrt(h_sq[k] / h_n[k])), float(h_abs[k] / h_n[k]), int(h_n[k]))
    return rmse, mae, n, len(nights_hit), by_horizon


def main():
    print("=" * 88)
    print("DERIVATIVE-AWARE NBEATSx LOSS: lambda sweep (point gate + slope metric)")
    print("=" * 88)
    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    grid = eh.merge_humidity(grid)   # long-lead Ridge features (lead-gated in rfm)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}", flush=True)

    # Whole-night LINEAR baseline (per-horizon-step Ridge, no NBEATSx). lam-free.
    print("\nLinear baseline (whole-night, per-horizon Ridge)...", flush=True)
    lin_rmse, lin_mae, lin_steps, lin_nights, lin_by_h = night_traj_linear(grid)
    print(f"  whole-night Linear: RMSE={lin_rmse:.3f} MAE={lin_mae:.3f} "
          f"({lin_steps} steps over {lin_nights} nights)", flush=True)

    results = []
    for lam in LAMBDAS:
        print(f"\n--- lambda = {lam} ---", flush=True)
        nf, hist, futr = train_nbeats(grid, lam)
        print("  predicting (train)...", flush=True)
        tr_preds = rfm.get_all_preds(nf, grid, tw_train, hist, futr, LEAD_TIMES_HOURS)
        print("  predicting (test)...", flush=True)
        te_preds = rfm.get_all_preds(nf, grid, tw_test, hist, futr, LEAD_TIMES_HOURS)
        rows = rfm.train_and_apply_ridge(grid, tr_preds, te_preds, LEAD_TIMES_HOURS)
        pr = point_rmse_by_lead(rows)
        slope = traj_slope_rmse(grid, te_preds)
        night_rmse, night_mae, n_steps, n_nights, by_h = night_traj_rmse(nf, grid, hist, futr)
        results.append((lam, pr, slope, night_rmse, night_mae, by_h))
        print(f"  3h point RMSE = {pr[3.0]:.3f}   slope RMSE = {slope:.4f}", flush=True)
        print(f"  whole-night NBEATSx-only: RMSE={night_rmse:.3f} MAE={night_mae:.3f} "
              f"({n_steps} steps over {n_nights} nights)", flush=True)

    print(f"\n{'='*88}")
    print(f"  {'lam':>5} | {'0.5h':>6} {'1h':>6} {'3h':>6} {'6h':>6} {'9h':>6} {'12h':>6} "
          f"| {'slope':>7} | {'nightRMSE':>9} {'nightMAE':>8}")
    print("-" * 88)
    for lam, pr, slope, night_rmse, night_mae, by_h in results:
        def g(L):
            v = pr.get(L, np.nan)
            return f"{v:6.3f}" if v == v else "   -  "
        print(f"  {lam:>5} | {g(0.5)} {g(1.0)} {g(3.0)} {g(6.0)} {g(9.0)} {g(12.0)} "
              f"| {slope:7.4f} | {night_rmse:9.3f} {night_mae:8.3f}")
    print("=" * 88)
    print("Point/slope are twilight-target (NBEATSx+Ridge); nightRMSE/MAE are")
    print("NBEATSx-ONLY over ALL night steps (phi in [0.5,1)), one sunset-issued")
    print("forecast per 2025 night. Gate: largest lambda with 3h ~flat (+/-0.02),")
    print("lowest slope, and night RMSE not regressed.")

    # Whole-night error vs HORIZON HOURS ahead of the sunset issue. NBEATSx-only
    # (per lambda) vs the Linear baseline. Both scored on the IDENTICAL 2025 night
    # window, no Ridge second stage.
    HORIZONS = [1.0, 3.0, 5.0, 6.0, 9.0]

    def cells_for(by_h, idx):
        out = []
        for h in HORIZONS:
            v = by_h.get(round(h, 1))
            out.append(f"{v[idx]:6.3f}" if v else "   -  ")
        return " ".join(out)

    print(f"\n{'='*88}")
    print("WHOLE-NIGHT error vs HORIZON HOURS (ahead of sunset issue), no Ridge")
    print("  Linear: even/odd night split (all seasons in both folds)")
    print("  NBEATSx: strict pre-2025 train / 2025 test")
    print(f"{'='*88}")
    hdr = "  " + " ".join(f"{h:>5.0f}h" for h in HORIZONS)
    print(f"  {'model':>14} | RMSE {hdr}")
    print("-" * 88)
    print(f"  {'Linear':>14} |      {cells_for(lin_by_h, 0)}")
    for lam, pr, slope, night_rmse, night_mae, by_h in results:
        print(f"  {'NBEATSx lam='+str(lam):>14} |      {cells_for(by_h, 0)}")
    print("-" * 88)
    print(f"  {'Linear':>14} | MAE  {cells_for(lin_by_h, 1)}")
    for lam, pr, slope, night_rmse, night_mae, by_h in results:
        print(f"  {'NBEATSx lam='+str(lam):>14} | MAE  {cells_for(by_h, 1)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
