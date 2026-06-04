"""CLOCK-TIME ablation of the paper model: direct-T NBEATSx + Ridge, NO solar grid.

This is the fair counterpart to forecast/run_final_model.py. It keeps the paper
architecture IDENTICAL and changes ONLY the time grid:

  run_final_model.py : resample onto a uniform SOLAR-time grid (48 steps/solar-day,
                       sunrise/sunset-anchored); deterministic exog = solar_sin/cos.
  run_nosolar.py     : resample onto a fixed CLOCK-time grid (48 steps/24h = 30-min
                       UTC bins); deterministic exog = hour_sin/cos.

Everything else matches run_final_model.py:
  - direct-T NBEATSx, width=16, 700 steps, identity scaler, HuberLoss, SELU,
    stacks [trend, seasonality, identity, exogenous], n_blocks=[1,1,1,1]
  - hist_exog = [y_raw, y_lag_24, trend_2h, y_diff_24h]
  - 2-stage Ridge correction (13-feature base set, clock analogs) + T_nb
  - strict pre-2025 train / 2025 test, twilight target = alt_sun crossing -20 deg

Because step k means the SAME elapsed clock time on both grids (30 min), every
lag/window has a matched physical meaning, so a difference in RMSE isolates the
grid choice. The one feature without a clean clock analog (velocity_noon, defined
at solar noon) is dropped from BOTH the table note and this run (12-feature base).

Usage:
    uv run --no-sync python forecast/analysis/run_nosolar.py
"""

import os, sys, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning", "lightning.pytorch", "lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

# forecast/ on path so we can reuse config + run.suppress_stdout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PATH, RESULTS_PATH, TEST_START_DATE,
                    NBEATS_INPUT_SIZE, NBEATS_HORIZON, NBEATS_MAX_STEPS,
                    LEAD_TIMES_HOURS, STEPS_PER_DAY)

# ── Clock-grid config (mirrors the solar grid's step COUNT, not its anchoring) ──
CLOCK_GRID_FREQ = "1800s"     # 30-min clock bins => 48 steps / 24h, same as solar
STEPS_PER_CLOCK_DAY = 48
HALFDAY_LAG_STEPS = 24        # 12h anchor (matches solar HALFDAY_LAG_STEPS)
WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
SUN_ALT_TARGET = -20.0

# Same feature ROLES as run_final_model.py; clock analogs of the solar names.
NB_HIST = ["y_raw", "y_lag_24", "trend_2h", "y_diff_24h"]
NB_FUTR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
# Base Ridge set (paper's 13 minus solar-noon-specific velocity_noon -> 12):
RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_2h", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "trend_4h", "dmean_1d",
]


@contextmanager
def suppress_stdout():
    so, se = sys.stdout.fileno(), sys.stderr.fileno()
    saved_o, saved_e = os.dup(so), os.dup(se)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, so); os.dup2(dn, se); os.close(dn)
    try:
        yield
    finally:
        os.dup2(saved_o, so); os.dup2(saved_e, se)
        os.close(saved_o); os.close(saved_e)


def _backward_ols_slope(y_arr, window):
    """Causal backward OLS slope over `window` steps (mirrors run.py)."""
    slopes = np.full(len(y_arr), np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()
    for i in range(window, len(y_arr)):
        seg = y_arr[i - window:i]
        if np.isnan(seg).any():
            continue
        slopes[i] = ((x - x_mean) * (seg - seg.mean())).sum() / var_x
    return slopes


def load_and_prepare_clock():
    """Load raw 15-min data, clean, resample to a FIXED 30-min CLOCK grid, build
    the same features as run_final_model (clock analogs). No solar resampling."""
    print("Loading data...")
    raw = pd.read_csv(DATA_PATH, comment="#", low_memory=False)
    raw["ds"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None)
    raw["y"] = raw["mean"]
    spread = raw["max"] - raw["min"]

    # Same high-spread filter + causal fill as run.py.
    SPREAD_THRESHOLD = 3.0
    bad = spread > SPREAD_THRESHOLD
    print(f"  Filtering {bad.sum()} high-spread points "
          f"({100*bad.sum()/len(raw):.1f}%, threshold={SPREAD_THRESHOLD}C)")
    raw.loc[bad, "y"] = np.nan
    raw["y"] = raw["y"].ffill().bfill()
    raw = raw[["ds", "y", "alt_sun"]].dropna(subset=["ds"]).sort_values("ds")

    # FIXED clock grid: 30-min bins. Mean-aggregate y; linear-interpolate alt_sun
    # (deterministic ephemeris, so interpolation is exact, not leakage).
    print("Resampling to FIXED 30-min CLOCK grid (48 steps/24h)...")
    raw = raw.set_index("ds")
    grid = pd.DataFrame()
    grid["y"] = raw["y"].resample(CLOCK_GRID_FREQ).mean()
    grid["alt_sun"] = raw["alt_sun"].resample(CLOCK_GRID_FREQ).mean()
    grid = grid.reset_index().rename(columns={"ds": "ds_real"})
    # Causal fill of any empty bins (small gaps only).
    grid["y"] = grid["y"].ffill()
    grid["alt_sun"] = grid["alt_sun"].interpolate(limit_direction="both")
    grid["ds"] = grid["ds_real"]  # NBEATSx needs a 'ds'; here clock == real time
    print(f"  Grid rows: {len(grid)}, range {grid['ds_real'].min()} -> {grid['ds_real'].max()}")

    y_arr = grid["y"].values

    # Lag features (CLOCK lags; step k = 30 min).
    grid["y_raw"] = grid["y"].copy()
    grid["y_lag_6"] = grid["y"].shift(6)    # 3h
    grid["y_lag_12"] = grid["y"].shift(12)  # 6h
    grid["y_lag_24"] = grid["y"].shift(HALFDAY_LAG_STEPS)  # 12h anchor
    grid["y_lag_48"] = grid["y"].shift(48)  # 24h
    # 24h difference T[t]-T[t-48] (matches run.py y_diff_24h).
    grid["y_diff_24h"] = y_arr - np.roll(y_arr, 48)
    grid.iloc[:48, grid.columns.get_loc("y_diff_24h")] = np.nan

    # Backward OLS slopes: 2h (4 steps) and 4h (8 steps).
    grid["trend_2h"] = _backward_ols_slope(y_arr, 4)
    grid["trend_4h"] = _backward_ols_slope(y_arr, 8)

    # 24h-mean drift (dmean_1d): rolling 48 = 24h, same window as solar.
    mean_24h = pd.Series(y_arr).rolling(48, min_periods=24).mean()
    grid["dmean_1d"] = mean_24h.values - mean_24h.shift(48).values

    # Deterministic time features: CLOCK hour (replaces solar phase) + day-of-year.
    hour_frac = grid["ds_real"].dt.hour + grid["ds_real"].dt.minute / 60.0
    grid["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    grid["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    doy = grid["ds_real"].dt.dayofyear
    grid["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    grid["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Drop warmup (same rule as run.py).
    n_warmup = NBEATS_INPUT_SIZE + HALFDAY_LAG_STEPS
    grid = grid.iloc[n_warmup:].reset_index(drop=True)
    print(f"  After warmup: {len(grid)} rows")
    return grid


def find_twilight_targets_clock(grid):
    """Twilight events = alt_sun crossing -20 deg (setting), interpolated.
    Uses a 3h-Gaussian-smoothed y for the validation actual (matches run.py)."""
    alt = grid["alt_sun"].values
    y_arr = grid["y"].values
    y_smooth = pd.Series(y_arr).rolling(
        6, min_periods=3, center=True, win_type="gaussian").mean(std=2).values
    adjusted = alt - SUN_ALT_TARGET
    crossings = np.where((adjusted[:-1] >= 0) & (adjusted[1:] < 0))[0]
    events = []
    for i in crossings:
        a0, a1 = adjusted[i], adjusted[i + 1]
        frac = a0 / (a0 - a1)
        y_interp = y_smooth[i] + frac * (y_smooth[i + 1] - y_smooth[i])
        events.append({"grid_idx": i, "ds_real": grid["ds_real"].iloc[i],
                       "y_actual": y_interp})
    return pd.DataFrame(events)


def lead_hours_to_steps(lead_hours):
    """Clock lead hours -> clock-grid step offset (48 steps / 24h)."""
    return round(lead_hours * STEPS_PER_CLOCK_DAY / 24.0)


def train_nbeats(grid):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds", "y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    print(f"  NBEATSx (clock): {len(nf_train)} pre-2025 rows, width={WIDTH}, steps={MAX_STEPS}")
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[WIDTH, WIDTH]], n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10, val_check_steps=50,
        logger=False, enable_checkpointing=False,
    )
    with suppress_stdout():
        nf = NeuralForecast(models=[model], freq=CLOCK_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train) * 0.1))
    return nf, hist, futr


def get_all_preds(nf, grid, tw_set, hist, futr, leads):
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in leads:
            iss = tgt - lead_hours_to_steps(L)
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0 + BATCH]
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
        with suppress_stdout():
            fc = nf.predict(pd.concat(all_h, ignore_index=True),
                            futr_df=pd.concat(all_f, ignore_index=True)).reset_index()
        mc = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]
        for (uid, tgt, iss, L, yact, ev_i, ds_r) in meta:
            u = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
            ps = tgt - iss
            if len(u) == 0 or ps >= len(u):
                continue
            preds[(ev_i, L)] = {"T_nb": float(u[mc].iloc[ps]),
                                "y_actual": yact, "iss": iss, "ds_real": ds_r}
    return preds


def train_and_apply_ridge(grid, train_preds, test_preds, leads):
    """Per-lead Ridge on [base features, T_nb]. Mirrors run_final_model."""
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    rows = []
    for L in leads:
        X_tr, y_tr = [], []
        for (ev_i, lead), p in train_preds.items():
            if lead != L:
                continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            X_tr.append(np.concatenate([feats, [p["T_nb"]]]))
            y_tr.append(p["y_actual"])
        if len(X_tr) < 30:
            continue
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        scaler = StandardScaler()
        ridge = SkRidge(alpha=1.0)
        ridge.fit(scaler.fit_transform(X_tr), y_tr)
        for (ev_i, lead), p in test_preds.items():
            if lead != L:
                continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            row_x = np.concatenate([feats, [p["T_nb"]]])
            T_pred = ridge.predict(scaler.transform(row_x.reshape(1, -1)))[0]
            rows.append({
                "twilight_time": pd.Timestamp(p["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "forecast_time": pd.Timestamp(p["ds_real"]).strftime("%Y-%m-%d %H:%M:%S.000"),
                "lead_time_hours": L, "actual_temp": p["y_actual"],
                "model": "NBEATSx-NoSolar", "forecast_temp": T_pred,
                "error": p["y_actual"] - T_pred,
            })
    return rows


def main():
    print("=" * 72)
    print("CLOCK-TIME ABLATION: direct-T NBEATSx + Ridge (NO solar grid)")
    print("=" * 72)

    grid = load_and_prepare_clock()
    tw = find_twilight_targets_clock(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}")

    print("\nTraining NBEATSx (clock grid)...")
    nf, hist, futr = train_nbeats(grid)

    print("\nNBEATSx predictions (train)...")
    train_preds = get_all_preds(nf, grid, tw_train, hist, futr, LEAD_TIMES_HOURS)
    print(f"  {len(train_preds)} preds")
    print("NBEATSx predictions (test)...")
    test_preds = get_all_preds(nf, grid, tw_test, hist, futr, LEAD_TIMES_HOURS)
    print(f"  {len(test_preds)} preds")

    print("\nTraining + applying Ridge...")
    result_rows = train_and_apply_ridge(grid, train_preds, test_preds, LEAD_TIMES_HOURS)
    results = pd.DataFrame(result_rows)
    print(f"  {len(results)} result rows")

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    out = RESULTS_PATH / "paper_results_nosolar.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print(f"\n{'lead':>6} {'RMSE':>7} {'<1C':>5} {'N':>4}")
    print("-" * 25)
    for L in [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]:
        s = results[results["lead_time_hours"] == L]
        if len(s) == 0:
            continue
        e = s["error"].values
        print(f"{L:>6} {np.sqrt((e**2).mean()):>7.3f} {(np.abs(e)<1).mean()*100:>4.0f}% {len(s):>4}")


if __name__ == "__main__":
    main()
