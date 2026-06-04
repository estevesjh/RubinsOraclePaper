"""Phase 3b: do humidity & wind (causal, issue-time) reduce winter/spring RMSE?

Diagnosis (docs/winter_spring_plan.md §2e): humidity carries independent winter
signal (partial r +0.18 vs |error|, controlling for synoptic |dT_day|); wind is
the stronger spring channel (+0.16). data/weather_wind_humidity.csv covers
2021->2026 so ALL seasons are testable (unlike the MB NWP file, which ends
2025-08-31).

Design: train the base NBEATSx ONCE (input_size=48, paper hist/futr), get T_nb
for every twilight/lead, then evaluate Ridge variants in sklearn (instant):
  BASE             paper 13 features
  +trend           Phase-1 winner: +y_diff_45h, y_diff_30h, last_std_24h, DTR_3d
  +hw              + humidity, windSpeed (causal, at issue time)
  +trend+hw        both
Per-season RMSE at 3h and 9h. Strict pre-2025 train / 2025 test.

Causality: humidity/wind are merged onto the grid by timestamp; the Ridge
feature is read at the ISSUE row (L hours before twilight), so it only uses
information available at forecast issuance.

Usage:
    uv run --no-sync python forecast/exp_humidity.py
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
from pathlib import Path
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import run
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON, RESULTS_PATH,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS,
                    LEAD_TIMES_HOURS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
RIDGE_BASE = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
]
TREND_FEATS = ["y_diff_45h", "y_diff_30h", "last_std_24h", "DTR_3d"]
HW_FEATS = ["humidity", "windSpeed", "hum_std_24h", "wind_std_24h"]
# Directional wind (user's idea): decompose |w| relative to the summit's
# prevailing axis. anchor learned pre-2025 (speed-weighted vector mean of the
# dominant NW lobe). u_par = along-axis flow, u_perp = cross-axis (front marker).
WDIR_FEATS = ["wind_u_par", "wind_u_perp", "wind_u_par_NE"]

HUM_FILE = Path(__file__).parent.parent / "data" / "weather_wind_humidity.csv"


def season_of(ts):
    m = pd.Timestamp(ts).month
    return {12: "Summer", 1: "Summer", 2: "Summer", 3: "Fall", 4: "Fall", 5: "Fall",
            6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring",
            11: "Spring"}[m]


def merge_humidity(grid):
    """Add humidity/windSpeed columns to the grid, aligned by nearest timestamp.

    Causal at the issue row: the value is the observation nearest the grid row's
    real time (within 30 min). Rolling 24h std added as a volatility proxy.
    """
    h = pd.read_csv(HUM_FILE, low_memory=False)
    h["ts"] = pd.to_datetime(h["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    h = h.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    hts = h["ts"].values.astype("datetime64[ns]")
    g_ts = pd.to_datetime(grid["ds_real"].values).values.astype("datetime64[ns]")

    def align(col):
        src = h[col].values
        out = np.full(len(g_ts), np.nan)
        idx = np.searchsorted(hts, g_ts)
        for k in range(len(g_ts)):
            cands = [j for j in (idx[k] - 1, idx[k]) if 0 <= j < len(hts)]
            if not cands:
                continue
            j = min(cands, key=lambda c: abs((hts[c] - g_ts[k]) / np.timedelta64(1, "s")))
            if abs((hts[j] - g_ts[k]) / np.timedelta64(1, "s")) <= 1800:
                out[k] = src[j]
        return out

    grid["humidity_raw"] = align("humidity")
    grid["windSpeed_raw"] = align("windSpeed")
    grid["windDir_raw"] = align("windDir")
    grid["hum_avail"] = grid["humidity_raw"].notna() & grid["windSpeed_raw"].notna()
    raw_cov = grid["hum_avail"].mean()

    # ── Directional wind decomposition (user's idea) ───────────────────────
    # phi = windDir - prevailing_anchor; u_par = |w|cos(phi), u_perp = |w|sin(phi).
    # Prevailing anchor = speed-weighted vector mean of the dominant NW lobe,
    # learned from PRE-2025 only (causal). A second NE anchor captures the spring
    # along-flow channel that correlated best in diagnostics (r=+0.31).
    train = pd.to_datetime(grid["ds_real"]).values < np.datetime64(TEST_START_DATE)
    wd_tr = grid["windDir_raw"].values[train]
    ws_tr = grid["windSpeed_raw"].values[train]
    good = ~np.isnan(wd_tr) & ~np.isnan(ws_tr)
    # restrict to the NW/N lobe (270-360 + 0-90) before vector-averaging so the
    # bimodal S lobe doesn't drag the anchor into the empty gap
    wdg = wd_tr[good]; wsg = ws_tr[good]
    lobe = (wdg >= 270) | (wdg <= 90)
    r = np.deg2rad(wdg[lobe])
    anchor = np.rad2deg(np.arctan2(np.sum(wsg[lobe] * np.sin(r)),
                                   np.sum(wsg[lobe] * np.cos(r)))) % 360
    ANCHOR_NE = 45.0
    ws_all = grid["windSpeed_raw"].values
    wd_all = grid["windDir_raw"].values
    phi = np.deg2rad(wd_all - anchor)
    grid["wind_u_par_raw"] = ws_all * np.cos(phi)
    grid["wind_u_perp_raw"] = np.abs(ws_all * np.sin(phi))
    phi_ne = np.deg2rad(wd_all - ANCHOR_NE)
    grid["wind_u_par_NE_raw"] = ws_all * np.cos(phi_ne)
    print(f"  wind prevailing anchor (pre-2025, NW lobe, speed-wtd): {anchor:.0f}°")

    # ── Wind TENDENCY / trend-lag (user's idea): a front shows up as wind
    # CHANGING before/with the temperature swing. Computed on the raw wind series
    # (so the tendency is genuine where data exists) and imputed below.
    #   dws_6h   : 6h windspeed change (strong SPRING error correlate, r=-0.42)
    #   wdir_var_24h : 24h directional variability (strong WINTER correlate, r=-0.34)
    # Direction differences are circular; variability uses rolling std of sin/cos.
    n_g = len(grid)
    grid["dws_6h_raw"] = ws_all - np.roll(ws_all, 12)
    grid.iloc[:12, grid.columns.get_loc("dws_6h_raw")] = np.nan
    sin_wd = np.sin(np.deg2rad(wd_all)); cos_wd = np.cos(np.deg2rad(wd_all))
    grid["wdir_var_24h_raw"] = (
        pd.Series(sin_wd).rolling(48, min_periods=24).std().values
        + pd.Series(cos_wd).rolling(48, min_periods=24).std().values
    )

    # ── Impute gaps so coverage does not confound the variant comparison. ──
    # Gaps are filled with a per-solar-time-bin climatology computed from PRE-2025
    # rows only (causal). This makes humidity/wind defined everywhere; on missing
    # days the feature reverts to the seasonal-diurnal mean (zero added info, but
    # no row dropping and no leakage).
    st = grid["SolarTime"].values
    bin_idx = (np.round(st * 48).astype(int)) % 48
    doy = pd.to_datetime(grid["ds_real"]).dt.dayofyear.values
    # month bin captures the seasonal cycle of humidity
    mo = pd.to_datetime(grid["ds_real"]).dt.month.values
    train = pd.to_datetime(grid["ds_real"]).values < np.datetime64(TEST_START_DATE)
    for col in ["humidity", "windSpeed", "wind_u_par", "wind_u_perp", "wind_u_par_NE",
                "dws_6h", "wdir_var_24h"]:
        raw = grid[f"{col}_raw"].values.copy()
        filled = raw.copy()
        # climatology keyed by (month, solar-time bin), pre-2025
        clim = {}
        for m_ in range(1, 13):
            for b_ in range(48):
                sel = (mo == m_) & (bin_idx == b_) & train & ~np.isnan(raw)
                if sel.sum() >= 5:
                    clim[(m_, b_)] = float(np.nanmean(raw[sel]))
        global_mean = float(np.nanmean(raw[train & ~np.isnan(raw)]))
        nanrows = np.where(np.isnan(filled))[0]
        for k in nanrows:
            filled[k] = clim.get((mo[k], bin_idx[k]), global_mean)
        grid[col] = filled
    # 24h rolling std (48 steps) volatility proxy on the imputed series; causal.
    grid["hum_std_24h"] = pd.Series(grid["humidity"].values).rolling(48, min_periods=24).std().values
    grid["wind_std_24h"] = pd.Series(grid["windSpeed"].values).rolling(48, min_periods=24).std().values

    # ── RH terciles (user's idea): the spring humidity effect is NONLINEAR —
    # flat across low/mid RH, strong only in the high-RH bin. A single linear
    # coefficient can't capture that. Encode low/mid/high as two dummy columns
    # (mid, high; low = reference). Thresholds = pre-2025 RH terciles (causal).
    rh = grid["humidity"].values
    q33, q67 = np.nanpercentile(rh[train], [33.333, 66.667])
    grid["rh_mid"] = ((rh >= q33) & (rh < q67)).astype(float)
    grid["rh_high"] = (rh >= q67).astype(float)
    print(f"  humidity/wind merged: raw coverage={raw_cov:.2f}, imputed to 1.00 "
          f"(per month×solar-bin climatology, pre-2025)")
    print(f"  RH terciles (pre-2025): low<{q33:.0f}%  mid<{q67:.0f}%  high>={q67:.0f}%")
    return grid


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
    print(f"  NBEATSx: {len(nf_train)} pre-2025 rows, hist={hist}")
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
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
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
            ls = run.lead_hours_to_steps(L)
            iss = tgt - ls
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
        with run.suppress_stdout():
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


def ridge_eval(grid, train_preds, test_preds, leads, feat_cols):
    """Per-lead Ridge. With imputed humidity/wind no rows drop, so every variant
    is evaluated on the IDENTICAL twilight set. Also records hum_avail so we can
    additionally restrict to the humidity-observed subset downstream."""
    feat_cols = [c for c in feat_cols if c in grid.columns]
    avail = grid["hum_avail"].values if "hum_avail" in grid.columns else None
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
            rows.append({"lead": L, "error": p["y_actual"] - T_pred,
                         "season": season_of(p["ds_real"]),
                         "hum_avail": bool(avail[p["iss"]]) if avail is not None else True})
    return pd.DataFrame(rows)


def seasonal_rmse(df, leads=(3.0, 9.0)):
    out = {}
    for L in leads:
        d = df[np.abs(df["lead"] - L) < 0.01]
        out[(L, "ALL")] = float(np.sqrt((d["error"] ** 2).mean())) if len(d) else np.nan
        for s in ["Summer", "Fall", "Winter", "Spring"]:
            ds = d[d["season"] == s]
            out[(L, s)] = float(np.sqrt((ds["error"] ** 2).mean())) if len(ds) else np.nan
    return out


def fmt_row(tag, m):
    def g(L, s):
        v = m.get((L, s), np.nan)
        return f"{v:.3f}" if v == v else "  -  "
    obj = np.nanmean([m.get((9.0, "Winter"), np.nan), m.get((9.0, "Spring"), np.nan)])
    return (f"  {tag:18s} | 3h ALL={g(3.0,'ALL')} Wi={g(3.0,'Winter')} Sp={g(3.0,'Spring')} "
            f"| 9h ALL={g(9.0,'ALL')} Wi={g(9.0,'Winter')} Sp={g(9.0,'Spring')} | WS9={obj:.3f}")


def main():
    print("=" * 100)
    print("PHASE 3b: humidity & wind as causal Ridge features")
    print("=" * 100)
    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    grid = merge_humidity(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}", flush=True)

    nf, hist, futr = train_nbeats(grid)
    print("  predicting (train)...", flush=True)
    tr_preds = get_all_preds(nf, grid, tw_train, hist, futr, LEAD_TIMES_HOURS)
    print("  predicting (test)...", flush=True)
    te_preds = get_all_preds(nf, grid, tw_test, hist, futr, LEAD_TIMES_HOURS)
    print(f"  train preds={len(tr_preds)}, test preds={len(te_preds)}", flush=True)

    # HW with DIRECTIONAL wind (user's idea) instead of raw windSpeed
    HW_DIR = ["humidity", "hum_std_24h"] + WDIR_FEATS
    # RH TERCILE bins (user's idea): nonlinear humidity via low/mid/high dummies.
    RH_BINS = ["rh_mid", "rh_high"]
    HW_DIR_BINNED = RH_BINS + ["hum_std_24h"] + WDIR_FEATS  # binned RH replaces linear humidity
    WTEND = ["dws_6h", "wdir_var_24h"]                       # wind tendency (user's idea)
    variants = {
        "BASE": RIDGE_BASE,
        "+trend": RIDGE_BASE + TREND_FEATS,
        "+hw": RIDGE_BASE + HW_FEATS,
        "+trend+hw": RIDGE_BASE + TREND_FEATS + HW_FEATS,
        "+hw_dir": RIDGE_BASE + HW_DIR,
        "+trend+hw_dir": RIDGE_BASE + TREND_FEATS + HW_DIR,
        "+trend+rhbin": RIDGE_BASE + TREND_FEATS + RH_BINS,
        "+trend+hw_dir_rhbin": RIDGE_BASE + TREND_FEATS + HW_DIR_BINNED,  # PRODUCTION
        "+trend+hw_dir+rhbin": RIDGE_BASE + TREND_FEATS + HW_DIR + RH_BINS,
        "+trend+hw_dir_rhbin+wtend": RIDGE_BASE + TREND_FEATS + HW_DIR_BINNED + WTEND,
    }
    # Evaluate every variant once; humidity/wind imputed so the twilight set is
    # IDENTICAL across variants (fixes the coverage confound).
    evals = {name: ridge_eval(grid, tr_preds, te_preds, LEAD_TIMES_HOURS, feats)
             for name, feats in variants.items()}

    print(f"\n{'='*100}\nALL 2025 TWILIGHTS (imputed hum/wind; identical set across variants)")
    print(f"WS9 = mean winter & spring 9h RMSE\n{'='*100}")
    for name, df in evals.items():
        print(fmt_row(name, seasonal_rmse(df)), flush=True)

    print(f"\n{'='*100}\nHUMIDITY-OBSERVED SUBSET ONLY (hum_avail=True; honest hw comparison)")
    print(f"{'='*100}")
    for name, df in evals.items():
        sub = df[df["hum_avail"]]
        print(fmt_row(name, seasonal_rmse(sub)), flush=True)
    # report subset sizes so the reader knows the seasonal n
    sub9 = evals["BASE"][(np.abs(evals["BASE"]["lead"] - 9.0) < 0.01)]
    print("  subset n by season @9h (hum_avail):",
          sub9[sub9["hum_avail"]].groupby("season").size().to_dict(),
          "| full:", sub9.groupby("season").size().to_dict())


if __name__ == "__main__":
    main()
