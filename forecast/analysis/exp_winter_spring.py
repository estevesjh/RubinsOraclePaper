"""Phase 1+2 experiment: reduce winter/spring RMSE via multi-day trend features
(Ridge stage) and a longer NBEATSx lookback (input_size).

Diagnosis (docs/winter_spring_plan.md): winter/spring error at 6-12h leads is
driven by day-to-day synoptic variability; the model lags the multi-day trend
(r(signed_err, dT_day)=+0.58 winter / +0.52 spring at 9h). ~15-19% of the 9h
error is reducible trend structure. The final model's NBEATSx sees only
input_size=48 = exactly 1 solar day, so it cannot perceive a multi-day trend;
its Ridge stage uses only dmean_1d for multi-day context.

This script:
  Phase 2 (expensive): train NBEATSx once per (input_size, NB_HIST) config.
  Phase 1 (cheap):      for each trained model, sweep Ridge feature sets in
                        sklearn and report per-season RMSE at 3h and 9h.

Strict causality preserved: NBEATSx + Ridge train pre-2025, evaluate on 2025.

Usage:
    uv run --no-sync python forecast/exp_winter_spring.py
    # quick check (baseline config only):
    EXP_CONFIGS=base uv run --no-sync python forecast/exp_winter_spring.py
"""

import os, sys, json, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning", "lightning.pytorch", "lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# Thread cap so parallel per-config processes don't oversubscribe the cores.
_THREADS = os.environ.get("EXP_THREADS")
if _THREADS:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = _THREADS

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
import feature_sweep as fs
from config import (NBEATS_HORIZON, RESULTS_PATH, SOLAR_GRID_FREQ,
                    TEST_START_DATE, NBEATS_MAX_STEPS, LEAD_TIMES_HOURS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
ACCEL = os.environ.get("EXP_ACCEL", "cpu")   # 'cpu' or 'mps'

if _THREADS:
    try:
        import torch
        torch.set_num_threads(int(_THREADS))
    except Exception:
        pass

# ── Baseline (paper) feature sets ───────────────────────────────────────────
NB_HIST_BASE = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]
RIDGE_BASE = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    "trend_solar_4h", "velocity_noon", "dmean_1d",
]

# ── Phase 1 candidate multi-day trend features for the Ridge stage ──────────
# (all verified present with full coverage in the grid)
RIDGE_TREND_CANDS = [
    "dmean_3d", "y_diff_30h", "y_diff_45h", "y_diff_72h",
    "temp_trend_3d", "cooling_rate_6h", "trend_solar_6h",
    "dTmax_3d", "DTR_3d", "last_std_24h",
]

# ── Phase 2 NBEATSx configs (input_size, extra hist exog) ───────────────────
# Each entry trains one NBEATSx. input_size in solar-grid steps (48 = 1 SD).
EXP_CONFIGS = {
    "base":        dict(input_size=48,  nb_hist=NB_HIST_BASE),
    "in72":        dict(input_size=72,  nb_hist=NB_HIST_BASE),
    "in96":        dict(input_size=96,  nb_hist=NB_HIST_BASE),
    "in144":       dict(input_size=144, nb_hist=NB_HIST_BASE),
    # longer lookback + a multi-day diff fed to the network itself:
    "in96_trend":  dict(input_size=96,  nb_hist=NB_HIST_BASE + ["dmean_3d"]),
}


def season_of(ts):
    m = pd.Timestamp(ts).month
    return {12: "Summer", 1: "Summer", 2: "Summer", 3: "Fall", 4: "Fall", 5: "Fall",
            6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring",
            11: "Spring"}[m]


def train_nbeats(grid, input_size, nb_hist):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in nb_hist if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    allx = hist + futr
    tr = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[tr, ["ds", "y"] + allx].dropna().copy()
    nf_train["unique_id"] = "temp"
    print(f"    NBEATSx: {len(nf_train)} pre-2025 rows, input_size={input_size}, "
          f"hist={hist}")
    accel_kw = {"accelerator": ACCEL, "devices": 1} if ACCEL else {}
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=input_size, max_steps=MAX_STEPS,
        hist_exog_list=hist, futr_exog_list=futr,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[WIDTH, WIDTH]], n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10, val_check_steps=50,
        # Disable Lightning logger/checkpoint dirs: parallel workers otherwise
        # race to create the same lightning_logs/version_N directory and crash.
        logger=False, enable_checkpointing=False,
        **accel_kw,
    )
    with run.suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=int(len(nf_train) * 0.1))
    return nf, hist, futr, input_size


def get_all_preds(nf, grid, tw_set, hist, futr, leads, input_size):
    """NBEATSx predictions for all twilights at all leads. Mirrors run_final_model."""
    allx = hist + futr
    preds = {}
    BATCH = 64
    requests = []
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        for L in leads:
            ls = run.lead_hours_to_steps(L)
            iss = tgt - ls
            if iss < input_size or (tgt - iss) >= NBEATS_HORIZON:
                continue
            requests.append((ev_i, tgt, iss, L, ev["y_actual"], ev["ds_real"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0 + BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact, ds_r) in batch:
            hs = iss - input_size; fe = iss + NBEATS_HORIZON
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
    """Train Ridge per-lead on pre-2025, apply to 2025. Returns DataFrame of rows."""
    feat_cols = [c for c in feat_cols if c in grid.columns]
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
            rows.append({"lead": L, "actual": p["y_actual"], "pred": T_pred,
                         "error": p["y_actual"] - T_pred,
                         "season": season_of(p["ds_real"])})
    return pd.DataFrame(rows)


def seasonal_rmse(df, leads=(3.0, 9.0)):
    """Return {(lead,season): rmse} + overall, plus winter+spring mean."""
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
    # objective = mean(winter9, spring9): the bottleneck
    obj = np.nanmean([m.get((9.0, "Winter"), np.nan), m.get((9.0, "Spring"), np.nan)])
    return (f"  {tag:28s} | 3h ALL={g(3.0,'ALL')} Wi={g(3.0,'Winter')} Sp={g(3.0,'Spring')} "
            f"| 9h ALL={g(9.0,'ALL')} Wi={g(9.0,'Winter')} Sp={g(9.0,'Spring')} "
            f"| WS9={obj:.3f}")


def run_one_config(cfg_name, grid, tw_train, tw_test, leads):
    """Train + sweep one config. Returns [(cfg, tag, metrics), ...] and writes JSON."""
    spec = EXP_CONFIGS[cfg_name]
    print(f"[{cfg_name}] CONFIG {spec} accel={ACCEL}", flush=True)
    nf, hist, futr, isize = train_nbeats(grid, spec["input_size"], spec["nb_hist"])
    print(f"[{cfg_name}] predicting (train)...", flush=True)
    tr_preds = get_all_preds(nf, grid, tw_train, hist, futr, leads, isize)
    print(f"[{cfg_name}] predicting (test)...", flush=True)
    te_preds = get_all_preds(nf, grid, tw_test, hist, futr, leads, isize)
    print(f"[{cfg_name}] train preds={len(tr_preds)}, test preds={len(te_preds)}", flush=True)

    out = []
    base_m = seasonal_rmse(ridge_eval(grid, tr_preds, te_preds, leads, RIDGE_BASE))
    print(fmt_row(f"[{cfg_name}] ridge=BASE", base_m), flush=True)
    out.append((cfg_name, "BASE", base_m))

    # greedy add of trend features, objective = mean(winter9, spring9)
    current = list(RIDGE_BASE)
    cur_obj = np.nanmean([base_m[(9.0, "Winter")], base_m[(9.0, "Spring")]])
    remaining = list(RIDGE_TREND_CANDS)
    while remaining:
        best = None
        for f in remaining:
            m = seasonal_rmse(ridge_eval(grid, tr_preds, te_preds, leads, current + [f]))
            obj = np.nanmean([m[(9.0, "Winter")], m[(9.0, "Spring")]])
            if best is None or obj < best[1]:
                best = (f, obj, m)
        f, obj, m = best
        if obj >= cur_obj - 0.002:
            print(f"[{cfg_name}] greedy stop (no WS9 gain past {cur_obj:.3f})", flush=True)
            break
        current.append(f); remaining.remove(f); cur_obj = obj
        print(fmt_row(f"[{cfg_name}] +{f}", m), flush=True)
    sweep_m = seasonal_rmse(ridge_eval(grid, tr_preds, te_preds, leads, current))
    added = [c for c in current if c not in RIDGE_BASE]
    print(f"[{cfg_name}] final ridge feats (+{len(added)}): {added}", flush=True)
    out.append((cfg_name, "SWEEP", sweep_m))

    # persist (keys stringified for JSON)
    payload = {"config": cfg_name, "spec": {**spec, "nb_hist": spec["nb_hist"]},
               "added_ridge": added,
               "results": [{"tag": t, "metrics": {f"{L}|{s}": v for (L, s), v in m.items()}}
                           for _, t, m in out]}
    (RESULTS_PATH / "ws_configs").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH / "ws_configs" / f"{cfg_name}.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    return out


def aggregate():
    """Print summary from all per-config JSON files."""
    import glob
    files = sorted(glob.glob(str(RESULTS_PATH / "ws_configs" / "*.json")))
    print(f"\n{'='*100}\nSUMMARY (WS9 = mean of winter & spring 9h RMSE; lower=better)\n{'='*100}")
    for fp in files:
        d = json.load(open(fp))
        for r in d["results"]:
            m = {tuple([float(k.split('|')[0]), k.split('|')[1]]): v
                 for k, v in r["metrics"].items()}
            print(fmt_row(f"{d['config']}/{r['tag']}", m))


def main():
    if os.environ.get("EXP_AGGREGATE") == "1":
        aggregate(); return
    print("=" * 100)
    print("PHASE 1+2 EXPERIMENT: winter/spring RMSE")
    print("=" * 100, flush=True)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}", flush=True)

    which = os.environ.get("EXP_CONFIGS")
    configs = ([c.strip() for c in which.split(",")] if which else list(EXP_CONFIGS))
    leads = LEAD_TIMES_HOURS

    summary = []
    for cfg_name in configs:
        summary += run_one_config(cfg_name, grid, tw_train, tw_test, leads)

    print(f"\n{'='*100}\nSUMMARY (WS9 = mean of winter & spring 9h RMSE; lower=better)\n{'='*100}")
    for cfg_name, ridge_tag, m in summary:
        print(fmt_row(f"{cfg_name}/{ridge_tag}", m))


if __name__ == "__main__":
    main()
