"""NBEATSx + Ridge feature sweep v2.

Architecture:
  - NBEATSx: pre-2025 train, direct-T, width=16, 700 steps, 4 hist features.
             Fixed — not swept here.
  - Ridge: trained on 2025 EVEN days, tested on 2025 ODD days.
           This captures the real extrapolation residual pattern.

Also tests: NBEATSx with vs without futr_exog (solar/doy phase features).

Usage:
    uv run --no-sync python forecast/exp_ridge_sweep_v2.py
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import run
import feature_sweep as fs
from config import (NBEATS_INPUT_SIZE, NBEATS_HORIZON,
                    SOLAR_GRID_FREQ, TEST_START_DATE, NBEATS_MAX_STEPS)

WIDTH = 16
MAX_STEPS = NBEATS_MAX_STEPS
LEADS = [3.0, 6.0, 9.0, 12.0]
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

# Ridge base features (evaluated at issue time)
RIDGE_BASE = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "solar_sin", "solar_cos", "doy_sin", "doy_cos",
]

# Ridge candidate features to sweep
RIDGE_CANDIDATES = [
    "y_lag_96", "y_diff_24h", "y_diff_60", "y_diff_72h",
    "y_diff_30h", "y_diff_39h", "y_diff_45h",
    "cooling_rate_3h", "cooling_rate_6h",
    "trend_solar_4h", "trend_solar_6h",
    "last_std_24h", "last_max_24h", "last_min_24h",
    "DTR", "DTR_3d", "dDTR_1d",
    "dmean_1d", "dmean_3d",
    "dTmax_1d", "dTmax_3d", "temp_trend_3d", "Tmean_24h",
    "max_minus_sunrise", "temp_last_sunrise",
    "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
    "rate_sunrise_to_midday", "rate_midday_to_twilight",
    "velocity_sunrise", "velocity_noon",
]


def train_nbeats(grid, use_futr=True):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns] if use_futr else []
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


def get_nb_preds(nf, grid, tw_set, hist, futr, leads):
    """Get T_nb for each twilight at each lead. Returns dict[(ev_i, L)] -> {...}"""
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
            requests.append((ev_i, tgt, iss, L, ev["y_actual"]))
    for b0 in range(0, len(requests), BATCH):
        batch = requests[b0:b0+BATCH]
        all_h, all_f, meta = [], [], []
        for (ev_i, tgt, iss, L, yact) in batch:
            hs=iss-NBEATS_INPUT_SIZE; fe=iss+NBEATS_HORIZON
            if hs<0 or fe>len(grid): continue
            hd=grid.iloc[hs:iss][["ds","y"]+allx].copy()
            if hd[allx].isna().any().any(): continue
            fd=grid.iloc[iss:fe][["ds"]+(futr if futr else [])].copy()
            if futr and fd[futr].isna().any().any(): continue
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


def eval_ridge_set(grid, preds_even, preds_odd, feat_list, lead):
    """Train Ridge on even-day preds, test on odd-day preds. Return RMSE."""
    feat_cols = [c for c in feat_list if c in grid.columns]

    def build_Xy(preds):
        X, Y = [], []
        for (ev_i, L), p in preds.items():
            if L != lead:
                continue
            feats = grid.iloc[p["iss"]][feat_cols].values
            if np.any(pd.isna(feats)):
                continue
            row = np.concatenate([feats, [p["T_nb"]]])
            X.append(row); Y.append(p["y_actual"])
        return np.array(X), np.array(Y)

    X_tr, y_tr = build_Xy(preds_even)
    X_te, y_te = build_Xy(preds_odd)
    if len(X_tr) < 30 or len(X_te) < 30:
        return np.nan

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)

    preds_te = ridge.predict(scaler.transform(X_te))
    errs = y_te - preds_te
    return float(np.sqrt((errs**2).mean()))


def main():
    print("="*72)
    print(f"NBEATSx + RIDGE SWEEP v2  |  Ridge: even/odd 2025 split")
    print("="*72)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])

    # 2025 twilights split into even/odd days
    tw_2025 = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    tw_2025["day_of_year"] = tw_2025["ds_real"].dt.dayofyear
    tw_even = tw_2025[tw_2025["day_of_year"] % 2 == 0].reset_index(drop=True)
    tw_odd = tw_2025[tw_2025["day_of_year"] % 2 == 1].reset_index(drop=True)
    print(f"  2025 even-day twilights (Ridge train): {len(tw_even)}")
    print(f"  2025 odd-day twilights  (Ridge test):  {len(tw_odd)}")

    # ========== TEST: futr_exog impact ==========
    print("\n--- TEST: futr_exog impact on NBEATSx ---")
    print("Training NBEATSx WITH futr_exog...")
    nf_with, hist_w, futr_w = train_nbeats(grid, use_futr=True)
    preds_w = get_nb_preds(nf_with, grid, tw_odd, hist_w, futr_w, [3.0])
    errs_w = [p["y_actual"] - p["T_nb"] for p in preds_w.values()]
    rmse_w = np.sqrt(np.mean(np.array(errs_w)**2))

    print("Training NBEATSx WITHOUT futr_exog...")
    nf_no, hist_n, futr_n = train_nbeats(grid, use_futr=False)
    preds_n = get_nb_preds(nf_no, grid, tw_odd, hist_n, futr_n, [3.0])
    errs_n = [p["y_actual"] - p["T_nb"] for p in preds_n.values()]
    rmse_n = np.sqrt(np.mean(np.array(errs_n)**2))

    print(f"  With futr_exog:    RMSE@3h = {rmse_w:.3f}")
    print(f"  Without futr_exog: RMSE@3h = {rmse_n:.3f}")
    print(f"  Delta: {100*(rmse_n-rmse_w)/rmse_w:+.1f}%")

    # Use the better one for the Ridge sweep
    use_futr = rmse_w <= rmse_n
    nf = nf_with if use_futr else nf_no
    hist = hist_w if use_futr else hist_n
    futr = futr_w if use_futr else futr_n
    print(f"  -> Using {'WITH' if use_futr else 'WITHOUT'} futr_exog")

    # ========== Get NBEATSx predictions for even + odd 2025 ==========
    print("\nGetting NBEATSx predictions (even days)...")
    preds_even = get_nb_preds(nf, grid, tw_even, hist, futr, LEADS)
    print(f"  {len(preds_even)} preds")
    print("Getting NBEATSx predictions (odd days)...")
    preds_odd = get_nb_preds(nf, grid, tw_odd, hist, futr, LEADS)
    print(f"  {len(preds_odd)} preds")

    # ========== Ridge feature sweep at 3h ==========
    SWEEP_LEAD = 3.0
    print(f"\n--- RIDGE FEATURE SWEEP @{SWEEP_LEAD}h ---")
    base_rmse = eval_ridge_set(grid, preds_even, preds_odd, RIDGE_BASE, SWEEP_LEAD)
    print(f"Base ({len(RIDGE_BASE)} feats + T_nb): RMSE = {base_rmse:.3f}")

    available = [c for c in RIDGE_CANDIDATES if c in grid.columns
                 and c not in RIDGE_BASE
                 and grid[c].notna().sum() > 0.8 * len(grid)]
    print(f"Candidates: {len(available)}")

    current_feats = list(RIDGE_BASE)
    current_rmse = base_rmse
    round_num = 0

    while available:
        round_num += 1
        print(f"\n  Round {round_num} ({len(current_feats)} feats, RMSE={current_rmse:.3f})")
        results = []
        for feat in available:
            rmse = eval_ridge_set(grid, preds_even, preds_odd, current_feats + [feat], SWEEP_LEAD)
            results.append((feat, rmse))
        valid = [(f, r) for f, r in results if not np.isnan(r)]
        if not valid:
            break
        best_feat, best_rmse = min(valid, key=lambda x: x[1])
        imp = (current_rmse - best_rmse) / current_rmse * 100
        if best_rmse >= current_rmse - 0.003:
            print(f"  No improvement (best: {best_feat} {best_rmse:.3f}, +{imp:.1f}%). Stopping.")
            break
        current_feats.append(best_feat)
        available.remove(best_feat)
        current_rmse = best_rmse
        print(f"  -> Added '{best_feat}': RMSE={current_rmse:.3f} ({imp:+.1f}%)")

    # ========== Final eval at all leads ==========
    print(f"\n{'='*72}")
    print(f"FINAL NB+Ridge ({len(current_feats)} feats + T_nb)")
    print(f"Features: {current_feats}")
    print(f"{'='*72}")
    print(f"\n{'lead':>5} {'NBEATSx':>8} {'Ridge-only':>10} {'NB+Ridge':>9} {'Δ vs Ridge':>10}")
    print("-"*48)
    for L in LEADS:
        # NBEATSx alone
        nb_errs = [p["y_actual"]-p["T_nb"] for (ei,l),p in preds_odd.items() if l==L]
        rmse_nb = np.sqrt(np.mean(np.array(nb_errs)**2)) if nb_errs else np.nan
        # Ridge-only (no T_nb)
        rmse_lin = eval_ridge_set(grid, preds_even, preds_odd,
                                  current_feats, L)  # this still has T_nb in the function
        # Actually need a version without T_nb for "Ridge-only"
        # Hack: eval with base feats only, no T_nb column -> need separate fn
        # For now just report NB+Ridge
        rmse_full = eval_ridge_set(grid, preds_even, preds_odd, current_feats, L)
        print(f"{L:>5} {rmse_nb:>8.3f} {'--':>10} {rmse_full:>9.3f}")

    print(f"\nOptimal features: {current_feats}")
    print(f"RMSE@3h: {current_rmse:.3f} (vs Linear alone ~0.857, vs base NB+Ridge {base_rmse:.3f})")


if __name__ == "__main__":
    main()
PY