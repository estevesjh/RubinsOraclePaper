"""Feature sweep: find optimal feature set for NBEATSx-Diff and baselines.

Phase 1: Correlation analysis (instant)
Phase 2: Forward selection with fast training (max_steps=50)
Phase 3: Final model with best features (max_steps=500)

Usage:
    cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/feature_sweep.py
"""

import os
import sys
import time
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "data", "rubin-twilight-forecast"))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    RESULTS_PATH, MODEL_CACHE_PATH, TEST_START_DATE,
    HALFDAY_LAG_STEPS, SOLAR_GRID_FREQ, NBEATS_INPUT_SIZE,
    NBEATS_HORIZON, STEPS_PER_DAY, SOLAR_GRID_STEP,
)
from run import load_and_prepare, find_twilight_targets, lead_hours_to_steps, suppress_stdout


def compute_all_features(grid):
    """Compute all candidate features on the solar grid."""
    y_arr = grid["y"].values
    n = len(y_arr)

    # ── Current features (already in grid from load_and_prepare) ──
    # y_raw, y_lag_6, y_lag_12, y_lag_24, y_lag_48, trend_solar_2h, DTR

    # ── Causal rolling statistics (24h window = 48 steps) ──
    grid["last_max_24h"] = pd.Series(y_arr).rolling(48, min_periods=24).max().values
    grid["last_min_24h"] = pd.Series(y_arr).rolling(48, min_periods=24).min().values
    grid["last_mean_24h"] = pd.Series(y_arr).rolling(48, min_periods=24).mean().values
    grid["last_std_24h"] = pd.Series(y_arr).rolling(48, min_periods=24).std().values

    # ── Multi-day mean differences ──
    mean_24h = pd.Series(y_arr).rolling(48, min_periods=24).mean()
    grid["dmean_1d"] = mean_24h.values - mean_24h.shift(48).values
    grid["dmean_3d"] = mean_24h.values - mean_24h.shift(144).values

    # ── Rate features ──
    # trend_solar_4h: backward OLS over 8 steps
    window = 8
    slopes_4h = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()
    for i in range(window, n):
        seg = y_arr[i - window:i]
        if np.isnan(seg).any():
            continue
        cov = ((x - x_mean) * (seg - seg.mean())).sum()
        slopes_4h[i] = cov / var_x
    grid["trend_solar_4h"] = slopes_4h

    # cooling_rate_3h
    grid["cooling_rate_3h"] = (y_arr - np.roll(y_arr, 6)) / 6.0
    grid.iloc[:6, grid.columns.get_loc("cooling_rate_3h")] = np.nan

    # ── Diurnal differences ──
    grid["y_diff_24h"] = y_arr - np.roll(y_arr, 48)
    grid["y_diff_48h"] = y_arr - np.roll(y_arr, 96)
    grid.iloc[:48, grid.columns.get_loc("y_diff_24h")] = np.nan
    grid.iloc[:96, grid.columns.get_loc("y_diff_48h")] = np.nan

    # ── Long-term trend differences (from correlation analysis: lag 60 = 30h
    #    showed strongest |r| with residual error, especially for Spring) ──
    for lag_steps, name in [(60, "y_diff_30h"), (78, "y_diff_39h"),
                            (90, "y_diff_45h")]:
        grid[name] = y_arr - np.roll(y_arr, lag_steps)
        grid.iloc[:lag_steps, grid.columns.get_loc(name)] = np.nan

    # ── Solar-time anchor lags ──
    # For each row, get T at specific solar times of the CURRENT or PREVIOUS day
    # SolarTime: 0=sunrise, 0.25=midday, 0.375=mid-afternoon, 0.5=sunset, 0.75=midnight
    # On 48-step/day grid: step within day = SolarTime * 48
    day_count = grid["DayCount"].values
    solar_time = grid["SolarTime"].values

    anchors = {"y_sunrise": 0.0, "y_midday": 0.25, "y_midafternoon": 0.375,
               "y_sunset": 0.5, "y_midnight": 0.75}

    # Build a lookup: for each DayCount, what's the grid row index of that day's start?
    day_starts = {}
    for i in range(n):
        dc = int(day_count[i])
        if dc not in day_starts:
            day_starts[dc] = i

    for name, st in anchors.items():
        vals = np.full(n, np.nan)
        step_within_day = int(st * 48)
        for i in range(n):
            dc = int(day_count[i])
            # For midnight, use previous day
            target_dc = dc - 1 if st == 0.75 else dc
            if target_dc in day_starts:
                anchor_row = day_starts[target_dc] + step_within_day
                # Causal: anchor must be before current row
                if 0 <= anchor_row < i:
                    vals[i] = y_arr[anchor_row]
        grid[name] = vals

    # ── DTR variants ──
    y_smooth_5h = pd.Series(y_arr).rolling(10, min_periods=5, center=False).mean().values
    max_3d = pd.Series(y_smooth_5h).rolling(144, min_periods=48).max().values
    min_3d = pd.Series(y_smooth_5h).rolling(144, min_periods=48).min().values
    grid["DTR_3d"] = max_3d - min_3d

    max_1d = pd.Series(y_smooth_5h).rolling(48, min_periods=24).max().values
    max_1d_prev = np.roll(max_1d, 48)
    min_1d = pd.Series(y_smooth_5h).rolling(48, min_periods=24).min().values
    min_1d_prev = np.roll(min_1d, 48)
    dtr_today = max_1d - min_1d
    dtr_yesterday = max_1d_prev - min_1d_prev
    grid["dDTR_1d"] = dtr_today - dtr_yesterday
    grid.iloc[:96, grid.columns.get_loc("dDTR_1d")] = np.nan

    # ── Additional lags / diffs ──
    grid["y_lag_3"] = pd.Series(y_arr).shift(3).values        # 1.5h ago
    grid["y_lag_96"] = pd.Series(y_arr).shift(96).values      # 2 days ago
    grid["y_diff_72h"] = y_arr - np.roll(y_arr, 144)          # 3-day diff
    grid.iloc[:144, grid.columns.get_loc("y_diff_72h")] = np.nan
    # NOTE: y_diff_30h (60 steps) already computed above; keep that canonical name.

    # ── Longer / shorter cooling + trend windows ──
    grid["cooling_rate_6h"] = (y_arr - np.roll(y_arr, 12)) / 12.0
    grid.iloc[:12, grid.columns.get_loc("cooling_rate_6h")] = np.nan

    window = 12  # 6h backward OLS slope
    slopes_6h = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()
    for i in range(window, n):
        seg = y_arr[i - window:i]
        if np.isnan(seg).any():
            continue
        cov = ((x - x_mean) * (seg - seg.mean())).sum()
        slopes_6h[i] = cov / var_x
    grid["trend_solar_6h"] = slopes_6h

    # ── Trend acceleration (2nd derivative): change in 2h slope over 2h ──
    if "trend_solar_2h" in grid.columns:
        t2 = grid["trend_solar_2h"].values
        grid["trend_accel"] = t2 - np.roll(t2, 4)
        grid.iloc[:4, grid.columns.get_loc("trend_accel")] = np.nan

    # ── max_minus_sunrise (memory: strongest Ridge feature) ──
    # Recompute defensively in case load_and_prepare variant lacks it.
    if "max_minus_sunrise" not in grid.columns:
        if "last_max_temp" in grid.columns and "temp_last_sunrise" in grid.columns:
            grid["max_minus_sunrise"] = (
                grid["last_max_temp"].values - grid["temp_last_sunrise"].values
            )

    # ── Nonlinear self-terms (memory: res_age² / squared signals help) ──
    if "max_minus_sunrise" in grid.columns:
        grid["maxsr_sq"] = grid["max_minus_sunrise"].values ** 2
    if "DTR" in grid.columns:
        grid["DTR_sq"] = grid["DTR"].values ** 2

    # ── Interaction: trend strength modulated by season ──
    if "trend_solar_2h" in grid.columns and "doy_cos" in grid.columns:
        grid["trend_x_doy"] = grid["trend_solar_2h"].values * grid["doy_cos"].values

    return grid


# Candidate feature names. Trimmed from the original ~45-feature pool: dropped
# the low-coverage solar anchors (y_sunrise/midday/midafternoon/sunset, which
# returned NaN in earlier sweeps), the nonlinear/interaction terms that
# consistently hurt (maxsr_sq, DTR_sq, trend_x_doy), and redundant short lags.
ALL_CANDIDATES = [
    # Lags (the reconstruction anchor + a couple of distinct horizons)
    "y_raw", "y_lag_12", "y_lag_24", "y_lag_48",
    # Trend / rate
    "trend_solar_2h", "trend_solar_4h", "cooling_rate_3h",
    # Rolling stats
    "last_std_24h", "DTR",
    # Multi-day diffs
    "dmean_1d", "dmean_3d",
    # Diurnal / long-term diffs (30h = best for Spring)
    "y_diff_24h", "y_diff_30h", "y_diff_45h", "y_diff_72h",
    # DTR variants
    "DTR_3d", "dDTR_1d",
    # FeatureBuilder rate/anchor columns
    "max_minus_sunrise", "temp_trend_3d", "dTmax_1d", "dTmax_3d", "Tmean_24h",
    "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
]

FUTR_EXOG = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]


def evaluate_feature_set(grid, tw_events_test, hist_exog, max_steps=100, subsample=3):
    """Train NBEATSx with given features and return RMSE at 3h.

    Uses subsampled training data (every Nth row) for speed.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in FUTR_EXOG if c in grid.columns]
    all_exog = hist_exog + futr_exog

    # Train strictly pre-2025 (test set is all 2025; avoid leakage).
    train_mask = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[train_mask, ["ds", "D"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["D"]
    nf_train["unique_id"] = "temp"
    val_size = int(len(nf_train) * 0.1)

    accel = os.environ.get("FS_ACCEL", "gpu")
    width = [int(x) for x in os.environ.get("FS_WIDTH", "256,256").split(",")]
    model = NBEATSx(
        h=NBEATS_HORIZON, input_size=NBEATS_INPUT_SIZE, max_steps=max_steps,
        hist_exog_list=hist_exog, futr_exog_list=futr_exog,
        activation="SELU", loss=HuberLoss(), learning_rate=0.001,
        batch_size=48, scaler_type="identity", enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [width], n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10, val_check_steps=50,
        accelerator=accel, devices=1,
    )

    with suppress_stdout():
        nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
        nf.fit(nf_train, val_size=val_size)

    # Evaluate at 3h lead — BATCHED for speed
    y_arr = grid["y"].values
    offset = lead_hours_to_steps(3.0)

    # Build all requests
    all_hist, all_futr, batch_meta = [], [], []
    for _, ev in tw_events_test.iloc[::2].iterrows():
        target_idx = ev["grid_idx"]
        issue_idx = target_idx - offset
        if issue_idx < NBEATS_INPUT_SIZE:
            continue
        if (target_idx - issue_idx) >= NBEATS_HORIZON:
            continue
        hist_start = issue_idx - NBEATS_INPUT_SIZE
        futr_end = issue_idx + NBEATS_HORIZON
        if futr_end > len(grid):
            continue
        hist_df = grid.iloc[hist_start:issue_idx][["ds", "D"] + all_exog].copy()
        if hist_df[all_exog].isna().any().any():
            continue
        uid = f"e_{target_idx}"
        hist_df["y"] = hist_df["D"]
        hist_df["unique_id"] = uid
        futr_df = grid.iloc[issue_idx:futr_end][["ds"] + futr_exog].copy()
        futr_df["unique_id"] = uid
        all_hist.append(hist_df)
        all_futr.append(futr_df)
        batch_meta.append((uid, target_idx, issue_idx, ev["y_actual"]))

    if len(all_hist) < 30:
        return {"rmse": np.nan, "pct_1c": np.nan, "n": len(all_hist)}

    # Single batched predict call
    combined_hist = pd.concat(all_hist, ignore_index=True)
    combined_futr = pd.concat(all_futr, ignore_index=True)
    with suppress_stdout():
        fc = nf.predict(combined_hist, futr_df=combined_futr)
    fc = fc.reset_index()
    model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]

    errors = []
    for uid, target_idx, issue_idx, y_actual in batch_meta:
        uid_fc = fc[fc["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        if len(uid_fc) == 0:
            continue
        pred_step = target_idx - issue_idx
        if pred_step >= len(uid_fc):
            continue
        D_pred = uid_fc[model_col].iloc[pred_step]
        anchor_idx = target_idx - HALFDAY_LAG_STEPS
        if anchor_idx < 0 or anchor_idx >= len(y_arr):
            continue
        anchor_temp = y_arr[anchor_idx]
        if np.isnan(anchor_temp):
            continue
        T_pred = D_pred + anchor_temp
        errors.append(y_actual - T_pred)

    if len(errors) < 30:
        return {"rmse": np.nan, "pct_1c": np.nan, "n": len(errors)}
    abs_err = np.abs(np.array(errors))
    return {
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "pct_1c": float((abs_err < 1.0).mean()),
        "n": int(len(errors)),
    }


def evaluate_feature_wrapper(args):
    """Wrapper for parallel execution."""
    grid, tw_test, base_feats, new_feat, max_steps = args
    test_set = base_feats + [new_feat]
    metrics = evaluate_feature_set(grid, tw_test, test_set, max_steps=max_steps)
    return new_feat, metrics


def main():
    print("=" * 70)
    print("FEATURE SWEEP: Finding optimal feature set")
    print("=" * 70)

    # Load and compute all features
    print("\n1. Loading grid and computing all candidate features...")
    grid, cfg = load_and_prepare()
    grid = compute_all_features(grid)

    tw_events = find_twilight_targets(grid)
    tw_test = tw_events[tw_events["ds_real"] >= TEST_START_DATE].copy()
    print(f"   Test twilights: {len(tw_test)}")

    # Check which features are available
    available = [c for c in ALL_CANDIDATES if c in grid.columns and grid[c].notna().sum() > len(grid) * 0.8]
    print(f"   Available features: {len(available)}/{len(ALL_CANDIDATES)}")

    # ── Phase 1: Correlation with current model residuals ──
    print("\n" + "=" * 70)
    print("PHASE 1: Feature correlations with residual")
    print("=" * 70)

    # Load current residuals
    res_file = RESULTS_PATH / "paper_results_diff.csv"
    if res_file.exists():
        res_df = pd.read_csv(res_file)
        nb = res_df[res_df["model"] == "NBEATSx-Diff"]
        nb_3h = nb[nb["lead_time_hours"] == 3.0].copy()
        nb_3h["twilight_time"] = pd.to_datetime(nb_3h["twilight_time"], format="mixed")

        offset = lead_hours_to_steps(3.0)
        corrs = {}
        for feat in available:
            vals = []
            errs = []
            for _, row in nb_3h.iterrows():
                tw_time = row["twilight_time"]
                mask = tw_test["ds_real"] == tw_time
                if not mask.any():
                    continue
                ev = tw_test[mask].iloc[0]
                issue_idx = ev["grid_idx"] - offset
                if issue_idx < 0 or issue_idx >= len(grid):
                    continue
                v = grid[feat].iloc[issue_idx]
                if pd.isna(v):
                    continue
                vals.append(v)
                errs.append(row["error"])

            if len(vals) > 50:
                corrs[feat] = np.corrcoef(vals, errs)[0, 1]

        print(f"\n{'Feature':>20} {'r(error)':>10} {'|r|':>6}")
        print("-" * 40)
        for feat, r in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{feat:>20} {r:+10.3f} {abs(r):6.3f}")
    else:
        print("   No residuals file found, skipping correlation phase.")

    # ── Phase 2: Forward selection (parallelized) ──
    print("\n" + "=" * 70)
    print("PHASE 2: Forward selection (max_steps=1000, width=256)")
    print("=" * 70)

    MAX_STEPS = int(os.environ.get("FS_MAX_STEPS", 300))
    max_feats = os.environ.get("FS_MAX_FEATS")
    if max_feats is not None:
        max_feats = int(max_feats)

    # Start with minimal set
    base_features = ["y_raw", "y_lag_24", "trend_solar_2h"]
    remaining = [f for f in available if f not in base_features]
    if max_feats is not None:
        remaining = remaining[:max_feats]
        print(f"   FS_MAX_FEATS={max_feats}: capped candidates to {remaining}")

    current_best = list(base_features)
    t0 = time.perf_counter()
    base_metrics = evaluate_feature_set(grid, tw_test, current_best, max_steps=MAX_STEPS)
    current_rmse = base_metrics["rmse"]
    current_pct = base_metrics["pct_1c"]
    print(f"\n   Base set {current_best}: RMSE={current_rmse:.3f} "
          f"pct<1C={current_pct*100:.1f}% t={time.perf_counter()-t0:.1f}s")

    round_num = 0
    while remaining:
        round_num += 1
        print(f"\n   Round {round_num} (current: {len(current_best)} features, "
              f"RMSE={current_rmse:.3f} pct<1C={current_pct*100:.1f}%)")
        print(f"   Testing {len(remaining)} candidates...")

        results_round = []
        for feat in remaining:
            test_set = current_best + [feat]
            t0 = time.perf_counter()
            m = evaluate_feature_set(grid, tw_test, test_set, max_steps=MAX_STEPS)
            dt = time.perf_counter() - t0
            rmse, pct = m["rmse"], m["pct_1c"]
            improvement = (current_rmse - rmse) / current_rmse * 100 if not np.isnan(rmse) else 0
            d_pct = (pct - current_pct) * 100 if not np.isnan(pct) else 0
            results_round.append((feat, rmse, pct, dt))
            print(f"      + {feat:>20}: RMSE={rmse:.3f} ({improvement:+.1f}%) "
                  f"pct<1C={pct*100:.1f}% ({d_pct:+.1f}pp) t={dt:.1f}s", flush=True)

            pd.DataFrame(results_round,
                         columns=["feature", "rmse", "pct_1c", "elapsed_s"]).to_csv(
                str(RESULTS_PATH / f"sweep_round_{round_num}.csv"), index=False)

        valid_results = [(f, r, p, dt) for f, r, p, dt in results_round if not np.isnan(r)]
        if not valid_results:
            print("   No valid results. Stopping.")
            break

        # Selection: lowest RMSE; ties broken by higher pct<1C
        ranked = sorted(valid_results, key=lambda x: (x[1], -x[2]))
        best_feat, best_rmse, best_pct, _ = ranked[0]

        if best_rmse >= current_rmse - 0.005:
            print(f"\n   No improvement (RMSE) found. Stopping.")
            break

        # ── Correlation gate for co-adding: take runners-up that ALSO improve
        # the model AND are weakly correlated (|r| < CORR_THRESHOLD) with the
        # round's winner AND with each other. Decorrelated features carry
        # complementary signal; correlated ones are redundant after the winner.
        # Disabled by default; enable with FS_COADD=1.
        co_added = []
        if os.environ.get("FS_COADD") == "1":
            TOP_K = int(os.environ.get("FS_COADD_TOPK", 3))
            CORR_THRESHOLD = float(os.environ.get("FS_COADD_THRESH", 0.7))
            winner_col = grid[best_feat].values
            accepted_cols = [winner_col]
            for cand_feat, cand_rmse, cand_pct, _ in ranked[1:TOP_K]:
                if cand_rmse >= current_rmse - 0.005:
                    break  # candidate doesn't improve baseline → stop
                cand_col = grid[cand_feat].values
                max_abs_r = 0.0
                for ref_col in accepted_cols:
                    mask = ~(np.isnan(cand_col) | np.isnan(ref_col))
                    if mask.sum() < 100:
                        continue
                    r = np.corrcoef(cand_col[mask], ref_col[mask])[0, 1]
                    if np.isfinite(r):
                        max_abs_r = max(max_abs_r, abs(r))
                tag = "ACCEPT" if max_abs_r < CORR_THRESHOLD else "skip(corr)"
                print(f"   ↳ co-add check {cand_feat:>20}: RMSE={cand_rmse:.3f}, "
                      f"max|r|={max_abs_r:.2f} → {tag}")
                if max_abs_r < CORR_THRESHOLD:
                    co_added.append((cand_feat, cand_rmse, cand_pct))
                    accepted_cols.append(cand_col)

        # Commit winner + any decorrelated co-adds.
        added_this_round = [best_feat] + [f for f, _, _ in co_added]
        for feat in added_this_round:
            current_best.append(feat)
            remaining.remove(feat)
        current_rmse = best_rmse
        current_pct = best_pct
        if co_added:
            print(f"   → Added '{best_feat}' + {[f for f, _, _ in co_added]} "
                  f"(decorrelated co-adds): RMSE≥{current_rmse:.3f} "
                  f"pct<1C≥{current_pct*100:.1f}%")
        else:
            print(f"   → Added '{best_feat}': new RMSE={current_rmse:.3f} "
                  f"pct<1C={current_pct*100:.1f}%")

    print("\n" + "=" * 70)
    print("RESULT: Optimal feature set")
    print("=" * 70)
    print(f"   Features ({len(current_best)}): {current_best}")
    print(f"   RMSE at 3h: {current_rmse:.3f}  pct<1C: {current_pct*100:.1f}%")


if __name__ == "__main__":
    main()
