"""Ridge-stage PCA feature sweep (parallel over CPU cores).

Motivation: the Ridge correction pool carries heavy multicollinearity — many
overlapping lags/diffs, and interaction terms collinear with doy_sin/cos (see
project memory). PCA orthogonalizes that block into uncorrelated components, so
the Ridge stage sees a compact, stable, decorrelated input. This is the
"temporal / lagged PCA" regime (one summit series, many engineered features),
NOT cross-sectional factor PCA.

Design (mirrors exp_humidity.py / exp_winter_spring.py):
  1. Train the paper NBEATSx ONCE on MPS (pre-2025), get T_nb for every
     twilight × lead — the expensive, GPU-bound step, run a single time.
  2. Build, per (pool, lead), the engineered-feature matrix read at the ISSUE
     row (causal) for train and test, plus the T_nb anchor.
  3. Fan the PCA grid (pool × whiten × n_components × ridge alpha) across CPU
     processes. Each fit is pure numpy/sklearn and milliseconds; the grid is
     embarrassingly parallel.

Leakage discipline (essay §4 "Data Leakage Risk"): StandardScaler and PCA are
fit on pre-2025 issue rows ONLY, then applied to 2025. T_nb is the
reconstruction anchor, not an exogenous feature, so it bypasses PCA and is
standardized on its own train statistics before being concatenated after the
PCs.

Usage:
    uv run --no-sync python forecast/exp_pca_sweep.py
    # quick smoke test (base+trend pools, fewer alphas):
    PCA_POOLS=base,trend PCA_ALPHAS=1.0 uv run --no-sync python forecast/exp_pca_sweep.py
    # cap worker processes:
    PCA_WORKERS=8 uv run --no-sync python forecast/exp_pca_sweep.py
"""

import os, sys, warnings, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
import logging
for _n in ["pytorch_lightning", "lightning.pytorch", "lightning"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# Workers do BLAS-heavy sklearn fits; cap per-process threads so the pool does
# not oversubscribe the 18 cores (same guard as exp_winter_spring.py).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
import feature_sweep as fs
import exp_humidity as eh  # reuse train_nbeats / get_all_preds / season_of / seasonal_rmse
from config import (TEST_START_DATE, RESULTS_PATH, LEAD_TIMES_HOURS,
                    HALFDAY_LAG_STEPS, NBEATS_INPUT_SIZE, NBEATS_HORIZON)

# ── Mode ──────────────────────────────────────────────────────────────────
# "nbeats" (default): PCA(features) + standardized T_nb → Ridge predicts T.
#                     The NBEATSx prediction is the dominant input; PCA only
#                     reshapes the side features that *correct* it.
# "ridge":  pure-linear analog of run.py — NO NBEATSx. Ridge predicts the
#           differenced target D = y - y[t-12h] from PCA(features) ALONE, then
#           reconstructs T = D_pred + anchor (anchor = y[target-12h], causal).
#           This isolates PCA's value: with no T_nb to lean on, the components
#           must carry the forecast themselves.
MODE = os.environ.get("PCA_MODE", "nbeats")

# ── Raw lag bank: y[issue-j] for j=1..96 (2 solar days). Highly autocorrelated
# → the canonical "temporal/lagged PCA" regime (Approach B), where PCA actually
# has redundant variance to compress, unlike the hand-curated pools below. ──
LAG_BANK_N = 96
LAG_COLS = [f"lag_{j:03d}" for j in range(1, LAG_BANK_N + 1)]
# Differenced lag banks: each base series differenced (stationary; removes the
# diurnal level — the essay's "make it stationary before PCA" step), then lagged
# 1..96. d24 = y - y[t-48] (vs yesterday, same solar time); d12 = y - y[t-24].
D24_COLS = [f"d24_lag_{j:03d}" for j in range(1, LAG_BANK_N + 1)]
D12_COLS = [f"d12_lag_{j:03d}" for j in range(1, LAG_BANK_N + 1)]

# ── Feature pools fed to PCA (the engineered block; T_nb appended after) ──
POOLS = {
    "base":     eh.RIDGE_BASE,
    "trend":    eh.RIDGE_BASE + eh.TREND_FEATS,
    "hw":       eh.RIDGE_BASE + eh.HW_FEATS,
    "full":     eh.RIDGE_BASE + eh.TREND_FEATS + eh.HW_FEATS,
    "lagbank":  LAG_COLS,
    "d24bank":  D24_COLS,
    "d12bank":  D12_COLS,
    "alllags":  LAG_COLS + D24_COLS + D12_COLS,  # every lagged value, one PCA
}


def add_lag_bank(grid, n=LAG_BANK_N):
    """Add raw + differenced lag banks, all read at the issue row → causal.

    lag_j[i]     = y[i-j]                         (raw level, j steps back)
    d24_lag_j[i] = (y - y.shift(48))[i-j]         (vs yesterday same solar time)
    d12_lag_j[i] = (y - y.shift(24))[i-j]         (vs 12h earlier)

    At the issue row i=issue, lag index j>=1 → every lookup (y[issue-j] and its
    24h/12h predecessors) is strictly before issuance. Leading rows are NaN and
    drop out in build_matrices."""
    y = pd.Series(grid["y"].values)
    series = {"lag": y.values,
              "d24": (y - y.shift(48)).values,
              "d12": (y - y.shift(24)).values}
    for prefix, arr in series.items():
        for j in range(1, n + 1):
            col = np.full(len(arr), np.nan)
            col[j:] = arr[:-j]
            name = f"lag_{j:03d}" if prefix == "lag" else f"{prefix}_lag_{j:03d}"
            grid[name] = col
    return grid

# ── Sweep axes (override via env for quick runs) ──
WHITEN_GRID = [False, True]
ALPHA_GRID = [float(a) for a in os.environ.get("PCA_ALPHAS", "0.1,0.3,1.0,3.0,10.0").split(",")]
VAR_THRESHOLDS = [0.90, 0.95, 0.99]  # auto-k rows: smallest k explaining >= threshold

KEY_LEADS = (3.0, 9.0)

# Worker-global payload (set via Pool initializer to avoid re-pickling per task).
_DATA = None


def get_all_preds_ridge(grid, tw_set, leads):
    """Pure-linear analog of eh.get_all_preds — NO NBEATSx.

    For each (twilight, lead) record the causal issue index and the
    reconstruction anchor y[target-12h]. Uses the SAME issue/horizon filter as
    the NBEATSx path so both modes evaluate an identical twilight×lead set.
    The 'anchor' rides in the T_nb slot so build_matrices is mode-agnostic.
    """
    y_arr = grid["y"].values
    preds = {}
    for ev_i, (_, ev) in enumerate(tw_set.iterrows()):
        tgt = int(ev["grid_idx"])
        anchor_idx = tgt - HALFDAY_LAG_STEPS
        if anchor_idx < 0 or anchor_idx >= len(y_arr) or np.isnan(y_arr[anchor_idx]):
            continue
        for L in leads:
            iss = tgt - run.lead_hours_to_steps(L)
            if iss < NBEATS_INPUT_SIZE or (tgt - iss) >= NBEATS_HORIZON:
                continue
            preds[(ev_i, L)] = {"T_nb": float(y_arr[anchor_idx]),  # anchor in T_nb slot
                                "y_actual": ev["y_actual"], "iss": iss,
                                "ds_real": ev["ds_real"]}
    return preds


def build_matrices(grid, tr_preds, te_preds, pool_cols, leads):
    """Per-lead feature matrices read at the issue row (causal).

    Returns {L: dict(X_tr, y_tr, Tnb_tr, X_te, y_te, Tnb_te, season_te,
                      hum_avail_te)} with NaN-feature rows dropped (humidity is
    imputed upstream, so base/trend pools drop nothing).
    """
    cols = [c for c in pool_cols if c in grid.columns]
    avail = grid["hum_avail"].values if "hum_avail" in grid.columns else None
    feat_mat = grid[cols].values  # row-indexable by issue idx
    out = {}
    for L in leads:
        Xtr, ytr, ntr = [], [], []
        for (ev_i, lead), p in tr_preds.items():
            if lead != L:
                continue
            f = feat_mat[p["iss"]]
            if np.any(np.isnan(f)):
                continue
            Xtr.append(f); ytr.append(p["y_actual"]); ntr.append(p["T_nb"])
        Xte, yte, nte, ste, ate = [], [], [], [], []
        for (ev_i, lead), p in te_preds.items():
            if lead != L:
                continue
            f = feat_mat[p["iss"]]
            if np.any(np.isnan(f)):
                continue
            Xte.append(f); yte.append(p["y_actual"]); nte.append(p["T_nb"])
            ste.append(eh.season_of(p["ds_real"]))
            ate.append(bool(avail[p["iss"]]) if avail is not None else True)
        if len(Xtr) < 30 or len(Xte) < 10:
            continue
        out[L] = dict(
            X_tr=np.asarray(Xtr), y_tr=np.asarray(ytr), Tnb_tr=np.asarray(ntr),
            X_te=np.asarray(Xte), y_te=np.asarray(yte), Tnb_te=np.asarray(nte),
            season_te=np.asarray(ste), hum_avail_te=np.asarray(ate),
        )
    return out, cols


def _init_worker(data):
    global _DATA
    _DATA = data


def _fit_one(args):
    """One (pool, whiten, k, alpha) point → per-(lead, season) RMSE rows.

    PCA pipeline per lead, fit on train only:
      features --StandardScaler--> --PCA(whiten)--> PCs[:k]
      T_nb     --(own StandardScaler)-->          concat after PCs --> Ridge
    Whitening is left meaningful by NOT re-standardizing the PC block: with
    whiten=False low-variance PCs stay small and Ridge naturally downweights
    them; with whiten=True every retained PC enters on equal footing.
    """
    pool, whiten, k, alpha, k_tag = args
    per_lead = _DATA[pool]
    rows = []
    for L, d in per_lead.items():
        Xtr, Xte = d["X_tr"], d["X_te"]
        n_feat = Xtr.shape[1]
        kk = min(k, n_feat)
        if kk < 1:
            continue
        fscaler = StandardScaler().fit(Xtr)
        pca = PCA(n_components=kk, whiten=whiten).fit(fscaler.transform(Xtr))
        Ptr = pca.transform(fscaler.transform(Xtr))
        Pte = pca.transform(fscaler.transform(Xte))
        if MODE == "ridge":
            # Pure linear: Ridge predicts D = y - anchor from PCs ALONE (Tnb slot
            # holds the anchor, NOT a regressor), then T = D_pred + anchor.
            anchor_tr, anchor_te = d["Tnb_tr"], d["Tnb_te"]
            D_tr = d["y_tr"] - anchor_tr
            ridge = SkRidge(alpha=alpha).fit(Ptr, D_tr)
            T_pred = ridge.predict(Pte) + anchor_te
            err = d["y_te"] - T_pred
        else:
            # NBEATSx-correction: standardized T_nb is the dominant regressor;
            # PCs only reshape the side features that correct it.
            nb_mu, nb_sd = d["Tnb_tr"].mean(), d["Tnb_tr"].std() or 1.0
            Ztr = np.column_stack([Ptr, (d["Tnb_tr"] - nb_mu) / nb_sd])
            Zte = np.column_stack([Pte, (d["Tnb_te"] - nb_mu) / nb_sd])
            ridge = SkRidge(alpha=alpha).fit(Ztr, d["y_tr"])
            err = d["y_te"] - ridge.predict(Zte)
        cumvar = float(pca.explained_variance_ratio_[:kk].sum())
        for e, s, ha in zip(err, d["season_te"], d["hum_avail_te"]):
            rows.append((pool, whiten, kk, k_tag, alpha, L, s, ha, float(e), cumvar))
    return rows


def auto_k_for_pool(per_lead, thresholds):
    """Smallest k whose pre-2025 PCA explains >= each variance threshold,
    computed at a representative lead (3h) and shared across the pool."""
    ref = per_lead.get(3.0) or next(iter(per_lead.values()))
    Xtr = ref["X_tr"]
    fscaler = StandardScaler().fit(Xtr)
    pca = PCA().fit(fscaler.transform(Xtr))
    cum = np.cumsum(pca.explained_variance_ratio_)
    ks = {}
    for thr in thresholds:
        k = int(np.searchsorted(cum, thr) + 1)
        ks[thr] = min(k, Xtr.shape[1])
    return ks, Xtr.shape[1]


def rmse(errs):
    errs = np.asarray(errs, dtype=float)
    return float(np.sqrt(np.mean(errs ** 2))) if len(errs) else np.nan


def main():
    t_start = time.perf_counter()
    print("=" * 100)
    print(f"PCA SWEEP [mode={MODE}]  (pool × whiten × n_components × alpha, parallel)")
    print("=" * 100, flush=True)

    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    grid = eh.merge_humidity(grid)
    grid = add_lag_bank(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    tw_train = tw[tw["ds_real"] < TEST_START_DATE].copy().reset_index(drop=True)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy().reset_index(drop=True)
    print(f"  train twilights: {len(tw_train)}, test: {len(tw_test)}", flush=True)

    if MODE == "ridge":
        # Pure-linear: no NBEATSx, no GPU. Anchor-only "predictions".
        print("  MODE=ridge: pure linear (no NBEATSx); target D=y-y[t-12h]", flush=True)
        tr_preds = get_all_preds_ridge(grid, tw_train, LEAD_TIMES_HOURS)
        te_preds = get_all_preds_ridge(grid, tw_test, LEAD_TIMES_HOURS)
    else:
        # ── Expensive step, run ONCE: NBEATSx + all preds (GPU/MPS) ──
        nf, hist, futr = eh.train_nbeats(grid)
        print("  predicting (train)...", flush=True)
        tr_preds = eh.get_all_preds(nf, grid, tw_train, hist, futr, LEAD_TIMES_HOURS)
        print("  predicting (test)...", flush=True)
        te_preds = eh.get_all_preds(nf, grid, tw_test, hist, futr, LEAD_TIMES_HOURS)
    print(f"  train preds={len(tr_preds)}, test preds={len(te_preds)}", flush=True)

    pool_names = os.environ.get("PCA_POOLS")
    pool_names = [p.strip() for p in pool_names.split(",")] if pool_names else list(POOLS)

    # ── Build per-pool feature matrices once (cheap, single process) ──
    DATA, auto_ks = {}, {}
    for name in pool_names:
        per_lead, used_cols = build_matrices(grid, tr_preds, te_preds, POOLS[name], LEAD_TIMES_HOURS)
        if not per_lead:
            print(f"  [skip] pool '{name}': no usable leads")
            continue
        DATA[name] = per_lead
        ks, n_feat = auto_k_for_pool(per_lead, VAR_THRESHOLDS)
        auto_ks[name] = (ks, n_feat)
        thr_str = ", ".join(f"{int(t*100)}%→k={ks[t]}" for t in VAR_THRESHOLDS)
        print(f"  pool '{name}': {n_feat} features ({len(used_cols)} present) | var {thr_str}")

    # ── Assemble the parallel grid: explicit k = 2..n_feat plus auto-k rows ──
    tasks = []
    ks_override = os.environ.get("PCA_KS")  # e.g. "1,2,3" pins the k-grid exactly
    for name, per_lead in DATA.items():
        ks, n_feat = auto_ks[name]
        if ks_override:
            k_grid = [int(x) for x in ks_override.split(",")]
            k_values = sorted({k for k in k_grid if 1 <= k <= n_feat})
        else:
            if n_feat <= 24:
                k_grid = list(range(2, n_feat + 1))       # dense for small pools
            else:
                # log-spaced k for wide banks (96 cols → ~12 points, not 95)
                k_grid = sorted(set(int(round(x)) for x in
                                    np.geomspace(2, n_feat, 12)))
            k_values = sorted(set(k_grid + list(ks.values())))
        for whiten in WHITEN_GRID:
            for k in k_values:
                # tag auto-k points so we can flag the variance-threshold elbows
                tag = next((f"var{int(t*100)}" for t, kv in ks.items() if kv == k), "")
                for alpha in ALPHA_GRID:
                    tasks.append((name, whiten, k, alpha, tag))
    print(f"\n  grid: {len(tasks)} (pool×whiten×k×alpha) points "
          f"× {len(LEAD_TIMES_HOURS)} leads", flush=True)

    n_workers = int(os.environ.get("PCA_WORKERS", min(16, (os.cpu_count() or 4) - 2)))
    print(f"  fanning out across {n_workers} CPU workers...", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(DATA,)) as pool:
        chunks = pool.map(_fit_one, tasks, chunksize=4)
    rows = [r for chunk in chunks for r in chunk]
    print(f"  done in {time.perf_counter()-t0:.1f}s ({len(rows)} error rows)", flush=True)

    df = pd.DataFrame(rows, columns=[
        "pool", "whiten", "k", "k_tag", "alpha", "lead", "season",
        "hum_avail", "error", "cumvar"])
    RESULTS_PATH.mkdir(exist_ok=True)
    suffix = f"_{MODE}" if MODE != "nbeats" else ""
    out_csv = RESULTS_PATH / f"pca_sweep_errors{suffix}.csv"
    summary_csv = RESULTS_PATH / f"pca_sweep_summary{suffix}.csv"
    df.to_csv(out_csv, index=False)

    # ── Aggregate: per config, ALL/Winter/Spring RMSE at key leads + WS9 ──
    cfg_keys = ["pool", "whiten", "k", "k_tag", "alpha"]
    summary = []
    for cfg, g in df.groupby(cfg_keys):
        rec = dict(zip(cfg_keys, cfg))
        for L in KEY_LEADS:
            gl = g[np.abs(g["lead"] - L) < 0.01]
            rec[f"{int(L)}h_ALL"] = rmse(gl["error"].values)
            for s in ("Winter", "Spring"):
                rec[f"{int(L)}h_{s}"] = rmse(gl[gl["season"] == s]["error"].values)
        rec["WS9"] = np.nanmean([rec.get("9h_Winter", np.nan), rec.get("9h_Spring", np.nan)])
        summary.append(rec)
    sdf = pd.DataFrame(summary).sort_values("WS9").reset_index(drop=True)
    sdf.to_csv(summary_csv, index=False)

    print(f"\n{'='*100}\nTOP 15 CONFIGS by WS9 (mean winter & spring 9h RMSE; lower=better)\n{'='*100}")
    show = ["pool", "whiten", "k", "k_tag", "alpha",
            "3h_ALL", "9h_ALL", "9h_Winter", "9h_Spring", "WS9"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(sdf[show].head(15).to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))

    # Best per pool, for the elbow story (how few PCs suffice).
    print(f"\n{'='*100}\nBEST CONFIG PER POOL\n{'='*100}")
    for name in DATA:
        b = sdf[sdf.pool == name].head(1)
        if len(b):
            r = b.iloc[0]
            tag = f"  ({r.k_tag})" if r.k_tag else ""
            print(f"  {name:6s}: k={int(r.k):2d} whiten={r.whiten} alpha={r.alpha:<4} "
                  f"3h_ALL={r['3h_ALL']:.3f} 9h_ALL={r['9h_ALL']:.3f} WS9={r.WS9:.3f}{tag}")

    print(f"\n  errors  -> {out_csv}")
    print(f"  summary -> {summary_csv}")
    print(f"  total wall time: {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
