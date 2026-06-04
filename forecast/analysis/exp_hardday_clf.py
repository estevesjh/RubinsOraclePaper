"""Can we CLASSIFY synoptic hard days at issue time?

Earlier (docs/winter_spring_plan.md §3.7) a temperature-only causal flag failed:
the day-to-day twilight swing has ~0 autocorrelation, so yesterday's temperature
history doesn't foreshadow today's swing. But we now have humidity + wind, which
may sense an approaching synoptic change that temperature alone cannot.

Target (causal, deployable): for each twilight, is the change from the PREVIOUS
twilight to THIS one a "hard" night, |dT_tw| > Q75? At the morning issue time
(9 h before twilight) the previous twilight is known but today's is not, so this
is a genuine prediction. Label threshold set on pre-2025 only.

Features at the 9 h issue row (all causal / issue-time):
  TEMP : last_std_24h, DTR, trend_solar_2h, dmean_3d, doy_sin, doy_cos
  +HW  : humidity, hum_std_24h, windSpeed, wind_u_par, wind_u_perp, wind_u_par_NE

Compares TEMP-only vs TEMP+HW classifiers (logistic + gradient boosting), AUC on
2025 overall and per season. Strict pre-2025 train.

Usage:
    uv run --no-sync python forecast/exp_hardday_clf.py
"""

import os, sys, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
import feature_sweep as fs
import exp_humidity as eh
from config import TEST_START_DATE

TEMP_FEATS = ["last_std_24h", "DTR", "trend_solar_2h", "dmean_3d", "doy_sin", "doy_cos"]
HW_FEATS = ["humidity", "hum_std_24h", "windSpeed",
            "wind_u_par", "wind_u_perp", "wind_u_par_NE"]
ISSUE_LEAD_H = 9.0


def season_of(ts):
    m = pd.Timestamp(ts).month
    return {12: "Summer", 1: "Summer", 2: "Summer", 3: "Fall", 4: "Fall", 5: "Fall",
            6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring",
            11: "Spring"}[m]


def build_table(grid, tw):
    """Per-twilight: issue-time features + hard-day label |dT_tw|>Q75 (causal)."""
    tw = tw.sort_values("ds_real").reset_index(drop=True)
    y_tw = tw["y_actual"].values
    dT = np.abs(np.concatenate([[np.nan], np.diff(y_tw)]))  # |this tw - prev tw|
    off = run.lead_hours_to_steps(ISSUE_LEAD_H)
    all_feats = TEMP_FEATS + HW_FEATS
    rows = []
    for i, (_, ev) in enumerate(tw.iterrows()):
        iss = int(ev["grid_idx"]) - off
        if iss < 0 or iss >= len(grid) or np.isnan(dT[i]):
            continue
        rec = {c: grid[c].iloc[iss] for c in all_feats if c in grid.columns}
        if any(pd.isna(v) for v in rec.values()):
            continue
        rec["dT_tw"] = dT[i]
        rec["ds_real"] = ev["ds_real"]
        rec["season"] = season_of(ev["ds_real"])
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(df, feat_cols, label):
    tr = df[pd.to_datetime(df["ds_real"]) < TEST_START_DATE]
    te = df[pd.to_datetime(df["ds_real"]) >= TEST_START_DATE]
    thr = tr["dT_tw"].quantile(0.75)            # hard-day threshold, pre-2025
    ytr = (tr["dT_tw"] > thr).astype(int).values
    yte = (te["dT_tw"] > thr).astype(int).values
    Xtr = tr[feat_cols].values; Xte = te[feat_cols].values
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    out = {}
    # logistic
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr_s, ytr)
    p_lr = lr.predict_proba(Xte_s)[:, 1]
    # gradient boosting (nonlinear)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=0)
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]

    for name, p in [("logreg", p_lr), ("gbm", p_gb)]:
        auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
        # operating point at the same 25% positive rate
        cut = np.quantile(p, 0.75)
        pred = (p >= cut).astype(int)
        prec = precision_score(yte, pred, zero_division=0)
        rec = recall_score(yte, pred, zero_division=0)
        out[name] = dict(auc=auc, precision=prec, recall=rec)
        # per-season AUC
        per = {}
        for s in ["Summer", "Fall", "Winter", "Spring"]:
            m = (te["season"] == s).values
            if m.sum() > 10 and len(np.unique(yte[m])) > 1:
                per[s] = roc_auc_score(yte[m], p[m])
        out[name]["per_season_auc"] = per
    base_rate = yte.mean()
    return out, base_rate, thr


def main():
    print("=" * 90)
    print("HARD-DAY CLASSIFIER: can humidity+wind see the synoptic swing coming?")
    print("=" * 90)
    grid, _ = run.load_and_prepare()
    grid = fs.compute_all_features(grid)
    grid = eh.merge_humidity(grid)
    tw = run.find_twilight_targets(grid)
    tw["ds_real"] = pd.to_datetime(tw["ds_real"])
    df = build_table(grid, tw)
    n_tr = (pd.to_datetime(df["ds_real"]) < TEST_START_DATE).sum()
    n_te = (pd.to_datetime(df["ds_real"]) >= TEST_START_DATE).sum()
    print(f"  twilights usable: train={n_tr}, test={n_te}")

    for label, feats in [("TEMP-only", TEMP_FEATS), ("TEMP+HW", TEMP_FEATS + HW_FEATS)]:
        feats = [c for c in feats if c in df.columns]
        out, base, thr = evaluate(df, feats, label)
        print(f"\n=== {label}  (hard-day = |dT_tw|>{thr:.2f}C, test base rate={base:.0%}) ===")
        for clf in ["logreg", "gbm"]:
            o = out[clf]
            ps = "  ".join(f"{s}={v:.2f}" for s, v in o["per_season_auc"].items())
            print(f"  {clf:7s} AUC={o['auc']:.3f}  prec={o['precision']:.2f} "
                  f"recall={o['recall']:.2f} | per-season AUC: {ps}")
    print("\n(AUC 0.5 = no skill; >0.65 = useful; >0.75 = strong)")


if __name__ == "__main__":
    main()
