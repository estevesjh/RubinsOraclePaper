"""Retroactive correlation analysis on the in-progress feature sweep.

Reads results/sweep_round_N.csv (the per-round results saved by feature_sweep.py)
and reports: for each round, what the top-3 candidates are, their pairwise
Pearson |r| on the grid, and which ones the new "decorrelated co-add" rule
would have accepted in addition to the round's winner.

This is offline — no model training, just CSV + grid columns.
"""

import os
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run
from feature_sweep import compute_all_features
from config import RESULTS_PATH

CORR_THRESHOLD = 0.7
TOP_K = 3


def main():
    print("Building grid + features for correlation lookup...")
    grid, _ = run.load_and_prepare()
    grid = compute_all_features(grid)

    round_files = sorted(RESULTS_PATH.glob("sweep_round_*.csv"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if not round_files:
        print("No sweep_round_*.csv files found. Is the sweep running?")
        return

    print(f"\nFound {len(round_files)} round files.")
    print(f"Threshold: |r| < {CORR_THRESHOLD}, top-K = {TOP_K}\n")

    for rf in round_files:
        rnd = int(rf.stem.split("_")[-1])
        df = pd.read_csv(rf).dropna(subset=["rmse"])
        if len(df) < 1:
            continue
        df = df.sort_values("rmse").reset_index(drop=True)
        top = df.head(TOP_K)
        print("=" * 70)
        print(f"Round {rnd}: {len(df)} valid candidates evaluated")
        print(f"  Top-{len(top)}:")
        for _, r in top.iterrows():
            print(f"    {r['feature']:>26}  RMSE={r['rmse']:.4f}")

        if len(top) < 2:
            print("  (only one candidate — nothing to gate)")
            continue

        winner = top.iloc[0]["feature"]
        if winner not in grid.columns:
            print(f"  WARN: winner '{winner}' not in grid — skipping")
            continue

        win_col = grid[winner].values
        accepted = [winner]
        accepted_cols = [win_col]

        print(f"\n  Co-add gate (vs '{winner}'):")
        for _, r in top.iloc[1:].iterrows():
            feat = r["feature"]
            if feat not in grid.columns:
                print(f"    {feat:>26}  not in grid")
                continue
            col = grid[feat].values
            max_abs_r = 0.0
            r_vs_winner = np.nan
            for i, ref in enumerate(accepted_cols):
                mask = ~(np.isnan(col) | np.isnan(ref))
                if mask.sum() < 100:
                    continue
                rho = np.corrcoef(col[mask], ref[mask])[0, 1]
                if np.isfinite(rho):
                    if i == 0:
                        r_vs_winner = rho
                    max_abs_r = max(max_abs_r, abs(rho))
            tag = "ACCEPT" if max_abs_r < CORR_THRESHOLD else "skip(corr)"
            print(f"    {feat:>26}  r_vs_winner={r_vs_winner:+.2f}  "
                  f"max|r|={max_abs_r:.2f}  → {tag}")
            if max_abs_r < CORR_THRESHOLD:
                accepted.append(feat)
                accepted_cols.append(col)

        if len(accepted) > 1:
            print(f"\n  → Would add: {accepted}")
        else:
            print(f"\n  → Would add: ['{winner}'] only")

    print("\n" + "=" * 70)
    print("Done. Note: this is retroactive — current sweep used winner-only logic.")
    print("Updated feature_sweep.py applies this gate to future runs.")


if __name__ == "__main__":
    main()
