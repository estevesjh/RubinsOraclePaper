"""Plot RMSE vs NBEATSx max_steps (log-log) from results/sweep_maxsteps.csv.

Usage:
    uv run --no-sync python forecast/plot_maxsteps.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
from config import RESULTS_PATH, FIGURES_PATH

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

KEY_LEADS = [0.5, 1.0, 3.0, 6.0, 12.0]


def main():
    df = pd.read_csv(RESULTS_PATH / "sweep_maxsteps.csv").sort_values("max_steps")

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(KEY_LEADS)))

    for lead, color in zip(KEY_LEADS, cmap):
        col = f"rmse_{lead}h"
        ax.plot(df["max_steps"], df[col], "o-", color=color,
                label=f"{lead:g} h", markersize=6, linewidth=1.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_steps (training iterations)")
    ax.set_ylabel("RMSE [$^\\circ$C]")
    ax.set_title("NBEATSx RMSE vs training steps (width=16, pre-2025 train)")

    ax.set_xticks(df["max_steps"])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title="lead time", frameon=False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGURES_PATH / f"fig_rmse_vs_maxsteps.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    # Console summary: relative RMSE reduction from first to last step
    print("\nRMSE reduction (first -> last max_steps):")
    s0, s1 = df.iloc[0], df.iloc[-1]
    for lead in KEY_LEADS:
        col = f"rmse_{lead}h"
        pct = 100 * (s0[col] - s1[col]) / s0[col]
        print(f"  {lead:>4g} h: {s0[col]:.3f} -> {s1[col]:.3f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
