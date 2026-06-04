"""Plot RMSE vs NBEATSx MLP width (log-log) from results/sweep_mlp.csv.

Usage:
    uv run --no-sync python forecast/plot_mlp.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH, FIGURES_PATH  # noqa: E402

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

KEY_LEADS = [0.5, 1.0, 3.0, 6.0, 12.0]


def main():
    df = pd.read_csv(RESULTS_PATH / "sweep_mlp.csv").sort_values("width")

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(KEY_LEADS)))

    for lead, color in zip(KEY_LEADS, cmap):
        ax.plot(df["width"], df[f"rmse_{lead}h"], "o-", color=color,
                label=f"{lead:g} h", markersize=6, linewidth=1.8)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MLP width (units per layer)")
    ax.set_ylabel("RMSE [$^\\circ$C]")
    ax.set_title("NBEATSx RMSE vs MLP width (max\\_steps=200, pre-2025 train)")

    ax.set_xticks(df["width"])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title="lead time", frameon=False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGURES_PATH / f"fig_rmse_vs_mlpwidth.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    print("\nRMSE@3h by width:")
    for _, r in df.iterrows():
        print(f"  width={int(r['width']):>5}  RMSE@3h={r['rmse_3.0h']:.3f}  fit={r['fit_s']:.0f}s")


if __name__ == "__main__":
    main()
