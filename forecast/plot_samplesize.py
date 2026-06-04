"""Plot RMSE vs training sample size (NBEATSx+Ridge).

Usage:
    uv run --no-sync python forecast/plot_samplesize.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH, FIGURES_PATH

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_PATH / "sweep_samplesize.csv").sort_values("n_train")
    ends = pd.to_datetime(df["window_end"], format="ISO8601")

    fig, ax = plt.subplots(figsize=(8, 5))

    leads = [12.0, 9.0, 6.0, 3.0]
    colors = {"12.0": "#023e8a", "9.0": "#9467bd", "6.0": "#ffb703",
              "3.0": "#17becf"}

    n = df["n_train"].values

    # Power-law extrapolation range (extend to ~40k = Jan 2026 survey start)
    n_extrap = np.linspace(n.min(), 40000, 100)

    for L in leads:
        col = f"rmse_{L}h"
        if col not in df.columns:
            continue
        vals = df[col].values
        ax.plot(n, vals, "o-", color=colors[str(L)],
                label=f"{L:g} h", markersize=5, linewidth=2)

        # Power-law fit on last 6 points: log(rmse) = a * log(n) + b
        fit_n = n[-6:]
        fit_v = vals[-6:]
        valid = ~np.isnan(fit_v) & (fit_v > 0)
        if valid.sum() >= 3:
            a, b = np.polyfit(np.log(fit_n[valid]), np.log(fit_v[valid]), 1)
            fit_line = np.exp(b) * n_extrap**a
            ax.plot(n_extrap, fit_line, "--", color=colors[str(L)],
                    linewidth=1.2, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Y ticks and limits
    # Compute y limits from data
    all_vals = []
    for L in leads:
        col = f"rmse_{L}h"
        if col in df.columns:
            all_vals.extend(df[col].dropna().tolist())
    ymax = max(all_vals) * 1.1
    ymin = 0.7
    yticks_major = [0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
    yticks_major = [v for v in yticks_major if ymin <= v <= ymax]
    yticks_minor = [0.8, 0.9, 1.2, 1.7, 2.5, 3.5]
    yticks_minor = [v for v in yticks_minor if ymin <= v <= ymax]
    ax.set_yticks(yticks_major)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks_major], fontsize=9)
    ax.set_yticks(yticks_minor, minor=True)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(ymin, ymax)

    # Grid: solid for major, light dotted for minor
    for ref in yticks_major:
        ax.axhline(ref, color="gray", ls="-", lw=0.5, alpha=0.4)
    for ref in yticks_minor:
        ax.axhline(ref, color="gray", ls=":", lw=0.3, alpha=0.3)

    ax.grid(False)
    ax.set_xlabel("training window end date")
    ax.set_ylabel("RMSE [°C]")
    ax.set_title("NBEATSx+Ridge RMSE vs training data volume")
    ax.legend(title="lead time", frameon=False, loc="upper right")

    # Extra xticks at Southern-hemisphere season changes + survey start
    # ~48 rows/day from Oct 2023: Mar 2025~25k, Jun 2025~29k, Sep 2025~34k, Jan 2026~40k
    extrap_n = np.array([25000, 29000, 34000, 40000])
    all_n = np.concatenate([n, extrap_n])
    extrap_dates = ["Mar\n2025", "Jun\n2025", "Sep\n2025", "Jan\n2026"]
    all_dates = list(ends.dt.strftime("%b\n%Y")) + extrap_dates
    all_sizes = [f"{int(v/1000)}k" for v in all_n]

    # Bottom x: dates
    ax.set_xticks(all_n)
    ax.set_xticklabels(all_dates, fontsize=8, ha="center")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(n.min() * 0.9, 44000)

    # Top x: sample size
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xscale("log")
    ax2.set_xticks(all_n)
    ax2.set_xticklabels(all_sizes, fontsize=8)
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_xlabel("sample size")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_PATH / f"fig_rmse_vs_samplesize.{ext}", dpi=200, bbox_inches="tight")
    print("Saved fig_rmse_vs_samplesize")


if __name__ == "__main__":
    main()
