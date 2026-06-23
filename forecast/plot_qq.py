"""Q-Q plot of NBEATSx+Ridge residuals vs normal, per lead time."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH, FIGURES_PATH

FIGURES_PATH.mkdir(parents=True, exist_ok=True)
LEADS = [3.0, 6.0, 9.0, 12.0]
COLORS = {3.0: "#17becf", 6.0: "#ffb703", 9.0: "#9467bd", 12.0: "#023e8a"}


def main():
    df = pd.read_csv(RESULTS_PATH / "paper_results_diff.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, L in zip(axes.flat, LEADS):
        e = df[np.abs(df["lead_time_hours"] - L) < 0.01]["error"].values
        e = e[np.isfinite(e)]
        (osm, osr), (slope, inter, r) = stats.probplot(e, dist="norm")
        ax.scatter(osm, osr, s=14, color=COLORS[L], alpha=0.6, edgecolors="none")
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, np.array(lim) * slope + inter, "k--", lw=1, alpha=0.7)
        W, p = stats.shapiro(e)
        ku = stats.kurtosis(e)
        ax.set_title(f"{L:g} h  (Shapiro p={p:.1e}, exc.kurt={ku:.2f})", fontsize=11)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Residual quantiles [°C]")
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle("NBEATSx+Ridge residual Q-Q vs Normal", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_PATH / f"fig_qq_residuals.{ext}", dpi=200, bbox_inches="tight")
    print("Saved fig_qq_residuals")


if __name__ == "__main__":
    main()
