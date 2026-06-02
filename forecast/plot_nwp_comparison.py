"""Plot RMSE and p95 comparison: NBEATSx with vs without MeteoBlue NWP.

Requires:
  - results/paper_results_diff_nwp.csv (from: python forecast/run.py --mb --name NBEATSx_nwp)
  - results/paper_results_diff.csv     (from: python forecast/run.py --name NBEATSx_local)
  - results/paper_results_final.csv    (from: python forecast/build_final.py)

Usage:
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/plot_nwp_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH, FIGURES_PATH

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def blend_with_linear(nb_df, lr_df):
    """Blend NBEATSx with Linear (per-lead optimal alpha)."""
    nb_df = nb_df.copy()
    nb_df["twilight_time"] = pd.to_datetime(nb_df["twilight_time"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S.000")
    merged = nb_df.merge(lr_df[["twilight_time", "lead_time_hours", "forecast_temp"]],
                         on=["twilight_time", "lead_time_hours"], suffixes=("_nb", "_lr"), how="inner")
    rows = []
    for lh in sorted(merged["lead_time_hours"].unique()):
        sub = merged[merged["lead_time_hours"] == lh]
        T_nb, T_lr, T_act = sub["forecast_temp_nb"].values, sub["forecast_temp_lr"].values, sub["actual_temp"].values
        res = minimize_scalar(lambda a: np.sqrt(((T_act - (a * T_lr + (1 - a) * T_nb)) ** 2).mean()),
                              bounds=(0, 1), method="bounded")
        T_blend = res.x * T_lr + (1 - res.x) * T_nb
        for i in range(len(sub)):
            rows.append({"lead_time_hours": lh, "error": T_act[i] - T_blend[i]})
    return pd.DataFrame(rows)


def bootstrap_stat(errors, stat_fn, n_boot=200):
    """Bootstrap confidence interval for a statistic."""
    vals = []
    n = len(errors)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        vals.append(stat_fn(errors[idx]))
    return np.percentile(vals, [5, 95])


def main():
    # Load results
    nwp_file = RESULTS_PATH / "paper_results_diff_nwp.csv"
    local_file = RESULTS_PATH / "paper_results_diff.csv"
    final_file = RESULTS_PATH / "paper_results_final.csv"

    if not nwp_file.exists():
        print(f"Missing {nwp_file}. Run: python forecast/run.py --mb --name NBEATSx_nwp")
        return
    if not local_file.exists():
        print(f"Missing {local_file}. Run: python forecast/run.py --name NBEATSx_local")
        return

    nwp = pd.read_csv(nwp_file)
    local = pd.read_csv(local_file)
    final = pd.read_csv(final_file)

    nwp_nb = nwp[nwp["model"] == "NBEATSx-Diff"]
    local_nb = local[local["model"] == "NBEATSx-Diff"]
    lr = final[final["model"] == "Linear"]

    # Blend each with Linear
    print("Blending NWP + Linear...")
    blend_nwp = blend_with_linear(nwp_nb, lr)
    print("Blending Local + Linear...")
    blend_local = blend_with_linear(local_nb, lr)

    leads = sorted(nwp_nb["lead_time_hours"].unique())

    rmse_fn = lambda e: np.sqrt((e ** 2).mean())
    p95_fn = lambda e: np.percentile(np.abs(e), 95)

    rmse_bn, lo_bn, hi_bn = [], [], []
    rmse_bl, lo_bl, hi_bl = [], [], []
    p95_bn, p95lo_bn, p95hi_bn = [], [], []
    p95_bl, p95lo_bl, p95hi_bl = [], [], []

    for lh in leads:
        err_nwp = blend_nwp[blend_nwp["lead_time_hours"] == lh]["error"].values
        err_local = blend_local[blend_local["lead_time_hours"] == lh]["error"].values

        rmse_bn.append(rmse_fn(err_nwp))
        rmse_bl.append(rmse_fn(err_local))
        lo, hi = bootstrap_stat(err_nwp, rmse_fn)
        lo_bn.append(lo); hi_bn.append(hi)
        lo, hi = bootstrap_stat(err_local, rmse_fn)
        lo_bl.append(lo); hi_bl.append(hi)

        p95_bn.append(p95_fn(err_nwp))
        p95_bl.append(p95_fn(err_local))
        lo, hi = bootstrap_stat(err_nwp, p95_fn)
        p95lo_bn.append(lo); p95hi_bn.append(hi)
        lo, hi = bootstrap_stat(err_local, p95_fn)
        p95lo_bl.append(lo); p95hi_bl.append(hi)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Impact of Adding MeteoBlue NWP Forecast as Exogenous Feature", fontsize=14, y=0.98)

    # Left: RMSE
    ax1.fill_between(leads, lo_bn, hi_bn, color="#023e8a", alpha=0.15)
    ax1.fill_between(leads, lo_bl, hi_bl, color="#e63946", alpha=0.15)
    ax1.plot(leads, rmse_bn, "o-", color="#023e8a", lw=2.5, label="NBEATSx-Blend + NWP")
    ax1.plot(leads, rmse_bl, "s-", color="#e63946", lw=2, label="NBEATSx-Blend Local")
    ax1.axhline(1.0, color="orange", ls="--", alpha=0.5)
    ax1.set_xlabel("Lead time (hours)", fontsize=12)
    ax1.set_ylabel("RMSE (°C)", fontsize=12)
    ax1.set_title("RMSE vs Lead Time (90% CI)", fontsize=12)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.set_xlim(0, 13)
    ax1.set_ylim(0.2, 1.6)
    ax1.grid(True, alpha=0.3)

    # Right: p95
    ax2.fill_between(leads, p95lo_bn, p95hi_bn, color="#023e8a", alpha=0.15)
    ax2.fill_between(leads, p95lo_bl, p95hi_bl, color="#e63946", alpha=0.15)
    ax2.plot(leads, p95_bn, "o-", color="#023e8a", lw=2.5, label="NBEATSx-Blend + NWP")
    ax2.plot(leads, p95_bl, "s-", color="#e63946", lw=2, label="NBEATSx-Blend Local")
    ax2.axhline(1.0, color="orange", ls="--", alpha=0.5)
    ax2.set_xlabel("Lead time (hours)", fontsize=12)
    ax2.set_ylabel("|Error| 95th percentile (°C)", fontsize=12)
    ax2.set_title("95th Percentile |Error| vs Lead Time (90% CI)", fontsize=12)
    ax2.legend(fontsize=11, loc="lower right")
    ax2.set_xlim(0, 13)
    ax2.set_ylim(0.5, 3.2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_PATH / "fig_rmse_nwp_vs_local.png", dpi=150)
    fig.savefig(FIGURES_PATH / "fig_rmse_nwp_vs_local.pdf", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_PATH / 'fig_rmse_nwp_vs_local.png'}")

    # Print table
    print(f"\n{'Lead':>5} {'Blend+NWP':>10} {'Blend Local':>12} {'p95 NWP':>8} {'p95 Local':>10}")
    print("-" * 48)
    for i, lh in enumerate(leads):
        if lh in [1.0, 3.0, 6.0, 9.0, 12.0]:
            print(f"{lh:5.0f} {rmse_bn[i]:10.3f} {rmse_bl[i]:12.3f} {p95_bn[i]:8.2f} {p95_bl[i]:10.2f}")


if __name__ == "__main__":
    main()
