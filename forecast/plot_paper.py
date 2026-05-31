"""Generate ALL paper figures (fig0–fig8 + table1) from paper_results_final.csv.

Models: NBEATSx-Blend (labeled NBEATSx-Ridge in CSV), Linear, RF, MLP, Prophet, MeteoBlue, Persistence.

Usage:
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/plot_paper.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH, FIGURES_PATH, DATA_PATH

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")

# Models and colors
MODELS = ["NBEATSx-Ridge", "Linear", "RandomForest", "MLP", "Prophet", "Persistence"]
COLORS = {
    "NBEATSx-Ridge": "#023e8a",
    "Linear": "#e377c2",
    "RandomForest": "#7f7f7f",
    "MLP": "#bcbd22",
    "Prophet": "#00b4d8",
    "MeteoBlue": "#e63946",
    "Persistence": "#d62728",
}
LABELS = {
    "NBEATSx-Ridge": "NBEATSx-Blend",
    "RandomForest": "Random Forest",
}


def load_results():
    df = pd.read_csv(RESULTS_PATH / "paper_results_final.csv")
    df["twilight_time"] = pd.to_datetime(df["twilight_time"], format="mixed")
    df["abs_error"] = df["error"].abs()
    return df


def get_label(model):
    return LABELS.get(model, model)


# ── Fig 0: Dataset overview with midday forecast ─────────────────────────


def fig0_dataset_overview(df):
    """Dataset overview showing temperature series + forecast at midday (SolarTime=0.25)."""
    print("  Generating fig0...")

    raw = pd.read_csv(DATA_PATH, comment="#", low_memory=False)
    raw["ds"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None)
    raw["y"] = raw["mean"]

    # Pick a representative 5-day window in 2025
    window_start = pd.Timestamp("2025-06-15")
    window_end = pd.Timestamp("2025-06-20")
    window = raw[(raw["ds"] >= window_start) & (raw["ds"] <= window_end)]

    # Get forecasts for this window
    nb = df[(df["model"] == "NBEATSx-Ridge")]
    nb_window = nb[(nb["twilight_time"] >= window_start) & (nb["twilight_time"] <= window_end)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: full year 2025
    year = raw[(raw["ds"] >= "2025-01-01") & (raw["ds"] <= "2025-12-31")]
    axes[0].plot(year["ds"], year["y"], color="#1f77b4", lw=0.3, alpha=0.7)
    axes[0].axvspan(window_start, window_end, alpha=0.15, color="orange")
    axes[0].set_ylabel("Temperature (°C)", fontsize=12)
    axes[0].set_title("Ambient Temperature at Cerro Pachón (2025)", fontsize=13)

    # Bottom: 5-day window with forecasts
    axes[1].plot(window["ds"], window["y"], color="#1f77b4", lw=1.5, label="Observed")
    if len(nb_window) > 0:
        axes[1].scatter(nb_window["twilight_time"], nb_window["forecast_temp"],
                       color="#ff7f0e", s=60, zorder=5, marker="*",
                       label="NBEATSx-Blend forecast (midday issue)")
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("Temperature (°C)", fontsize=12)
    axes[1].set_title("5-day Detail with Twilight Forecasts", fontsize=12)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig0_dataset_overview.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig0_dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 2: RMSE heatmap ──────────────────────────────────────────────────


def fig2_rmse_heatmap(df):
    """RMSE by model and lead time heatmap."""
    print("  Generating fig2...")

    lead_bins = [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]
    models_order = ["NBEATSx-Ridge", "Prophet", "MLP", "RandomForest", "Linear", "Persistence"]

    rmse_matrix = []
    model_labels = []
    for m in models_order:
        mdf = df[df["model"] == m]
        if len(mdf) == 0:
            continue
        row = []
        for lh in lead_bins:
            sub = mdf[(mdf["lead_time_hours"] - lh).abs() < 0.25]
            if len(sub) > 0:
                row.append(np.sqrt((sub["error"] ** 2).mean()))
            else:
                row.append(np.nan)
        rmse_matrix.append(row)
        model_labels.append(get_label(m))

    rmse_arr = np.array(rmse_matrix)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(rmse_arr, cmap="RdYlBu_r", aspect="auto", vmin=0.3, vmax=1.8)

    ax.set_xticks(range(len(lead_bins)))
    ax.set_xticklabels([f"{l:.1f}" for l in lead_bins], fontsize=12)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=12)
    ax.set_xlabel("Hours from sunset (h)", fontsize=13)
    ax.set_title("RMSE by Model and Lead Time", fontsize=14)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(lead_bins)):
            val = rmse_arr[i, j]
            if not np.isnan(val):
                color = "white" if val > 1.2 or val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=13, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="RMSE (°C)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig2_rmse_heatmap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig2_rmse_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 3: RMSE vs lead time curves ─────────────────────────────────────


def fig3_rmse_vs_lead(df):
    """RMSE vs lead time for all models."""
    print("  Generating fig3...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for m in MODELS:
        mdf = df[df["model"] == m]
        if len(mdf) == 0:
            continue
        leads = sorted(mdf["lead_time_hours"].unique())
        rmse_vals = []
        lead_vals = []
        for lh in leads:
            sub = mdf[mdf["lead_time_hours"] == lh]
            if len(sub) >= 10:
                rmse_vals.append(np.sqrt((sub["error"] ** 2).mean()))
                lead_vals.append(lh)
        if lead_vals:
            ax.plot(lead_vals, rmse_vals, "o-", color=COLORS.get(m, "gray"),
                   lw=2, markersize=4, label=get_label(m))

    ax.axhline(1.0, color="orange", ls="--", alpha=0.5, label="1°C target")
    ax.set_xlabel("Lead time (hours)", fontsize=12)
    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_title("RMSE vs Lead Time", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig3_rmse_vs_lead_time.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig3_rmse_vs_lead_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 4: Error histograms at 3h ───────────────────────────────────────


def fig4_error_histograms(df):
    """Error distribution at 3h for key models."""
    print("  Generating fig4...")

    models_3h = ["NBEATSx-Ridge", "MLP", "Prophet", "Persistence"]
    at_3h = df[(df["lead_time_hours"] - 3.0).abs() < 0.25]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, m in enumerate(models_3h):
        ax = axes[i]
        sub = at_3h[at_3h["model"] == m]
        if len(sub) == 0:
            ax.set_title(get_label(m))
            continue

        rmse = np.sqrt((sub["error"] ** 2).mean())
        bias = sub["error"].mean()

        ax.hist(sub["error"], bins=30, color=COLORS.get(m, "gray"),
                alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(bias, color="red", ls="-", lw=1.5)
        ax.set_title(f"{get_label(m)} (RMSE={rmse:.2f}°C, N={len(sub)})", fontsize=11)
        ax.set_xlabel("Error (°C)")
        ax.set_ylabel("Count")
        ax.set_xlim(-5, 5)

    fig.suptitle("Error Distribution at 3h Lead Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig4_error_histograms_3h.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig4_error_histograms_3h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 5: CDF of absolute error at 3h ──────────────────────────────────


def fig5_cdf(df):
    """CDF of absolute error at 3h for all models."""
    print("  Generating fig5...")

    at_3h = df[(df["lead_time_hours"] - 3.0).abs() < 0.25]

    fig, ax = plt.subplots(figsize=(9, 6))

    for m in MODELS:
        sub = at_3h[at_3h["model"] == m]
        if len(sub) < 10:
            continue
        sorted_err = np.sort(sub["abs_error"].values)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf * 100, color=COLORS.get(m, "gray"),
               lw=2.5, label=f"{get_label(m)} (N={len(sub)})")

    ax.axvline(1.0, color="gray", ls="--", alpha=0.7, lw=1)
    ax.set_xlabel("Absolute Error (°C)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage (%)", fontsize=12)
    ax.set_title("CDF of Absolute Error at 3h Lead Time", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig5_cdf_error.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig5_cdf_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 6: Seasonal analysis ─────────────────────────────────────────────


def fig6_seasonal(df):
    """Seasonal performance breakdown at 3h lead."""
    print("  Generating fig6...")

    nb = df[(df["model"] == "NBEATSx-Ridge") & ((df["lead_time_hours"] - 3.0).abs() < 0.25)].copy()
    nb["month"] = nb["twilight_time"].dt.month

    def get_season(m):
        if m in [12, 1, 2]: return "Summer"
        elif m in [3, 4, 5]: return "Fall"
        elif m in [6, 7, 8]: return "Winter"
        else: return "Spring"

    nb["season"] = nb["month"].apply(get_season)
    season_order = ["Summer", "Fall", "Winter", "Spring"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # RMSE by season
    rmse_s = [np.sqrt((nb[nb["season"] == s]["error"] ** 2).mean()) for s in season_order]
    axes[0, 0].bar(season_order, rmse_s, color=["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"])
    axes[0, 0].axhline(np.sqrt((nb["error"] ** 2).mean()), color="k", ls="--", lw=1)
    axes[0, 0].set_ylabel("RMSE (°C)")
    axes[0, 0].set_title("RMSE by Season")

    # Bias by season
    bias_s = [nb[nb["season"] == s]["error"].mean() for s in season_order]
    axes[0, 1].bar(season_order, bias_s, color=["green" if b > 0 else "red" for b in bias_s], alpha=0.7)
    axes[0, 1].axhline(0, color="k", lw=0.5)
    axes[0, 1].set_ylabel("Bias (°C)")
    axes[0, 1].set_title("Bias by Season")

    # % < 1C by season
    pct_s = [(nb[nb["season"] == s]["abs_error"] < 1.0).mean() * 100 for s in season_order]
    axes[1, 0].bar(season_order, pct_s, color=["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"])
    axes[1, 0].set_ylabel("% < 1°C")
    axes[1, 0].set_title("Accuracy by Season")
    axes[1, 0].set_ylim(0, 100)

    # Error vs actual temp (trend dependence)
    axes[1, 1].scatter(nb["actual_temp"], nb["error"], alpha=0.4, s=15, c="#1f77b4")
    axes[1, 1].axhline(0, color="k", ls="--", lw=1)
    axes[1, 1].set_xlabel("Actual Temp (°C)")
    axes[1, 1].set_ylabel("Error (°C)")
    axes[1, 1].set_title("Error vs Temperature")

    fig.suptitle("NBEATSx-Blend Seasonal Analysis (3h Lead)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig6_seasonal_trend_analysis.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig6_seasonal_trend_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fig 8: Comparison scatter ─────────────────────────────────────────────


def fig8_comparison(df):
    """Compare NBEATSx-Blend vs Prophet vs MeteoBlue scatter plots."""
    print("  Generating fig8...")

    # NBEATSx at 12h, Prophet at 6h, MeteoBlue all available
    nbeats = df[(df["model"] == "NBEATSx-Ridge") & ((df["lead_time_hours"] - 12.0).abs() < 0.25)]
    prophet = df[(df["model"] == "Prophet") & ((df["lead_time_hours"] - 6.0).abs() < 0.5)]
    meteoblue = df[df["model"] == "MeteoBlue"]

    datasets = [
        (nbeats, "NBEATSx-Blend (12h)", COLORS["NBEATSx-Ridge"]),
        (prophet, "Prophet (6h)", COLORS["Prophet"]),
        (meteoblue, "MeteoBlue", COLORS["MeteoBlue"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [1.5, 1]})

    for col, (data, title, color) in enumerate(datasets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if len(data) == 0:
            ax_top.set_title(title)
            continue

        x = data["actual_temp"].values
        y = data["forecast_temp"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Scatter
        ax_top.scatter(x, y, alpha=0.4, s=20, color=color)
        lims = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
        ax_top.plot(lims, lims, "k--", lw=1)
        ax_top.set_xlim(lims)
        ax_top.set_ylim(lims)
        ax_top.set_title(title, fontsize=12)
        if col == 0:
            ax_top.set_ylabel("Forecast Temp (°C)")

        # Stats
        from scipy import stats
        slope, intercept, r, _, _ = stats.linregress(x, y)
        bias = (y - x).mean()
        std = (y - x).std()
        ax_top.text(0.05, 0.95, f"N = {len(x)}\nSlope = {slope:.2f}\nBias = {bias:.2f}°C\nStd = {std:.2f}°C\nR² = {r**2:.2f}",
                   transform=ax_top.transAxes, va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Residuals
        residuals = y - x
        ax_bot.scatter(x, residuals, alpha=0.4, s=20, color=color)
        ax_bot.axhline(0, color="k", ls="--", lw=1)
        ax_bot.set_xlabel("Actual Temp (°C)")
        if col == 0:
            ax_bot.set_ylabel("Residual (°C)")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "fig8_comparison.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "fig8_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Table 1: Summary at 3h ───────────────────────────────────────────────


def table1_summary(df):
    """Summary metrics at 3h lead time."""
    print("  Generating table1...")

    at_3h = df[(df["lead_time_hours"] - 3.0).abs() < 0.25]
    models_order = ["Persistence", "Linear", "RandomForest", "MLP", "Prophet", "NBEATSx-Ridge"]

    rows = []
    for m in models_order:
        sub = at_3h[at_3h["model"] == m]
        if len(sub) == 0:
            continue
        rmse = np.sqrt((sub["error"] ** 2).mean())
        mae = sub["abs_error"].mean()
        bias = sub["error"].mean()
        pct_1 = (sub["abs_error"] < 1.0).mean() * 100
        rows.append({
            "Model": get_label(m),
            "RMSE": f"{rmse:.2f}",
            "MAE": f"{mae:.2f}",
            "Bias": f"{bias:+.2f}",
            "<1°C": f"{pct_1:.0f}%",
            "N": len(sub),
        })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))
    table_df.to_csv(FIGURES_PATH / "table1_summary_3h_v2.csv", index=False)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)

    df = load_results()
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Total rows: {len(df)}")

    fig0_dataset_overview(df)
    fig2_rmse_heatmap(df)
    fig3_rmse_vs_lead(df)
    fig4_error_histograms(df)
    fig5_cdf(df)
    fig6_seasonal(df)
    fig8_comparison(df)
    table1_summary(df)

    print(f"\nAll figures saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
