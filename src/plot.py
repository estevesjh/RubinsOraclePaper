"""Generate figures for the twilight forecasting paper.

Includes NBEATSx-Ridge model from paper_results_final.csv.

Figures (matching main.tex):
- Fig 0: Dataset overview (full year + representative week)
- Fig 2: RMSE heatmap (model x lead time)
- Fig 5: CDF of absolute error
- Fig 6: Seasonal and temperature trend analysis
- Fig 7: Rate (dT/dt) forecast results
- Fig 8: Model comparison (NBEATSx-Ridge vs Prophet vs MeteoBlue)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PATH, FIGURES_PATH, RESULTS_PATH

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "Persistence": "#8c564b",  # Brown - baseline
    "Persistence-Twilight": "#d62728",  # Red - twilight baseline
    "Linear": "#e377c2",  # Pink
    "RandomForest": "#7f7f7f",  # Gray
    "MLP": "#bcbd22",  # Olive/yellow-green
    "NBEATSx-Oracle": "#9467bd",  # Purple
    "NBEATSx-Ridge": "#17becf",  # Cyan - keep unchanged
}
MODEL_ORDER = [
    "Persistence",
    "Persistence-Twilight",
    "Linear",
    "RandomForest",
    "MLP",
    "NBEATSx-Oracle",
    "NBEATSx-Ridge",
]


def load_results() -> pd.DataFrame:
    """Load paper results final."""
    results_file = RESULTS_PATH / "paper_results_final.csv"
    df = pd.read_csv(results_file)
    df["twilight_time"] = pd.to_datetime(df["twilight_time"], format="mixed")
    df["forecast_time"] = pd.to_datetime(df["forecast_time"], format="mixed")
    df["abs_error"] = np.abs(df["error"])
    return df


def fig0_dataset_overview():
    """Figure 0: Dataset overview with full year and representative 5-day window."""
    print("Generating Figure 0: Dataset Overview...")

    # Set seaborn style: white background, no grid
    sns.set_style("white")
    # Set seaborn context for better font sizing
    with sns.plotting_context("talk", font_scale=0.9):
        # 1. Load data (timestamps are in UTC)
        df = pd.read_csv(DATA_PATH)
        df["ds"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("ds").reset_index(drop=True)

        # 2. Filter to 2025
        df = df[(df["ds"] >= "2025-01-01") & (df["ds"] < "2026-01-01")].copy()

        # 3. Convert to Chile local time (America/Santiago)
        df["ds_local"] = (
            df["ds"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Santiago")
            .dt.tz_localize(None)
        )

        # 4. Compute twilight baseline from NBEATSx-Ridge forecast targets (alt_sun=-15)
        # Load our actual twilight events (from paper_results_final.csv)
        forecast_df_temp = pd.read_csv(RESULTS_PATH / "paper_results_final.csv")
        forecast_df_temp["twilight_time"] = pd.to_datetime(
            forecast_df_temp["twilight_time"], format="mixed"
        )
        # Get unique twilight events with actual temp
        nb_temp = forecast_df_temp[forecast_df_temp["model"] == "NBEATSx-Ridge"]
        tw_unique = nb_temp.groupby("twilight_time")["actual_temp"].first().reset_index()
        tw_unique["tw_local"] = (
            tw_unique["twilight_time"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Santiago")
            .dt.tz_localize(None)
        )
        # Filter to 2025
        tw_unique = tw_unique[
            (tw_unique["tw_local"] >= "2025-01-01") & (tw_unique["tw_local"] < "2026-01-01")
        ].sort_values("tw_local")

        twilight_times = tw_unique["tw_local"].values
        twilight_temps = tw_unique["actual_temp"].values
        # Mark rows in df closest to each twilight event
        twilight_mask = pd.Series(False, index=df.index)
        df["twilight_temp"] = np.nan
        for _, tw_row in tw_unique.iterrows():
            tw_local = tw_row["tw_local"]
            diffs = (df["ds_local"] - tw_local).abs()
            closest_idx = diffs.idxmin()
            if diffs[closest_idx] < pd.Timedelta(minutes=30):
                twilight_mask[closest_idx] = True
                df.loc[closest_idx, "twilight_temp"] = tw_row["actual_temp"]

        # Linear interpolation for all timestamps
        df["twilight_baseline"] = np.interp(
            df["ds_local"].astype(np.int64),
            pd.to_datetime(twilight_times).astype(np.int64),
            twilight_temps,
        )

        # 5. Representative 5-day window with trend change
        # Jun 7 2025: ~6°C mean temp change over 5 days (winter transition)
        center_twilight = pd.Timestamp("2025-06-07 22:30:00")
        window_start = center_twilight - pd.Timedelta(days=2.5)
        window_end = center_twilight + pd.Timedelta(days=2.5)
        # Compute differenced signals on full df before slicing
        df["D_half"] = df["y"] - df["y"].shift(48)   # 48 steps @ 15min = 12h = 0.5 SD
        df["D_full"] = df["y"] - df["y"].shift(96)   # 96 steps @ 15min = 24h = 1 SD
        window_data = df[
            (df["ds_local"] >= window_start) & (df["ds_local"] <= window_end)
        ].copy()

        # 6. Get twilight and sunrise events for the window (in local time)
        sunrise_mask = df["sunrise_temp"].notna()
        window_twilights = df[
            (df["ds_local"] >= window_start)
            & (df["ds_local"] <= window_end)
            & twilight_mask
        ]["ds_local"].values
        window_sunrises = df[
            (df["ds_local"] >= window_start - pd.Timedelta(days=1))
            & (df["ds_local"] <= window_end + pd.Timedelta(days=1))
            & sunrise_mask
        ]["ds_local"].values

        # 7. Load NBEATSx-Ridge forecasts at 9am local time for all twilights in window
        forecast_df = pd.read_csv(RESULTS_PATH / "paper_results_final.csv")
        forecast_df["twilight_time"] = pd.to_datetime(
            forecast_df["twilight_time"], format="mixed"
        )
        forecast_df["forecast_time"] = pd.to_datetime(
            forecast_df["forecast_time"], format="mixed"
        )
        # Convert to Chile local time
        forecast_df["twilight_local"] = (
            forecast_df["twilight_time"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Santiago")
            .dt.tz_localize(None)
        )
        forecast_df["forecast_local"] = (
            forecast_df["forecast_time"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Santiago")
            .dt.tz_localize(None)
        )
        forecast_df["forecast_local_hour"] = forecast_df["forecast_local"].dt.hour

        # Get NBEATSx-Ridge forecasts issued at midday (~6h lead before twilight)
        ridge_9am = forecast_df[
            (forecast_df["model"] == "NBEATSx-Ridge")
            & (np.abs(forecast_df["lead_time_hours"] - 6.0) < 0.5)
        ].copy()

        # Match forecasts to window twilights
        forecast_points = []
        for tw_time in window_twilights:
            tw_local = pd.Timestamp(tw_time)
            match = ridge_9am[
                abs((ridge_9am["twilight_local"] - tw_local).dt.total_seconds()) < 7200
            ]
            if len(match) > 0:
                forecast_points.append((tw_local, match.iloc[0]["forecast_temp"]))

        # 8. Create figure with 3 rows (middle and bottom share x-axis, no gap)
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[1, 1, 1],
            hspace=0.25,
            top=0.95,
            bottom=0.08,
        )
        ax1 = fig.add_subplot(gs[0])
        # Middle and bottom with shared x-axis and no gap between them
        gs_inner = gs[1:].subgridspec(2, 1, hspace=0)
        ax2 = fig.add_subplot(gs_inner[0])
        ax3 = fig.add_subplot(gs_inner[1], sharex=ax2)
        axes = [ax1, ax2, ax3]

        # Row 1: Full year temperature
        ax1 = axes[0]
        ax1.plot(df["ds_local"], df["y"], color="#053061", linewidth=1.0, alpha=0.8)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("Full Year Temperature (2025)")
        ax1.set_xlim(df["ds_local"].min(), df["ds_local"].max())
        paper_ticks(ax1)

        # Highlight the representative window with shaded region + dashed outline
        ax1.axvspan(
            window_start,
            window_end,
            color="#D3D3D3",
            alpha=0.4,
            zorder=0,
        )
        ax1.axvline(window_start, color="#555555", linestyle="--", linewidth=1.2)
        ax1.axvline(window_end, color="#555555", linestyle="--", linewidth=1.2)

        # Add annotation for the window
        window_center = window_start + (window_end - window_start) / 2
        y_pos = ax1.get_ylim()[1] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        ax1.annotate(
            "Representative\n5-day window",
            xy=(window_center, y_pos),
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )

        # Format x-axis with abbreviated month names
        from matplotlib.dates import DateFormatter, MonthLocator

        ax1.xaxis.set_major_locator(MonthLocator())
        ax1.xaxis.set_major_formatter(DateFormatter("%b"))
        # Bold quarterly ticks (Jan, Apr, Jul, Oct)
        for label in ax1.get_xticklabels():
            if label.get_text() in ["Jan", "Apr", "Jul", "Oct"]:
                label.set_fontweight("bold")

        # Row 2: Representative 5-day window with twilight-trend
        ax2 = axes[1]

        # Night bands (twilight to next sunrise)
        first_night = True
        for tw_time in window_twilights:
            tw_time = pd.Timestamp(tw_time)
            future_sunrises = [s for s in window_sunrises if pd.Timestamp(s) > tw_time]
            if future_sunrises:
                next_sunrise = pd.Timestamp(future_sunrises[0])
                label_night = "Night (twilight to sunrise)" if first_night else None
                ax2.axvspan(
                    tw_time,
                    next_sunrise,
                    color="#2c3e50",
                    alpha=0.05,
                    label=label_night,
                )
                first_night = False

        # Raw temperature
        ax2.plot(
            window_data["ds_local"],
            window_data["y"],
            color="#053061",
            linewidth=2.0,
            label="Temperature",
        )
        # Twilight-trend baseline (thicker, darker)
        ax2.plot(
            window_data["ds_local"],
            window_data["twilight_baseline"],
            color="firebrick",
            linewidth=2.0,
            linestyle="--",
            label="Twilight-Trend",
        )
        # Add markers at twilight anchor points
        window_tw_data = window_data[window_data["twilight_temp"].notna()]
        ax2.scatter(
            window_tw_data["ds_local"],
            window_tw_data["twilight_temp"],
            color="firebrick",
            s=50,
            zorder=5,
            marker="s",
            edgecolors="white",
            linewidths=1,
        )
        # Twilight markers (vertical lines)
        for tw_time in window_twilights:
            ax2.axvline(
                pd.Timestamp(tw_time),
                color="gray",
                linestyle="-",
                linewidth=1.5,
                alpha=0.4,
            )

        # NBEATSx-Ridge forecast stars on all twilights
        for i, (tw_local, forecast_temp) in enumerate(forecast_points):
            label = "NBEATSx-Ridge \n midday forecast" if i == 0 else None
            ax2.plot(
                tw_local,
                forecast_temp,
                marker="*",
                markersize=12,
                color="gold",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
                zorder=10,
            )

        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title("Representative 5-Day Window")
        ax2.set_xlim(window_start, window_end)
        ax2.legend(loc="lower right", fontsize=11)
        paper_ticks(ax2)

        # Hide x-tick labels on ax2 (shared with ax3)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_xlabel("")

        # Row 3: D = T(tn) - T(tn - 1/2 solar day) — the differenced target
        ax3 = axes[2]

        # Night bands (match ax2 styling)
        for tw_time in window_twilights:
            tw_time = pd.Timestamp(tw_time)
            future_sunrises = [s for s in window_sunrises if pd.Timestamp(s) > tw_time]
            if future_sunrises:
                next_sunrise = pd.Timestamp(future_sunrises[0])
                ax3.axvspan(tw_time, next_sunrise, color="#2c3e50", alpha=0.05)

        # Twilight markers (match ax2 styling)
        for tw_time in window_twilights:
            ax3.axvline(
                pd.Timestamp(tw_time),
                color="gray",
                linestyle="-",
                linewidth=1.5,
                alpha=0.4,
            )

        ax3.plot(
            window_data["ds_local"],
            window_data["D_full"],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            label=r"$T(t) - T(t - 1\,\mathrm{SD})$",
        )
        ax3.plot(
            window_data["ds_local"],
            window_data["D_half"],
            color="#b2182b",
            linewidth=2.0,
            label=r"$\Delta T = T(t) - T(t - \frac{1}{2}\,\mathrm{SD})$",
        )
        ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax3.set_ylabel(r"$\Delta T$ (°C)")
        ax3.set_xlabel("Date")
        ax3.set_xlim(window_start, window_end)
        ax3.set_ylim(-9, 9)
        ax3.legend(loc="upper right", fontsize=11)
        paper_ticks(ax3)

        # Format x-axis with date and hour (shared with ax2)
        from matplotlib.dates import DateFormatter, HourLocator

        ax3.xaxis.set_major_locator(HourLocator(byhour=[0, 12]))
        ax3.xaxis.set_major_formatter(DateFormatter("%b %-d\n%H:%M"))

        # Add Chile season markers on top plot (Southern Hemisphere)
        seasons = [
            ("2025-01-03", "Summer"),
            ("2025-03-21", "Fall"),
            ("2025-06-21", "Winter"),
            ("2025-09-23", "Spring"),
        ]
        for date_str, season_name in seasons:
            season_date = pd.Timestamp(date_str)
            ax1.axvline(
                season_date, color="gray", linestyle=":", linewidth=1, alpha=0.7
            )
            ax1.text(
                season_date,
                -8.5,
                f"  {season_name}",
                fontsize=11,
                color="gray",
                rotation=0,
                verticalalignment="bottom",
                horizontalalignment="left",
            )

        plt.savefig(
            FIGURES_PATH / "fig0_dataset_overview.png", dpi=150, bbox_inches="tight"
        )
        plt.savefig(FIGURES_PATH / "fig0_dataset_overview.pdf", bbox_inches="tight")
        plt.close()
        print("  Saved fig0_dataset_overview.png/pdf")


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each model and lead time."""
    metrics = []
    for model in df["model"].unique():
        for lt in sorted(df["lead_time_hours"].unique()):
            data = df[(df["model"] == model) & (df["lead_time_hours"] == lt)]
            if len(data) > 0:
                metrics.append(
                    {
                        "model": model,
                        "lead_time_hours": lt,
                        "rmse": np.sqrt(np.mean(data["error"] ** 2)),
                        "mae": np.mean(data["abs_error"]),
                        "median": np.median(data["abs_error"]),
                        "bias": data["error"].mean(),
                        "pct_lt_1": (data["abs_error"] < 1.0).mean() * 100,
                        "n": len(data),
                    }
                )
    return pd.DataFrame(metrics)


def fig2_rmse_heatmap():
    """Figure 2: RMSE heatmap (model x lead time)."""
    import matplotlib.patheffects as pe

    sns.set_context("notebook", font_scale=1.1)

    print("Generating Figure 2: RMSE heatmap...")

    # Load paper_results_final.csv (has Prophet data)
    results_file = RESULTS_PATH / "paper_results_final.csv"
    df = pd.read_csv(results_file)

    # Define models and lead times
    models = ["Persistence", "RandomForest", "MLP", "Prophet", "NBEATSx-Ridge"]
    model_abbrev = {
        "Persistence": "Pers.",
        "RandomForest": "RF",
        "MLP": "MLP",
        "Prophet": "Prophet",
        "NBEATSx-Ridge": "NBEATS-R",
    }
    lead_times = [0.5, 1.0, 3.0, 6.0, 9.0, 12.0]

    df = df[df["model"].isin(models)]

    # Compute RMSE for each (model, lead_time) pair
    rmse_data = []
    for model in models:
        for lt in lead_times:
            subset = df[
                (df["model"] == model) & (np.abs(df["lead_time_hours"] - lt) < 0.1)
            ]
            if len(subset) > 0:
                rmse = np.sqrt(np.mean(subset["error"] ** 2))
            else:
                rmse = np.nan
            rmse_data.append({"model": model, "lead_time": lt, "rmse": rmse})

    rmse_df = pd.DataFrame(rmse_data)
    pivot = rmse_df.pivot(index="model", columns="lead_time", values="rmse")
    pivot = pivot.reindex(models)  # Keep model order

    # Create figure with imshow
    fig, ax = plt.subplots(figsize=(10, 8))

    vmin, vmax = 0.35, 1.75
    # Use RdYlBu_r: blue (low/good) to red (high/bad), matching fig5 color scheme
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="RdYlBu_r",
    )
    ax.grid(False)

    # Annotate cells with white text + gray outline
    for i, model in enumerate(pivot.index):
        for j, lt in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="w",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.025, foreground="gray")],
                )

    # Set ticks and labels
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels(lead_times)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(
        [model_abbrev[m] for m in models], rotation=45, ha="right", fontsize=14
    )

    ax.set_xlabel("Hours from sunset (h)", fontsize=18)
    ax.set_title("RMSE by Model and Lead Time", fontsize=18)

    # Set black borders with thicker linewidth
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("RMSE (°C)")

    plt.savefig(FIGURES_PATH / "fig2_rmse_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_PATH / "fig2_rmse_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig2_rmse_heatmap.png/pdf")


def paper_ticks(ax):
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        bottom=True,  # add
        left=True,  # add
    )
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=2, width=0.8)


def fig3_rmse_vs_lead_time():
    """Figure 3: RMSE vs lead time line plot (same style as fig5_cdf)."""
    print("Generating Figure 3: RMSE vs lead time...")

    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.8,
        rc={
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        },
    )

    results = load_results()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.minorticks_on()

    models_to_plot = [
        "Persistence",
        "Prophet",
        "RandomForest",
        "MLP",
        "Linear",
        "NBEATSx-Ridge",
    ]

    # Color palette: warm for baselines, cool for deep learning
    line_colors = {
        "Persistence": "#ffb703",       # Gold/yellow
        "Prophet": "#8338ec",           # Purple
        "RandomForest": "#fb8500",      # Orange
        "MLP": "#e63946",              # Coral red
        "Linear": "#00b4d8",           # Bright blue
        "NBEATSx-Ridge": "#023e8a",    # Dark navy
    }

    line_labels = {
        "Persistence": "Persistence",
        "Prophet": "Prophet",
        "RandomForest": "Random Forest",
        "MLP": "MLP",
        "Linear": "Linear",
        "NBEATSx-Ridge": "NBEATSx-Blend",
    }

    lead_times = np.arange(0.5, 12.5, 0.5)
    N_BOOT = 1000

    for model in models_to_plot:
        model_data = results[results["model"] == model]
        if len(model_data) == 0:
            continue
        rmses = []
        rmse_lo = []
        rmse_hi = []
        valid_lts = []
        for lt in lead_times:
            subset = model_data[np.abs(model_data["lead_time_hours"] - lt) < 0.01]
            if len(subset) > 0:
                errors = subset["error"].values
                rmse = np.sqrt(np.mean(errors ** 2))
                rmses.append(rmse)
                valid_lts.append(lt)
                # Bootstrap 95% CI
                boot_rmses = np.array([
                    np.sqrt(np.mean(np.random.choice(errors, size=len(errors), replace=True) ** 2))
                    for _ in range(N_BOOT)
                ])
                rmse_lo.append(np.percentile(boot_rmses, 2.5))
                rmse_hi.append(np.percentile(boot_rmses, 97.5))

        valid_lts = np.array(valid_lts)
        rmses = np.array(rmses)
        rmse_lo = np.array(rmse_lo)
        rmse_hi = np.array(rmse_hi)

        ax.fill_between(
            valid_lts, rmse_lo, rmse_hi,
            color=line_colors[model], alpha=0.15,
        )
        ax.plot(
            valid_lts,
            rmses,
            label=line_labels[model],
            color=line_colors[model],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

    # Reference lines
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.axvline(3.0, color="#023e8a", linestyle=":", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("RMSE vs Lead Time")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3.0)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    paper_ticks(ax)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "fig3_rmse_vs_lead_time.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_PATH / "fig3_rmse_vs_lead_time.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig3_rmse_vs_lead_time.png/pdf")


def fig5_cdf_absolute_error():
    """Figure 5: CDF of absolute error at 3h."""
    print("Generating Figure 5: CDF of absolute error...")

    # Set style

    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.8,
        rc={
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        },
    )

    plt.minorticks_on()
    sns.despine(top=False, right=False, left=False, bottom=False)

    # Load paper_results_final.csv
    results_file = RESULTS_PATH / "paper_results_final.csv"
    results = pd.read_csv(results_file)
    results["abs_error"] = np.abs(results["error"])
    at_3h = results[np.abs(results["lead_time_hours"] - 3.0) < 0.01]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.minorticks_on()

    # Same models, colors, and labels as fig3
    models_to_plot = [
        "Persistence",
        "Prophet",
        "RandomForest",
        "MLP",
        "Linear",
        "NBEATSx-Ridge",
    ]

    cdf_colors = {
        "Persistence": "#ffb703",       # Gold/yellow
        "Prophet": "#8338ec",           # Purple
        "RandomForest": "#fb8500",      # Orange
        "MLP": "#e63946",              # Coral red
        "Linear": "#00b4d8",           # Bright blue
        "NBEATSx-Ridge": "#023e8a",    # Dark navy
    }

    cdf_labels = {
        "Persistence": "Persistence",
        "Prophet": "Prophet",
        "RandomForest": "Random Forest",
        "MLP": "MLP",
        "Linear": "Linear",
        "NBEATSx-Ridge": "NBEATSx-Blend",
    }

    # Store NBEATSx-Ridge data for reference line
    nbeats_ridge_pct = None

    for model in models_to_plot:
        model_data = at_3h[at_3h["model"] == model]
        if len(model_data) == 0:
            continue
        sorted_errors = np.sort(model_data["abs_error"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(
            sorted_errors,
            cdf,
            label=cdf_labels[model],
            color=cdf_colors[model],
            linewidth=2.5,
            linestyle="-",
        )

        if model == "NBEATSx-Ridge":
            nbeats_ridge_pct = (model_data["abs_error"] < 1.0).mean()

    # Reference line at 1°C threshold
    ax.axvline(
        1.0, color=cdf_colors["NBEATSx-Ridge"], linestyle="--", alpha=0.7, linewidth=1.5
    )

    # Horizontal line for NBEATSx-Blend at 1°C
    if nbeats_ridge_pct is not None:
        ax.hlines(
            nbeats_ridge_pct,
            0,
            1.0,
            colors=cdf_colors["NBEATSx-Ridge"],
            linestyles="--",
            linewidth=1.5,
        )
        ax.plot(
            1.0, nbeats_ridge_pct, "o", color=cdf_colors["NBEATSx-Ridge"], markersize=8
        )
        ax.text(
            1.05,
            nbeats_ridge_pct - 0.025,
            f"{nbeats_ridge_pct * 100:.0f}%",
            fontsize=12,
            va="center",
            color=cdf_colors["NBEATSx-Ridge"],
            fontweight="bold",
        )

    ax.set_xlabel("Absolute Error (°C)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Absolute Error at 3h Lead Time")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1)

    # Black borders
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    paper_ticks(ax)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "fig5_cdf_error.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_PATH / "fig5_cdf_error.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig5_cdf_error.png/pdf")


def fig6_seasonal_trend_analysis():
    """Figure 6: Seasonal effects and temperature trend analysis."""
    print("Generating Figure 6: Seasonal and Trend Analysis...")

    results = load_results()

    # Focus on NBEATSx-Ridge at 3h
    df = results[
        (results["model"] == "NBEATSx-Ridge") & (results["lead_time_hours"] == 3.0)
    ].copy()
    df["twilight_time"] = pd.to_datetime(df["twilight_time"])
    df["month"] = df["twilight_time"].dt.month
    df["day_of_year"] = df["twilight_time"].dt.dayofyear

    # Define seasons (Southern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return "Summer"
        elif month in [3, 4, 5]:
            return "Fall"
        elif month in [6, 7, 8]:
            return "Winter"
        else:
            return "Spring"

    df["season"] = df["month"].apply(get_season)

    # Calculate temperature trends from the actual temperatures
    df = df.sort_values("twilight_time").reset_index(drop=True)
    df["temp_diff_1d"] = df["actual_temp"].diff(1)  # 1-day change (since odd days only)
    df["temp_diff_3d"] = df["actual_temp"].diff(3)  # ~3-twilight change

    # Categorize trends
    def categorize_trend(diff):
        if pd.isna(diff):
            return "Unknown"
        elif diff < -1.5:
            return "Strong cooling"
        elif diff < -0.5:
            return "Moderate cooling"
        elif diff < 0.5:
            return "Stable"
        elif diff < 1.5:
            return "Moderate warming"
        else:
            return "Strong warming"

    df["trend_category"] = df["temp_diff_3d"].apply(categorize_trend)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Error distribution by season (box plot)
    ax1 = axes[0, 0]
    season_order = ["Summer", "Fall", "Winter", "Spring"]
    # Use RdYlBu colormap for seasons
    cmap = plt.cm.RdYlBu
    season_colors = {
        "Summer": cmap(0.1),   # Red (hot)
        "Fall": cmap(0.35),    # Orange/Yellow
        "Winter": cmap(0.9),   # Blue (cold)
        "Spring": cmap(0.65),  # Light blue
    }

    box_data = [df[df["season"] == s]["error"].dropna().values for s in season_order]
    bp = ax1.boxplot(box_data, labels=season_order, patch_artist=True)
    for patch, season in zip(bp["boxes"], season_order):
        patch.set_facecolor(season_colors[season])
        patch.set_alpha(0.6)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel("Prediction Error (°C)")
    ax1.set_title("Error Distribution by Season")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Percentage within thresholds by season
    ax2 = axes[0, 1]
    season_stats = []
    for s in season_order:
        subset = df[df["season"] == s]
        abs_error = np.abs(subset["error"])
        season_stats.append(
            {
                "season": s,
                "pct_05": (abs_error < 0.5).mean() * 100,
                "pct_1": (abs_error < 1.0).mean() * 100,
                "n": len(subset),
            }
        )
    stats_df = pd.DataFrame(season_stats)

    x = np.arange(len(season_order))
    width = 0.35
    ax2.bar(
        x - width / 2,
        stats_df["pct_05"],
        width,
        label="< 0.5°C",
        color="steelblue",
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2, stats_df["pct_1"], width, label="< 1°C", color="coral", alpha=0.8
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(season_order, stats_df["n"])])
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Forecast Accuracy by Season")
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Bias by season
    ax3 = axes[1, 0]
    bias_by_season = df.groupby("season")["error"].mean().reindex(season_order)
    colors = [season_colors[s] for s in season_order]
    bars = ax3.bar(
        season_order, bias_by_season.values, color=colors, alpha=0.7, edgecolor="black"
    )
    ax3.axhline(0, color="black", linestyle="-", linewidth=1)
    ax3.set_ylabel("Bias (°C)")
    ax3.set_title("Prediction Bias by Season")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: RMS by temperature trend category
    ax4 = axes[1, 1]
    trend_order = [
        "Strong cooling",
        "Moderate cooling",
        "Stable",
        "Moderate warming",
        "Strong warming",
    ]
    trend_colors = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]

    trend_stats = []
    for t in trend_order:
        subset = df[df["trend_category"] == t]
        if len(subset) > 5:
            trend_stats.append(
                {
                    "trend": t,
                    "rmse": np.sqrt((subset["error"] ** 2).mean()),
                    "n": len(subset),
                }
            )
        else:
            trend_stats.append({"trend": t, "rmse": np.nan, "n": len(subset)})

    trend_df = pd.DataFrame(trend_stats)
    valid_mask = ~trend_df["rmse"].isna()

    bars = ax4.bar(
        range(len(trend_order)),
        trend_df["rmse"].values,
        color=trend_colors,
        edgecolor="black",
        alpha=0.8,
    )
    ax4.set_xticks(range(len(trend_order)))
    ax4.set_xticklabels(
        [f"{t}\n(n={n})" for t, n in zip(trend_order, trend_df["n"])],
        fontsize=8,
        rotation=15,
        ha="right",
    )
    ax4.set_ylabel("RMSE (°C)")
    ax4.set_title(
        "RMS Error by Temperature Trend Category\n(based on 3-twilight temperature change)"
    )
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / "fig6_seasonal_trend_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(FIGURES_PATH / "fig6_seasonal_trend_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig6_seasonal_trend_analysis.png/pdf")


def fig7_rate_forecast():
    """Figure 7: Rate (dT/dt) forecast results using Δ(dT/dt) formulation."""
    print("Generating Figure 7: Rate forecast results...")

    # Load rate forecast results from delta formulation
    rate_file = RESULTS_PATH / "rate_predictions_ridge_delta_3h.csv"
    if not rate_file.exists():
        print(
            f"  Warning: {rate_file.name} not found. Run rate_forecast_delta.py first."
        )
        return

    df = pd.read_csv(rate_file)
    df["tw_time"] = pd.to_datetime(df["tw_time"])
    df = df.sort_values("tw_time").reset_index(drop=True)

    # Compute errors
    df["error"] = df["dT_dt_ridge"] - df["dT_dt_tw_actual"]
    df["abs_error"] = np.abs(df["error"])
    df["pers_error"] = df["dT_dt_tw_last"] - df["dT_dt_tw_actual"]

    # Compute metrics
    rmse = np.sqrt((df["error"] ** 2).mean())
    mae = df["abs_error"].mean()
    pers_rmse = np.sqrt((df["pers_error"] ** 2).mean())
    signal_std = df["dT_dt_tw_actual"].std()
    nrmse = rmse / signal_std * 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1 (top left): Predicted vs Actual scatter
    ax1 = axes[0, 0]
    # Persistence (previous twilight rate)
    ax1.scatter(
        df["dT_dt_tw_actual"],
        df["dT_dt_tw_last"],
        alpha=0.4,
        s=30,
        c="gray",
        marker="x",
        label=f"Persistence (RMSE={pers_rmse:.2f})",
    )
    # NBEATSx-Ridge
    ax1.scatter(
        df["dT_dt_tw_actual"],
        df["dT_dt_ridge"],
        alpha=0.6,
        s=40,
        c="#17becf",
        edgecolor="white",
        linewidth=0.5,
        label=f"NBEATSx-Ridge (RMSE={rmse:.2f})",
    )
    lims = [df["dT_dt_tw_actual"].min() - 0.2, df["dT_dt_tw_actual"].max() + 0.2]
    ax1.plot(lims, lims, "k--", linewidth=1.5, label="1:1 line")
    ax1.set_xlabel("Actual dT/dt at twilight (°C/h)")
    ax1.set_ylabel("Predicted dT/dt (°C/h)")
    ax1.set_title(f"Predicted vs Actual (NRMSE={nrmse:.1f}%)")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2 (top right): Error distribution (NBEATSx-Ridge only)
    ax2 = axes[0, 1]
    ax2.hist(
        df["error"],
        bins=25,
        alpha=0.7,
        color="#17becf",
        edgecolor="black",
    )
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.axvline(
        df["error"].mean(),
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Bias: {df['error'].mean():.3f}°C/h",
    )
    ax2.set_xlabel("Prediction Error (°C/h)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"NBEATSx-Ridge Error Distribution\nMAE={mae:.3f}°C/h")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3 (bottom left): Time series of actual vs predicted
    ax3 = axes[1, 0]
    ax3.plot(
        df["tw_time"],
        df["dT_dt_tw_actual"],
        "k-",
        linewidth=1,
        alpha=0.8,
        label="Actual",
    )
    ax3.plot(
        df["tw_time"],
        df["dT_dt_ridge"],
        "-",
        linewidth=1.5,
        alpha=0.7,
        color="#17becf",
        label="NBEATSx-Ridge",
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("dT/dt at twilight (°C/h)")
    ax3.set_title("Twilight Temperature Rate: Actual vs Predicted")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Panel 4 (bottom right): CDF of absolute error
    ax4 = axes[1, 1]

    # Persistence CDF
    sorted_pers_err = np.sort(np.abs(df["pers_error"]))
    cdf_pers = np.arange(1, len(sorted_pers_err) + 1) / len(sorted_pers_err)
    ax4.plot(sorted_pers_err, cdf_pers, color="gray", linewidth=2, label="Persistence")

    # NBEATSx-Ridge CDF
    sorted_err = np.sort(df["abs_error"])
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax4.plot(sorted_err, cdf, color="#17becf", linewidth=2, label="NBEATSx-Ridge")

    # Threshold line at 0.2°C/h
    threshold = 0.2
    ax4.axvline(threshold, color="red", linestyle="--", linewidth=1.5)

    # Find CDF value at threshold and draw horizontal line
    pct_below_threshold = (df["abs_error"] < threshold).mean()
    ax4.hlines(
        pct_below_threshold, 0, threshold, colors="red", linestyles="--", linewidth=1.5
    )
    ax4.plot(threshold, pct_below_threshold, "ro", markersize=6)
    ax4.text(
        threshold + 0.02,
        pct_below_threshold - 0.08,
        f"{pct_below_threshold * 100:.0f}%",
        fontsize=10,
        va="top",
        color="red",
    )

    ax4.set_xlabel("Absolute Error (°C/h)")
    ax4.set_ylabel("Cumulative Probability")
    ax4.set_title("CDF of Absolute Rate Error")
    ax4.legend(loc="lower right")
    ax4.set_xlim(0, 0.5)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "fig7_rate_forecast.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_PATH / "fig7_rate_forecast.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig7_rate_forecast.png/pdf")
    print(
        f"  RMSE: {rmse:.3f}°C/h, NRMSE: {nrmse:.1f}%, Persistence RMSE: {pers_rmse:.3f}°C/h"
    )


def _draw_density_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 80,
    levels: tuple = (0.5, 0.8, 0.95),
    linewidth: float = 1.5,
    linecolor: str = "k",
    zorder: int = 5,
):
    """Draw percentile-mass contours of point density on ax."""
    if x.size == 0:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    H = H.T
    total = H.sum()
    if total <= 0:
        return

    H_flat = H.ravel()
    order = np.argsort(H_flat)[-1::-1]
    H_sorted = H_flat[order]
    cumsum = np.cumsum(H_sorted) / total

    thr = []
    for p in levels:
        k = np.searchsorted(cumsum, p, side="left")
        k = min(k, len(H_sorted) - 1)
        thr.append(float(H_sorted[k]))

    pairs = sorted(zip(thr, levels), key=lambda t: t[0])
    dedup = []
    for t, p in pairs:
        if not dedup or t != dedup[-1][0]:
            dedup.append([t, p])
        else:
            dedup[-1][1] = max(dedup[-1][1], p)

    thr_sorted = np.array([t for t, _ in dedup], dtype=float)
    mass_sorted = tuple(p for _, p in dedup)

    if thr_sorted.size == 0:
        return

    eps = np.finfo(float).eps * max(1.0, float(np.nanmax(thr_sorted)))
    thr_strict = np.maximum.accumulate(thr_sorted)
    thr_strict = thr_strict + eps * np.arange(thr_strict.size)

    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])

    CS = ax.contour(
        Xc,
        Yc,
        H,
        levels=thr_strict,
        linewidths=linewidth,
        colors=linecolor,
        antialiased=True,
        zorder=zorder,
    )
    fmt = {lvl: f"{int(p * 100)}%" for lvl, p in zip(CS.levels, mass_sorted)}
    ax.clabel(CS, inline=True, fmt=fmt, fontsize=9)


def _mad_sigma(residuals: np.ndarray) -> float:
    """Robust scale estimate from MAD."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad


def _robust_inlier_mask(
    y: np.ndarray, yhat: np.ndarray, sigma: float = 3.0
) -> np.ndarray:
    """Sigma-clip around the initial fit using MAD."""
    r = y - yhat
    s = _mad_sigma(r)
    if not np.isfinite(s) or s == 0.0:
        q1, q3 = np.percentile(r, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return np.ones_like(r, dtype=bool)
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return (r >= lo) & (r <= hi)
    return np.abs(r) <= sigma * s


def fig8_comparison_nbeats_prophet_meteoblue():
    """Figure 8: Compare NBEATSx-Ridge vs Prophet vs MeteoBlue at twilight."""
    import seaborn as sns

    print("Generating Figure 8: Model Comparison...")

    # Set seaborn style: white background, no grid
    sns.set_style("white")
    sns.set_context("talk", font_scale=0.9)

    # Load paper_results_final.csv
    results_file = RESULTS_PATH / "paper_results_final.csv"
    if not results_file.exists():
        print(
            f"  Warning: {results_file} not found. Run add_external_forecasts.py first."
        )
        return

    df = pd.read_csv(results_file)
    df["twilight_time"] = pd.to_datetime(df["twilight_time"])

    # NBEATSx and Prophet at 6h lead; MeteoBlue uses all available
    nbeats_df = df[(df["model"] == "NBEATSx-Ridge") & (np.abs(df["lead_time_hours"] - 6.0) < 0.01)].copy()
    prophet_df = df[(df["model"] == "Prophet") & (np.abs(df["lead_time_hours"] - 6.0) < 0.5)].copy()
    meteoblue_df = df[df["model"] == "MeteoBlue"].copy()

    print(f"  NBEATSx-Blend (6h): {len(nbeats_df)} points")
    print(f"  Prophet (6h): {len(prophet_df)} points")
    print(f"  MeteoBlue (all): {len(meteoblue_df)} points")

    # Colors (matching fig5 CDF palette)
    colors = {
        "NBEATSx-Ridge": "#023e8a",  # Dark navy
        "Prophet": "#00b4d8",  # Bright blue
        "MeteoBlue": "#e63946",  # Coral red
    }

    # Create 2x3 figure
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        gridspec_kw={"height_ratios": [1.75, 1.0], "wspace": 0, "hspace": 0},
        sharex="col",
        sharey="row",
    )

    datasets = [
        (nbeats_df, "NBEATSx-Blend (mid-day)", colors["NBEATSx-Ridge"]),
        (prophet_df, "Prophet (mid-day)", colors["Prophet"]),
        (meteoblue_df, "MeteoBlue", colors["MeteoBlue"]),
    ]

    for col, (data_df, title, color) in enumerate(datasets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if len(data_df) == 0:
            ax_top.set_title(title)
            continue

        x = data_df["actual_temp"].to_numpy()
        y = data_df["forecast_temp"].to_numpy()

        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) == 0:
            ax_top.set_title(title)
            continue

        # Set axis limits
        ax_top.set_xlim(-8, 25)
        ax_top.set_ylim(-8, 25)

        # Use all points (no subsampling needed with these counts)
        idx = np.arange(len(x))

        # Large marker size for all panels
        marker_size = 120

        # Scatter plot (top panel) - no contours, rasterized for smaller file size
        ax_top.scatter(
            x[idx],
            y[idx],
            s=marker_size,
            alpha=0.6,
            edgecolors="none",
            color=color,
            zorder=3,
            rasterized=True,
        )

        # 1:1 line
        ax_top.plot(
            [-8, 25],
            [-8, 25],
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.9,
            zorder=2,
        )

        # Robust fit
        if np.std(x) > 0:
            b0, a0 = np.polyfit(x, y, 1)
        else:
            b0, a0 = 0.0, float(np.mean(y))
        yhat0 = a0 + b0 * x

        inliers = _robust_inlier_mask(y, yhat0, sigma=3.0)
        xi, yi = x[inliers], y[inliers]

        if xi.size >= 2 and np.std(xi) > 0:
            b, a = np.polyfit(xi, yi, 1)
        else:
            b, a = b0, a0

        # Fitted line
        xfit = np.array([-8, 25])
        yfit = a + b * xfit
        ax_top.plot(xfit, yfit, color="lightgrey", lw=1.4, zorder=7)

        # Stats
        n_in = int(np.sum(inliers))
        bias = float(np.mean(yi - xi)) if n_in > 0 else np.nan
        std = float(np.std(yi - xi, ddof=1)) if n_in > 1 else np.nan
        yhat_fit = a + b * xi
        ss_res = float(np.sum((yi - yhat_fit) ** 2)) if n_in > 0 else np.nan
        ss_tot = float(np.sum((yi - np.mean(yi)) ** 2)) if n_in > 0 else np.nan
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Stats text box
        txt = (
            f"N = {n_in}\n"
            f"Slope = {b:.2f}\n"
            f"Bias = {bias:.2f} °C\n"
            f"Std = {std:.2f} °C\n"
            f"R² = {r2:.2f}"
        )
        ax_top.text(
            0.02,
            0.98,
            txt,
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            color="k",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        ax_top.set_title(title, fontsize=18, fontweight="bold")
        paper_ticks(ax_top)

        # Bottom panel: Residuals
        ax_bot.set_xlim(-8, 25)
        ax_bot.set_ylim(-8, 8)

        residuals = y - x
        ax_bot.scatter(
            x[idx],
            residuals[idx],
            s=marker_size,
            alpha=0.6,
            edgecolors="none",
            color=color,
            zorder=3,
            rasterized=True,
        )

        # Zero line
        ax_bot.axhline(0.0, color="gray", lw=1.0, ls="--", alpha=0.9, zorder=2)

        # Fitted residual line
        rfit = a + (b - 1.0) * xfit
        ax_bot.plot(xfit, rfit, color="lightgrey", lw=1.4, zorder=7)

        paper_ticks(ax_bot)

    # Labels
    axes[0, 0].set_ylabel(r"Forecast Temp [°C]", fontsize=16)
    axes[1, 0].set_ylabel(r"Residual [°C]", fontsize=16)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Actual Temp [°C]", fontsize=16)

    # Remove internal ticks
    for ax in axes[0, 1:]:
        ax.tick_params(labelleft=False)
    for ax in axes[1, 1:]:
        ax.tick_params(labelleft=False)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "fig8_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_PATH / "fig8_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig8_comparison.png/pdf")


def table1_summary():
    """Generate Table 1: Summary metrics at 3h."""
    print("Generating Table 1: Summary metrics at 3h...")

    results = load_results()
    metrics = compute_metrics(results)

    at_3h = metrics[metrics["lead_time_hours"] == 3.0].copy()
    at_3h = at_3h.set_index("model")
    at_3h = at_3h.reindex([m for m in MODEL_ORDER if m in at_3h.index])

    # Create LaTeX table
    latex = at_3h[["rmse", "mae", "median", "bias", "pct_lt_1"]].to_latex(
        float_format="%.2f",
        column_format="lrrrrr",
        header=["RMSE (C)", "MAE (C)", "Median (C)", "Bias (C)", "% < 1C"],
    )

    with open(FIGURES_PATH / "table1_summary_3h_v2.tex", "w") as f:
        f.write(latex)

    at_3h[["rmse", "mae", "median", "bias", "pct_lt_1"]].to_csv(
        FIGURES_PATH / "table1_summary_3h_v2.csv"
    )

    print("  Saved table1_summary_3h_v2.tex and table1_summary_3h_v2.csv")
    print("\nTable 1 - Summary at 3h Lead Time:")
    print(at_3h[["rmse", "mae", "median", "bias", "pct_lt_1"]].to_string())


def main():
    """Generate all figures."""
    print("=" * 60)
    print("PAPER PLOTS v2: Twilight Temperature Forecasting")
    print("Including NBEATSx-Ridge")
    print("=" * 60)

    # Create figures directory
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {FIGURES_PATH}")

    # Generate all figures (matching main.tex)
    fig0_dataset_overview()
    fig2_rmse_heatmap()
    fig3_rmse_vs_lead_time()
    fig5_cdf_absolute_error()
    fig6_seasonal_trend_analysis()
    fig7_rate_forecast()
    fig8_comparison_nbeats_prophet_meteoblue()
    table1_summary()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
