"""
Compute sunrise and twilight temperatures from gap-filled hourly data.

Pipeline:
1. Load gap-filled hourly data
2. Upsample to 15-min resolution (for mask accuracy)
3. Compute sunrise temperatures (sun rising, alt 0-15°)
4. Compute twilight temperatures (sun setting, alt 0 to -15°)
5. Merge and save combined dataframe (daily resolution)

Output: data/temp_history_all_dec2025_sunrise_sunset.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .sun_utils import (
    compute_sunrise_temperatures,
    compute_twilight_temperatures,
)

# =============================================================================
# Config
# =============================================================================

INPUT_PATH = Path("data/temp_history_all_dec2025_filled_gaps.csv")
OUTPUT_PATH = Path("data/temp_history_all_dec2025_sunrise_sunset.csv")


# =============================================================================
# Data Loading and Upsampling
# =============================================================================


def load_and_upsample(filepath: Path, target_freq: str = "15min") -> pd.DataFrame:
    """
    Load gap-filled data and upsample to higher resolution.

    Upsampling ensures accurate mask application for sunrise/twilight detection.

    Args:
        filepath: Path to gap-filled CSV
        target_freq: Target frequency for upsampling (default: 15min)

    Returns:
        Upsampled DataFrame with 'timestamp', 'tempMin', 'tempMax', 'tempMean' columns
    """
    print(f"[1] Loading {filepath}...")
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("ds")
    df["tempMean"] = (df["tempMin"] + df["tempMax"]) / 2.0

    # Ensure timezone
    if df["ds"].dt.tz is None:
        df["ds"] = df["ds"].dt.tz_localize("UTC")

    print(f"  Loaded: {len(df)} hours")
    print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Drop NaN values
    n_nan = df["y"].isna().sum()
    if n_nan > 0:
        print(f"  Dropping {n_nan} NaN values...")
        df = df.dropna(subset=["y"])

    print(f"\n[2] Upsampling to {target_freq}...")
    if target_freq == "15min":
        return df[["ds", "y", "timestamp", "tempMin", "tempMax", "tempMean"]]
    else:
        # # Create full datetime range at target frequency
        full_range = pd.date_range(
            start=df["ds"].min(),
            end=df["ds"].max(),
            freq=target_freq,
            tz="UTC",
        )

        # Reindex and interpolate
        df_upsampled = df.set_index("ds").reindex(full_range)
        df_upsampled["y"] = df_upsampled["y"].interpolate(method="time")
        df_upsampled = df_upsampled.reset_index().rename(columns={"index": "timestamp"})

        # Create tempMin, tempMax, tempMean columns (same value since we only have 'y')
        df_upsampled["tempMin"] = df_upsampled["y"]
        df_upsampled["tempMean"] = df_upsampled["y"]
        df_upsampled["tempMax"] = df_upsampled["y"]

        print(f"  Upsampled: {len(df_upsampled)} records")
        return df_upsampled


# =============================================================================
# Main Pipeline
# =============================================================================


def compute_all_sun_temperatures():
    """
    Compute sunrise and twilight temperatures using mask-based approach.

    Returns combined DataFrame with daily temperatures.
    """
    # Load and upsample
    df = load_and_upsample(INPUT_PATH)

    # Compute sunrise temperatures
    print("\n[3] Computing sunrise temperatures...")
    df_sunrise = compute_sunrise_temperatures(df)
    print(f"  Found {df_sunrise['sunrise_temp'].notna().sum()} days with sunrise data")

    # Compute twilight temperatures
    print("\n[4] Computing twilight temperatures...")
    df_twilight = compute_twilight_temperatures(df)
    print(
        f"  Found {df_twilight['twilight_temp'].notna().sum()} days with twilight data"
    )

    # Merge on date
    print("\n[5] Merging results...")
    df_combined = df_sunrise.merge(df_twilight, on="date", how="outer")

    # Sort by date
    df_combined = df_combined.sort_values("date").reset_index(drop=True)

    print(f"  Combined: {len(df_combined)} days")

    # Add derived columns
    df_combined["sunrise_to_twilight_delta"] = (
        df_combined["twilight_temp"] - df_combined["sunrise_temp"]
    )

    return df_combined


def merge_into_hourly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sunrise/twilight temps back into hourly gap-filled data.

    Each hourly row gets the sunrise_temp and twilight_temp for that day.

    Args:
        df_daily: DataFrame with date, sunrise_temp, twilight_temp columns

    Returns:
        Hourly DataFrame with sunrise_temp and twilight_temp columns added
    """
    print("\n[6] Merging into hourly data...")

    # Load the original hourly data
    df_hourly = pd.read_csv(INPUT_PATH)
    df_hourly["ds"] = pd.to_datetime(df_hourly["ds"])

    if df_hourly["ds"].dt.tz is None:
        df_hourly["ds"] = df_hourly["ds"].dt.tz_localize("UTC")

    df_hourly["ds"] = df_hourly["ds"].dt.tz_convert("America/Santiago")

    print(f"  Loaded {len(df_hourly)} hourly records")

    # Create date column for merging
    df_hourly["date"] = pd.to_datetime(df_hourly["ds"].dt.date)

    # Select only the columns we need from daily data
    df_to_merge = df_daily[
        ["date", "sunrise_temp", "twilight_temp", "sunrise_start", "twilight_start"]
    ].copy()

    # Merge on date
    df_merged = df_hourly.set_index("ds")
    for col in ["sunrise", "twilight"]:
        df_merged[f"{col}_temp"] = np.nan
        df_merged.loc[df_to_merge[f"{col}_start"].dropna(), f"{col}_temp"] = (
            df_to_merge[f"{col}_temp"].dropna().values
        )
        df_merged[f"last_{col}_temp"] = df_merged[f"{col}_temp"].ffill()

    # df_merged = df_hourly.merge(df_to_merge, on="date", how="left")

    # Drop the temporary date column
    # df_merged = df_merged.drop(columns=["date"])

    print(f"  Merged: {len(df_merged)} hourly records with sunrise/twilight temps")

    return df_merged


def main():
    """Generate sunrise/twilight temperature file."""
    print("=" * 60)
    print("Sunrise/Twilight Temperature Computation (Mask-Based)")
    print("=" * 60)

    # Compute daily temperatures
    df_daily = compute_all_sun_temperatures()

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nSunrise Temperature (sun alt 0-15°, rising):")
    print(f"  Mean: {df_daily['sunrise_temp'].mean():.2f} °C")
    print(f"  Std:  {df_daily['sunrise_temp'].std():.2f} °C")
    print(f"  Min:  {df_daily['sunrise_temp'].min():.2f} °C")
    print(f"  Max:  {df_daily['sunrise_temp'].max():.2f} °C")

    print("\nTwilight Temperature (sun alt 0 to -15°, setting):")
    print(f"  Mean: {df_daily['twilight_temp'].mean():.2f} °C")
    print(f"  Std:  {df_daily['twilight_temp'].std():.2f} °C")
    print(f"  Min:  {df_daily['twilight_temp'].min():.2f} °C")
    print(f"  Max:  {df_daily['twilight_temp'].max():.2f} °C")

    # Merge into hourly data
    df_hourly = merge_into_hourly(df_daily)

    # Save hourly data with sunrise/twilight columns
    print(f"\n[7] Saving to {OUTPUT_PATH}...")
    df_hourly.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved {len(df_daily)} hourly records")

    # Preview
    print("\n" + "=" * 60)
    print("Preview (first 10 rows)")
    print("=" * 60)
    df_daily["sunrise_start"] = df_daily["sunrise_start"].dt.tz_convert(
        "America/Santiago"
    )
    df_daily["sunrise_end"] = df_daily["sunrise_end"].dt.tz_convert("America/Santiago")
    print(df_daily.head(10).to_string())

    print("\nDone!")

    # return df_hourly


if __name__ == "__main__":
    main()
