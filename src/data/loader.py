"""Data loader for temperature forecast data."""

from pathlib import Path

import pandas as pd


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load temperature data from CSV.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame with parsed timestamps and computed offset.
    """
    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["timestamp"])
    df["offset"] = df["y"] - df["last_sunrise_temp"]
    df = df.dropna(subset=["y", "last_sunrise_temp"])
    return df


def load_data_linear_offset(data_path: str | Path) -> pd.DataFrame:
    """Load temperature data with linear interpolation offset.

    The offset is computed as the deviation from a linear interpolation
    between consecutive sunrise temperatures.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame with parsed timestamps and linear offset.
    """
    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["timestamp"])

    # Get last and next sunrise temps via ffill/bfill
    # Shift before bfill so at sunrise time we get the NEXT sunrise, not current
    df["last_sunrise_temp"] = df["sunrise_temp"].ffill()
    df["next_sunrise_temp"] = df["sunrise_temp"].shift(-1).bfill()

    # Get last and next sunrise times (same shift logic)
    sunrise_times = df["ds"].where(df["sunrise_temp"].notna())
    df["last_sunrise_time"] = sunrise_times.ffill()
    df["next_sunrise_time"] = sunrise_times.shift(-1).bfill()

    # Compute progress between sunrises (0 at last sunrise, 1 at next sunrise)
    total_duration = (df["next_sunrise_time"] - df["last_sunrise_time"]).dt.total_seconds()
    elapsed = (df["ds"] - df["last_sunrise_time"]).dt.total_seconds()
    df["progress"] = elapsed / total_duration

    # Linear interpolation baseline
    df["baseline"] = df["last_sunrise_temp"] + df["progress"] * (
        df["next_sunrise_temp"] - df["last_sunrise_temp"]
    )

    # Offset = deviation from linear baseline
    df["offset"] = df["y"] - df["baseline"]

    # Clean up
    df = df.dropna(subset=["y", "baseline"])

    # Preserve twilight_temp for twilight event detection
    # (already loaded from CSV, just ensure it's kept)

    return df
