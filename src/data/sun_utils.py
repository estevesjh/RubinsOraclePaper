import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

# Rubin Observatory location (Cerro Pachón, Chile)
RUBIN_SITE = EarthLocation.of_site("Rubin")
SUN_ALT_CUTOFF_TWILIGHT_MAX = 0.0
SUN_ALT_CUTOFF_TWILIGHT_MIN = -15.0


SUN_ALT_CUTOFF_SUNRISE_MAX = 0.0
SUN_ALT_CUTOFF_SUNRISE_MIN = -15.0


def get_sun_altitude(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute sun altitude at Rubin Observatory for given timestamps.

    Args:
        timestamps: DatetimeIndex of timestamps (UTC)

    Returns:
        Array of sun altitudes in degrees
    """
    times = Time(timestamps.values)
    altaz = AltAz(obstime=times, location=RUBIN_SITE)
    sun = get_sun(times).transform_to(altaz)
    return sun.alt.deg


def get_twilight_mask(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Get boolean mask for twilight hours (sun setting, alt between 0 and -15).

    Args:
        timestamps: DatetimeIndex of timestamps

    Returns:
        Boolean mask where True = twilight hour
    """
    sun_alt = get_sun_altitude(timestamps)
    sun_setting = np.diff(sun_alt, prepend=sun_alt[0]) < 0
    twilight_mask = (
        (sun_alt <= SUN_ALT_CUTOFF_TWILIGHT_MAX)
        & (sun_alt >= SUN_ALT_CUTOFF_TWILIGHT_MIN)
        & sun_setting
    )
    return twilight_mask


def get_sunrise_mask(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Get boolean mask for sunrise hours (sun rising, alt between 0 and 15).

    Args:
        timestamps: DatetimeIndex of timestamps

    Returns:
        Boolean mask where True = sunrise hour
    """
    sun_alt = get_sun_altitude(timestamps)
    sun_rising = np.diff(sun_alt, prepend=sun_alt[0]) > 0
    sunrise_mask = (
        (sun_alt >= SUN_ALT_CUTOFF_SUNRISE_MIN)
        & (sun_alt <= SUN_ALT_CUTOFF_SUNRISE_MAX)
        & sun_rising
    )
    return sunrise_mask


def compute_sunrise_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract mean temperature during sunrise for each day at Rubin Observatory.

    Sunrise: sun altitude in [0, 15] degrees and sun is rising.

    Args:
        df: DataFrame with timestamp and temperature columns (15-min resolution)

    Returns:
        DataFrame with date and sunrise_temp (one row per day)
    """
    # --- Make sure timestamps are timezone-aware UTC ---
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        "America/Santiago"
    )

    timestamps = pd.DatetimeIndex(df["timestamp"])

    print("Computing sun altitude and sunrise mask...")
    sunrise_mask = get_sunrise_mask(timestamps)

    # Use normalized UTC datetime for grouping/merging (NOT .dt.date)
    df["date"] = pd.to_datetime(timestamps.date)
    df["is_sunrise"] = sunrise_mask

    n_sunrise = int(sunrise_mask.sum())
    print(f"  Found {n_sunrise} sunrise samples out of {len(df)} total")

    print("Computing daily sunrise temperatures...")
    sunrise_data = (
        df[df["is_sunrise"]]
        .groupby("date", as_index=False)
        .agg(
            sunrise_temp=("y", "min"),
            sunrise_start=("timestamp", "min"),
            sunrise_end=("timestamp", "max"),
        )
    )

    # --- Handle days with no sunrise data: create full daily index and merge sunrise summary ---
    date_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )
    all_dates = pd.DataFrame({"date": date_range})

    result = all_dates.merge(sunrise_data, on="date", how="left")

    # Keep NaNs for days with no sunrise (DO NOT fill, DO NOT drop)
    print(
        f"  Returning {len(result)} days total; "
        f"{result['sunrise_temp'].isna().sum()} days have no sunrise data (NaN)."
    )

    return result


def compute_twilight_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract mean temperature during twilight for each day at Rubin Observatory.

    Twilight: sun altitude in [0, -15] degrees and sun is setting.

    Args:
        df: DataFrame with timestamp and temperature columns (15-min resolution)

    Returns:
        DataFrame with date and twilight_temp (one row per day)
    """
    # --- Make sure timestamps are timezone-aware UTC ---
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        "America/Santiago"
    )

    timestamps = pd.DatetimeIndex(df["timestamp"])

    print("Computing sun altitude and twilight mask...")
    twilight_mask = get_twilight_mask(timestamps)

    # Use normalized UTC datetime for grouping/merging (NOT .dt.date)
    df["date"] = pd.to_datetime(timestamps.date)
    df["is_twilight"] = twilight_mask

    n_twilight = int(twilight_mask.sum())
    print(f"  Found {n_twilight} twilight samples out of {len(df)} total")

    print("Computing daily twilight temperatures...")
    twilight_data = (
        df[df["is_twilight"]]
        .groupby("date", as_index=False)
        .agg(
            twilight_temp=("y", "mean"),
            twilight_start=("timestamp", "min"),
            twilight_end=("timestamp", "max"),
        )
    )

    # --- Handle days with no sunrise data: create full daily index and merge sunrise summary ---
    date_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )
    all_dates = pd.DataFrame({"date": date_range})

    result = all_dates.merge(twilight_data, on="date", how="left")

    # Keep NaNs for days with no twilight (DO NOT fill, DO NOT drop)
    print(
        f"  Returning {len(result)} days total; "
        f"{result['twilight_temp'].isna().sum()} days have no twilight data (NaN)."
    )

    return result
