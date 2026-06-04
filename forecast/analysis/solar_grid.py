"""Solar-grid utilities for causal twilight forecasting.

Ported from rubin-twilight-forecast/twilight/features.py + utils.py.
Provides: sun altitude, event detection, solar-time coordinates,
and resampling onto a regular solar-time grid.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u

# Observatory site
RUBIN_SITE = EarthLocation(
    lat=-30.2446 * u.deg,
    lon=-70.7494 * u.deg,
    height=2663.0 * u.m,
)
CHILE_TZ = "America/Santiago"
SUN_ALT_MIDPOINT = -15.0


def get_sun_altitude(timestamps):
    """Sun altitude in degrees at Rubin for UTC timestamps."""
    times = Time(timestamps.values)
    altaz = AltAz(obstime=times, location=RUBIN_SITE)
    return get_sun(times).transform_to(altaz).alt.deg


def detect_events(df, alt_col="alt_sun", midpoint=SUN_ALT_MIDPOINT):
    """Set event flags from zero-crossings of sun altitude.

    event_sunrise: alt crosses 0 rising
    event_sunset: alt crosses 0 setting
    twilight_event_sunset: alt crosses midpoint (-15) setting
    """
    df = df.copy()
    alt = df[alt_col]
    alt_prev = alt.shift(1)

    df["event_sunrise"] = (alt_prev < 0) & (alt >= 0)
    df["event_sunset"] = (alt_prev >= 0) & (alt < 0)

    adjusted = alt - midpoint
    adjusted_prev = adjusted.shift(1)
    df["twilight_event_sunset"] = (adjusted_prev >= 0) & (adjusted < 0)

    return df


def add_solar_time(df):
    """Add SolarTime, DayCount, solarDayHour columns.

    SolarTime in [0,1): 0=sunrise, 0.25=midday, 0.5=sunset, 0.75=solar midnight.
    DayCount increments at each sunrise (first=0).
    solarDayHour = DayCount + SolarTime (monotonically increasing).
    """
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    last_sunrise = ds.where(df["event_sunrise"]).ffill()
    next_sunrise = ds.where(df["event_sunrise"]).bfill()
    last_sunset = ds.where(df["event_sunset"]).ffill()
    next_sunset = ds.where(df["event_sunset"]).bfill()

    is_day = (last_sunrise > last_sunset) | last_sunset.isna()

    def hrs(a, b):
        return (a - b).dt.total_seconds() / 3600.0

    day_frac = 0.5 * hrs(ds, last_sunrise) / hrs(next_sunset, last_sunrise)
    night_frac = 0.5 + 0.5 * hrs(ds, last_sunset) / hrs(next_sunrise, last_sunset)
    df["SolarTime"] = np.where(is_day, day_frac, night_frac)

    df["DayCount"] = df["event_sunrise"].astype(int).cumsum() - 1
    df["solarDayHour"] = df["DayCount"] + df["SolarTime"]

    df = df[df["DayCount"] >= 0].copy()
    df = df[df["solarDayHour"].notna()].copy()
    return df


def resample_to_solar_grid(df, step=0.01, fillna=True):
    """Interpolate all numeric features onto a regular solarDayHour grid.

    Synthetic clock: each solar day = 24h, sunrise at 06:00.
    step=0.01 gives 100 points per solar day.
    """
    df = df.sort_values("solarDayHour")
    df = df[~df["solarDayHour"].duplicated(keep="first")]

    x = df["solarDayHour"].to_numpy(dtype=float)
    grid = np.arange(0.0, float(x.max()), step)

    out = pd.DataFrame({"solarDayHour": grid})
    out["DayCount"] = np.floor(grid).astype(int)
    out["SolarTime"] = grid - out["DayCount"]

    drop = {
        "solarDayHour", "DayCount", "SolarTime", "ds", "ds_real", "ds_local",
        "event_sunrise", "event_sunset", "twilight_event_sunset",
    }
    num_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in num_cols:
        out[c] = np.interp(grid, x, df[c].to_numpy(dtype=float))

    # Synthetic regular solar clock
    first_sunrise = pd.Timestamp(df["ds"].iloc[0])
    base = first_sunrise.normalize() + pd.Timedelta(hours=6)
    step_delta = pd.Timedelta(seconds=round(step * 24.0 * 3600.0))
    out["ds"] = base + np.arange(len(grid)) * step_delta

    # Preserve real UTC wall-clock time
    ds_ns = pd.to_datetime(df["ds"]).astype("int64").to_numpy()
    out["ds_real"] = pd.to_datetime(np.interp(grid, x, ds_ns).astype("int64"))

    if fillna:
        out[num_cols] = out[num_cols].ffill().bfill()

    # Regenerate event flags from interpolated alt_sun
    if "alt_sun" in out.columns:
        out = detect_events(out)

    return out


def find_twilight_targets(df):
    """Find grid indices where alt_sun crosses -15 deg (setting).

    Returns DataFrame with one row per twilight event:
    columns: grid_idx, ds_real, y_actual (temperature at -15 setting).
    """
    mask = df["twilight_event_sunset"].values.astype(bool)
    idxs = np.where(mask)[0]

    events = []
    for i in idxs:
        events.append({
            "grid_idx": i,
            "solarDayHour": df["solarDayHour"].iloc[i],
            "ds_real": df["ds_real"].iloc[i],
            "DayCount": df["DayCount"].iloc[i],
            "y_actual": df["y"].iloc[i],
        })
    return pd.DataFrame(events)
