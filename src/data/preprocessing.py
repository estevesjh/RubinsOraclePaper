"""Preprocessing functions for temperature forecast data."""

import numpy as np
import pandas as pd


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute data to hourly.

    Args:
        df: DataFrame with 15-minute data.

    Returns:
        DataFrame resampled to hourly frequency.
    """
    df = df.set_index("ds")

    hourly = pd.DataFrame()
    hourly["tempMax"] = df["tempMax"].resample("h").max()
    hourly["tempMin"] = df["tempMin"].resample("h").min()
    hourly["y"] = (hourly["tempMax"] + hourly["tempMin"]) / 2
    hourly["last_sunrise_temp"] = df["last_sunrise_temp"].resample("h").first()
    hourly["sunrise_temp"] = df["sunrise_temp"].resample("h").first()

    hourly = hourly.dropna(subset=["y", "last_sunrise_temp"])
    hourly["offset"] = hourly["y"] - hourly["last_sunrise_temp"]
    hourly = hourly.reset_index()

    return hourly


def resample_to_hourly_linear_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute data to hourly, preserving linear interpolation offset.

    Args:
        df: DataFrame with 15-minute data including baseline column.

    Returns:
        DataFrame resampled to hourly frequency with linear offset.
    """
    df = df.set_index("ds")

    hourly = pd.DataFrame()
    hourly["tempMax"] = df["tempMax"].resample("h").max()
    hourly["tempMin"] = df["tempMin"].resample("h").min()
    hourly["y"] = (hourly["tempMax"] + hourly["tempMin"]) / 2
    hourly["last_sunrise_temp"] = df["last_sunrise_temp"].resample("h").first()
    hourly["sunrise_temp"] = df["sunrise_temp"].resample("h").first()
    hourly["baseline"] = df["baseline"].resample("h").first()
    hourly["twilight_temp"] = df["twilight_temp"].resample("h").first()

    hourly = hourly.dropna(subset=["y", "baseline"])
    hourly["offset"] = hourly["y"] - hourly["baseline"]
    hourly = hourly.reset_index()

    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_sin and hour_cos features for neural models.

    Args:
        df: DataFrame with ds column.

    Returns:
        DataFrame with added time features.
    """
    hour = df["ds"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def add_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add yearly cyclical features (year_sin, year_cos).

    Args:
        df: DataFrame with ds column.

    Returns:
        DataFrame with added yearly features.
    """
    day_of_year = df["ds"].dt.dayofyear
    df["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["hour_cos"] = np.cos(2 * np.pi * df["ds"].dt.hour / 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["ds"].dt.hour / 24)
    return df


def add_twilight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add twilight-relative sin/cos features.

    Piecewise progress:
    - Daytime (sunrise to twilight): progress 0→1, duration = daylight hours
    - Nighttime (twilight to next sunrise): progress 1→2, duration = night hours

    twilight_sin = sin(π * progress) - peaks at mid-day, 0 at sunrise/twilight
    twilight_cos = cos(π * progress) - +1 at sunrise, -1 at twilight, +1 at next sunrise

    Args:
        df: DataFrame with ds, sunrise_temp, and twilight_temp columns.

    Returns:
        DataFrame with twilight_sin and twilight_cos features.
    """
    df = df.copy()

    # Get most recent sunrise (ffill)
    last_sunrise = df["ds"].where(df["sunrise_temp"].notna()).ffill()

    # Get most recent twilight (ffill)
    last_twilight = df["ds"].where(df["twilight_temp"].notna()).ffill()

    # Get next twilight (bfill) - for daytime duration
    next_twilight = df["ds"].where(df["twilight_temp"].notna()).bfill()

    # Get next sunrise (bfill) - for nighttime duration
    next_sunrise = df["ds"].where(df["sunrise_temp"].notna()).bfill()

    # Determine if daytime: last sunrise is more recent than last twilight
    # (i.e., sunrise happened after the previous twilight)
    is_daytime = last_sunrise > last_twilight

    # Handle edge case: before first twilight (last_twilight is NaT)
    is_daytime = is_daytime | last_twilight.isna()

    # Daylight duration: from last sunrise to next twilight
    daylight_seconds = (next_twilight - last_sunrise).dt.total_seconds()

    # Night duration: from last twilight to next sunrise
    night_seconds = (next_sunrise - last_twilight).dt.total_seconds()

    # Daytime progress: 0 (sunrise) to 1 (twilight)
    day_elapsed = (df["ds"] - last_sunrise).dt.total_seconds()
    day_progress = (day_elapsed / daylight_seconds).clip(0, 1)

    # Nighttime progress: 1 (twilight) to 2 (next sunrise)
    night_elapsed = (df["ds"] - last_twilight).dt.total_seconds()
    night_progress = 1 + (night_elapsed / night_seconds).clip(0, 1)

    progress = np.where(is_daytime, day_progress, night_progress)

    # Twilight-relative features (full cycle: 0→1→2 maps to cos/sin over 2π)
    df["twilight_sin"] = np.sin(np.pi * progress)
    df["twilight_cos"] = np.cos(np.pi * progress)

    return df


def train_test_split(
    df: pd.DataFrame, test_days: int = 365
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Args:
        df: DataFrame with ds column.
        test_days: Number of days to use for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    cutoff = df["ds"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["ds"] < cutoff].copy()
    test_df = df[df["ds"] >= cutoff].copy()
    return train_df, test_df


def get_sunrise_times(df: pd.DataFrame) -> pd.Series:
    """Identify sunrise hours from sunrise_temp column.

    Args:
        df: DataFrame with ds and sunrise_temp columns.

    Returns:
        Series of sunrise timestamps.
    """
    sunrise_mask = df["sunrise_temp"].notna()
    return df.loc[sunrise_mask, "ds"]


def get_valid_forecast_times(
    df: pd.DataFrame, sunrise_times: pd.Series
) -> list[pd.Timestamp]:
    """Get all valid forecast times: hourly from 1h after sunrise to 1h before next sunrise.

    Args:
        df: DataFrame with ds column (to filter valid times).
        sunrise_times: Series of sunrise timestamps.

    Returns:
        List of valid forecast timestamps.
    """
    valid_ds = set(df["ds"])
    sunrise_list = sorted(sunrise_times.tolist())
    forecast_times = []

    for i, sunrise in enumerate(sunrise_list[:-1]):
        next_sunrise = sunrise_list[i + 1]

        # Start 1h after sunrise, end 1h before next sunrise
        start_time = sunrise + pd.Timedelta(hours=1)
        end_time = next_sunrise - pd.Timedelta(hours=1)

        # Generate hourly forecast times
        current = start_time
        while current <= end_time:
            if current in valid_ds:
                forecast_times.append(current)
            current += pd.Timedelta(hours=1)

    return forecast_times


def get_next_sunrise(
    forecast_time: pd.Timestamp, sunrise_times: pd.Series
) -> pd.Timestamp | None:
    """Get the next sunrise after a forecast time.

    Args:
        forecast_time: Current forecast timestamp.
        sunrise_times: Series of sunrise timestamps.

    Returns:
        Next sunrise timestamp, or None if not found.
    """
    future = sunrise_times[sunrise_times > forecast_time]
    return future.iloc[0] if len(future) > 0 else None


def weighted_resample(
    df: pd.DataFrame,
    tau_hours: float = 7.0,
    sample_size_multiplier: float = 1.0,
    min_prob_floor: float = 1e-6,
    jitter: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap resample with exponential recency weighting and optional jitter.

    Args:
        df: DataFrame with ds, y columns. Optionally tempMax, tempMin for jitter.
        tau_hours: Half-life in hours for exponential decay.
        sample_size_multiplier: Output size = len(df) * multiplier.
        min_prob_floor: Minimum probability for oldest samples.
        jitter: If True, add Normal noise to y based on (tempMax-tempMin)/sqrt(12).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with resampled data plus all original rows.
    """
    if df.empty:
        return df

    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Recency weights
    t_end = df["ds"].max()
    ages_hours = (t_end - df["ds"]).dt.total_seconds() / 3600.0

    probs = np.exp(-ages_hours / max(tau_hours, 1e-9))
    probs = np.clip(probs, min_prob_floor, None)
    probs = probs / probs.sum()

    # Sample with replacement
    n = int(len(df) * max(sample_size_multiplier, 1.0 / len(df)))
    idx = rng.choice(df.index.to_numpy(), size=n, replace=True, p=probs)
    boot = df.loc[idx].copy()

    # Jitter y using observation uncertainty
    if jitter and {"tempMax", "tempMin"}.issubset(df.columns):
        rng_col = (boot["tempMax"] - boot["tempMin"]).astype(float).clip(lower=0.0)
        if rng_col.isna().any():
            global_range = (df["tempMax"] - df["tempMin"]).astype(float).clip(lower=0.0)
            fallback = float(np.nanmedian(global_range))
            rng_col = rng_col.fillna(fallback)
        sigma_obs = rng_col.to_numpy() / np.sqrt(12.0)
        eps = rng.normal(loc=0.0, scale=sigma_obs + 1e-12, size=len(boot))
        boot["y"] = boot["y"].to_numpy() + eps

    # Ensure all original timestamps present
    boot = pd.concat([boot, df], ignore_index=True)

    return boot


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-day temperature trend features (backward-looking only).

    Computes daily mean temps for past 3 days, then fits linear slope.
    This matches the approach in test_stubbs_features.py.

    Args:
        df: DataFrame with ds, y (temperature) columns at 15-min resolution.

    Returns:
        DataFrame with added trend features:
        - temp_mean_d1, temp_mean_d2, temp_mean_d3: Mean temp for each of last 3 days
        - temp_trend_3d: Linear slope over last 3 days (°C/day), positive = warming
    """
    df = df.copy()

    samples_per_day = 4 * 24  # 96 samples per day

    # Compute daily mean for each of the past 3 days
    # Day 1 = yesterday (24-48h ago), Day 2 = 2 days ago, Day 3 = 3 days ago
    for d in range(1, 4):
        # Shift by d days worth of samples, then take rolling mean of 1 day
        shifted = df["y"].shift(d * samples_per_day)
        df[f"temp_mean_d{d}"] = shifted.rolling(
            window=samples_per_day, min_periods=samples_per_day // 2
        ).mean()

    # Vectorized trend calculation (no apply!)
    # Linear fit to 3 points: days_ago = [1, 2, 3], temps = [d1, d2, d3]
    # Slope formula for 3 equally-spaced points: slope = (y3 - y1) / (x3 - x1) = (d3 - d1) / 2
    # But we want proper least squares: slope = sum((x-xmean)(y-ymean)) / sum((x-xmean)^2)
    # For x = [1,2,3]: xmean=2, (x-xmean) = [-1,0,1], sum((x-xmean)^2) = 2
    # slope = (-1*d1 + 0*d2 + 1*d3) / 2 = (d3 - d1) / 2
    # Negate for positive = warming: trend = -(d3 - d1) / 2 = (d1 - d3) / 2
    d1 = df["temp_mean_d1"]
    d3 = df["temp_mean_d3"]
    df["temp_trend_3d"] = (d1 - d3) / 2.0

    return df


def truncate_forecast_at_sunrise(
    forecast: pd.DataFrame, next_sunrise: pd.Timestamp, buffer_hours: int = 1
) -> pd.DataFrame:
    """Remove forecast steps that extend past buffer before next sunrise.

    Args:
        forecast: DataFrame with ds column.
        next_sunrise: Next sunrise timestamp.
        buffer_hours: Hours before sunrise to stop forecast.

    Returns:
        Truncated forecast DataFrame.
    """
    if next_sunrise is None:
        return forecast
    cutoff = next_sunrise - pd.Timedelta(hours=buffer_hours)

    # Normalize timestamps to be timezone-naive for comparison
    fc_times = pd.to_datetime(forecast["ds"])
    if fc_times.dt.tz is not None:
        fc_times = fc_times.dt.tz_localize(None)
    if hasattr(cutoff, "tzinfo") and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    return forecast[fc_times < cutoff].copy()


def add_trend_2h(df: pd.DataFrame, window_hours: float = 2.0) -> pd.DataFrame:
    """Add backward linear slope over 2 hours (vectorized).

    Computes the temperature rate of change using linear regression
    over a backward-looking window. Uses only past data (causal).

    Slope = cov(x,y) / var(x) where x is time indices.

    Args:
        df: DataFrame with ds and y columns at 15-min resolution.
        window_hours: Window size in hours (default 2.0).

    Returns:
        DataFrame with added trend_2h column (°C/hour).
    """
    df = df.copy()

    # Window size in samples (15-min resolution)
    dt = 0.25  # hours per sample
    window = int(window_hours / dt)  # 8 points for 2 hours

    y = df["y"]

    # Create index column for regression
    idx = pd.Series(np.arange(len(df)), index=df.index)

    # Rolling statistics
    y_mean = y.rolling(window, min_periods=window).mean()
    x_mean = idx.rolling(window, min_periods=window).mean()

    # Covariance: E[xy] - E[x]E[y]
    xy = y * idx
    xy_mean = xy.rolling(window, min_periods=window).mean()
    cov_xy = xy_mean - x_mean * y_mean

    # Variance of x: E[x^2] - E[x]^2
    x2_mean = (idx**2).rolling(window, min_periods=window).mean()
    var_x = x2_mean - x_mean**2

    # Slope in units per sample, convert to °C/hour
    slope_per_sample = cov_xy / var_x
    df["trend_2h"] = slope_per_sample / dt  # Convert to °C/hour

    return df


def add_trend_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Add backward linear slope over 1 hour (vectorized).

    Same as add_trend_2h but with 1-hour window.

    Args:
        df: DataFrame with ds and y columns at 15-min resolution.

    Returns:
        DataFrame with added trend_1h column (°C/hour).
    """
    df = df.copy()
    window_hours = 1.0
    dt = 0.25  # hours per sample
    window = int(window_hours / dt)  # 4 points for 1 hour

    y = df["y"]
    idx = pd.Series(np.arange(len(df)), index=df.index)

    y_mean = y.rolling(window, min_periods=window).mean()
    x_mean = idx.rolling(window, min_periods=window).mean()

    xy = y * idx
    xy_mean = xy.rolling(window, min_periods=window).mean()
    cov_xy = xy_mean - x_mean * y_mean

    x2_mean = (idx**2).rolling(window, min_periods=window).mean()
    var_x = x2_mean - x_mean**2

    slope_per_sample = cov_xy / var_x
    df["trend_1h"] = slope_per_sample / dt

    return df


def add_sunrise_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged sunrise daily trends.

    Computes temperature change between consecutive sunrises:
    - sunrise_daily_trend_1d: today's sunrise - yesterday's sunrise
    - sunrise_daily_trend_2d: yesterday's sunrise - 2 days ago
    - sunrise_daily_trend_3d: 2 days ago - 3 days ago

    Args:
        df: DataFrame with ds and sunrise_temp columns.

    Returns:
        DataFrame with added sunrise_daily_trend_1d/2d/3d columns.
    """
    df = df.copy()

    # Get sunrise rows only
    sunrise_mask = df["sunrise_temp"].notna()
    sunrise_df = df[sunrise_mask][["ds", "sunrise_temp"]].copy()

    # Compute daily trends (diff between consecutive sunrises)
    sunrise_df["trend_1d"] = sunrise_df["sunrise_temp"] - sunrise_df[
        "sunrise_temp"
    ].shift(1)
    sunrise_df["trend_2d"] = sunrise_df["sunrise_temp"].shift(1) - sunrise_df[
        "sunrise_temp"
    ].shift(2)
    sunrise_df["trend_3d"] = sunrise_df["sunrise_temp"].shift(2) - sunrise_df[
        "sunrise_temp"
    ].shift(3)

    # Merge back and forward-fill
    df = df.merge(
        sunrise_df[["ds", "trend_1d", "trend_2d", "trend_3d"]],
        on="ds",
        how="left",
    )
    df["sunrise_daily_trend_1d"] = df["trend_1d"].ffill()
    df["sunrise_daily_trend_2d"] = df["trend_2d"].ffill()
    df["sunrise_daily_trend_3d"] = df["trend_3d"].ffill()

    # Clean up temp columns
    df = df.drop(columns=["trend_1d", "trend_2d", "trend_3d"])

    return df


def add_temp_3d_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add 3-day temperature statistics.

    Args:
        df: DataFrame with y (temperature) column at 15-min resolution.

    Returns:
        DataFrame with added features:
        - temp_std_3d: Standard deviation over last 3 days
        - temp_min_3d: Minimum over last 3 days
        - temp_max_3d: Maximum over last 3 days
        - temp_range_3d: Range (max - min) over last 3 days
    """
    df = df.copy()
    WINDOW_3D = 288  # 3 days × 24h × 4 samples/h
    MIN_PERIODS = WINDOW_3D // 2

    df["temp_std_3d"] = df["y"].rolling(WINDOW_3D, min_periods=MIN_PERIODS).std()
    df["temp_min_3d"] = df["y"].rolling(WINDOW_3D, min_periods=MIN_PERIODS).min()
    df["temp_max_3d"] = df["y"].rolling(WINDOW_3D, min_periods=MIN_PERIODS).max()
    df["temp_range_3d"] = df["temp_max_3d"] - df["temp_min_3d"]
    return df


def add_lagged_twilight_temps(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous 1, 2, 3 twilight temperatures.

    Args:
        df: DataFrame with ds and twilight_temp columns.

    Returns:
        DataFrame with added T_tw_lag1, T_tw_lag2, T_tw_lag3 columns.
    """
    df = df.copy()
    tw_mask = df["twilight_temp"].notna()
    tw_df = df.loc[tw_mask, ["ds", "twilight_temp"]].copy()

    tw_df["T_tw_lag1"] = tw_df["twilight_temp"].shift(1)
    tw_df["T_tw_lag2"] = tw_df["twilight_temp"].shift(2)
    tw_df["T_tw_lag3"] = tw_df["twilight_temp"].shift(3)

    df = df.merge(
        tw_df[["ds", "T_tw_lag1", "T_tw_lag2", "T_tw_lag3"]], on="ds", how="left"
    )
    df["T_tw_lag1"] = df["T_tw_lag1"].ffill()
    df["T_tw_lag2"] = df["T_tw_lag2"].ffill()
    df["T_tw_lag3"] = df["T_tw_lag3"].ffill()
    return df


def add_rate_trend_2h(df: pd.DataFrame) -> pd.DataFrame:
    """Add 2-hour rate trend (temperature acceleration).

    Args:
        df: DataFrame with y_smooth column.

    Returns:
        DataFrame with added dT_dt_1h and rate_trend_2h columns.
    """
    df = df.copy()
    # 1-hour rate (°C/h): change over 4 samples (1 hour at 15-min resolution)
    df["dT_dt_1h"] = df["y_smooth"] - df["y_smooth"].shift(4)
    # 2-hour rate trend (acceleration): change in rate over 2 hours (8 samples)
    df["rate_trend_2h"] = df["dT_dt_1h"] - df["dT_dt_1h"].shift(8)
    return df


def add_key_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature at key time points, daylight hours, and all rate pairs.

    Computes:
    - last_midday_temp: Temperature at last midday (sunrise + daylight/2)
    - last_midnight_temp: Temperature at last midnight (twilight + nightlight/2)
    - daylight_hours: Hours of daylight (twilight - sunrise)

    Rate pairs (all 4 transitions in the daily cycle):
    - rate_sunrise_to_midday: (T_midday - T_sunrise) / (daylight/2) [°C/hour] - available after midday
    - rate_midday_to_twilight: (T_twilight - T_midday) / (daylight/2) [°C/hour] - available after twilight
    - rate_twilight_to_midnight: (T_midnight - T_twilight) / (nightlight/2) [°C/hour] - available after midnight
    - rate_midnight_to_sunrise: (T_sunrise - T_midnight) / (nightlight/2) [°C/hour] - available after sunrise

    Args:
        df: DataFrame with ds, y, sunrise_temp, twilight_temp columns.

    Returns:
        DataFrame with added features.
    """
    df = df.copy()

    # Get sunrise times and temps
    sunrise_mask = df["sunrise_temp"].notna()
    sunrise_rows = df[sunrise_mask][["ds", "sunrise_temp"]].copy()
    sunrise_list = list(zip(sunrise_rows["ds"], sunrise_rows["sunrise_temp"]))

    # Get twilight times and temps
    twilight_mask = df["twilight_temp"].notna()
    twilight_rows = df[twilight_mask][["ds", "twilight_temp"]].copy()
    twilight_list = list(zip(twilight_rows["ds"], twilight_rows["twilight_temp"]))

    # Compute midday temps and daylight hours
    midday_data = []
    for sr_time, sr_temp in sunrise_list:
        # Find next twilight after this sunrise (within 18 hours)
        next_tw_time = None
        next_tw_temp = None
        for tw_time, tw_temp in twilight_list:
            if tw_time > sr_time and (tw_time - sr_time).total_seconds() < 18 * 3600:
                next_tw_time = tw_time
                next_tw_temp = tw_temp
                break

        if next_tw_time is None:
            continue

        # Daylight hours
        daylight_hours = (next_tw_time - sr_time).total_seconds() / 3600

        # Midday time = sunrise + daylight/2
        midday_time = sr_time + pd.Timedelta(hours=daylight_hours / 2)

        # Find temperature at midday (closest row within 15 min)
        mask = (df["ds"] >= midday_time - pd.Timedelta(minutes=15)) & (
            df["ds"] <= midday_time + pd.Timedelta(minutes=15)
        )
        midday_rows = df[mask]

        if len(midday_rows) == 0:
            continue

        T_midday = midday_rows["y"].mean()

        # Rate from sunrise to midday [°C/hour] - available after midday
        rate_sunrise_to_midday = (
            (T_midday - sr_temp) / (daylight_hours / 2)
            if daylight_hours > 0
            else np.nan
        )

        # Rate from midday to twilight [°C/hour] - available after twilight
        rate_midday_to_twilight = (
            (next_tw_temp - T_midday) / (daylight_hours / 2)
            if daylight_hours > 0
            else np.nan
        )

        midday_data.append(
            {
                "midday_time": midday_time,
                "twilight_time": next_tw_time,
                "T_midday": T_midday,
                "daylight_hours": daylight_hours,
                "rate_sunrise_to_midday": rate_sunrise_to_midday,
                "rate_midday_to_twilight": rate_midday_to_twilight,
            }
        )

    # Compute midnight temps
    midnight_data = []
    for tw_time, tw_temp in twilight_list:
        # Find next sunrise after this twilight (within 18 hours)
        next_sr_time = None
        next_sr_temp = None
        for sr_time, sr_temp in sunrise_list:
            if sr_time > tw_time and (sr_time - tw_time).total_seconds() < 18 * 3600:
                next_sr_time = sr_time
                next_sr_temp = sr_temp
                break

        if next_sr_time is None:
            continue

        # Nightlight hours
        nightlight_hours = (next_sr_time - tw_time).total_seconds() / 3600

        # Midnight time = twilight + nightlight/2
        midnight_time = tw_time + pd.Timedelta(hours=nightlight_hours / 2)

        # Find temperature at midnight (closest row within 15 min)
        mask = (df["ds"] >= midnight_time - pd.Timedelta(minutes=15)) & (
            df["ds"] <= midnight_time + pd.Timedelta(minutes=15)
        )
        midnight_rows = df[mask]

        if len(midnight_rows) == 0:
            continue

        T_midnight = midnight_rows["y"].mean()

        # Rate from twilight to midnight [°C/hour] - available after midnight
        rate_twilight_to_midnight = (
            (T_midnight - tw_temp) / (nightlight_hours / 2)
            if nightlight_hours > 0
            else np.nan
        )

        # Rate from midnight to sunrise [°C/hour] - available after sunrise
        rate_midnight_to_sunrise = (
            (next_sr_temp - T_midnight) / (nightlight_hours / 2)
            if nightlight_hours > 0
            else np.nan
        )

        midnight_data.append(
            {
                "midnight_time": midnight_time,
                "sunrise_time": next_sr_time,  # Available after sunrise
                "T_midnight": T_midnight,
                "rate_twilight_to_midnight": rate_twilight_to_midnight,
                "rate_midnight_to_sunrise": rate_midnight_to_sunrise,
            }
        )

    # Initialize columns
    df["last_midday_temp"] = np.nan
    df["last_midnight_temp"] = np.nan
    df["daylight_hours"] = np.nan
    df["rate_sunrise_to_midday"] = np.nan
    df["rate_midday_to_twilight"] = np.nan
    df["rate_twilight_to_midnight"] = np.nan
    df["rate_midnight_to_sunrise"] = np.nan

    # 15-min range expressed as rate (°C/hour)
    # Divide by 0.25 hours (15 min interval) to get rate
    df["temp_range"] = (df["tempMax"] - df["tempMin"]) / 0.25

    # Rate of change of variability (acceleration of high-freq fluctuations)
    df["temp_high_freq_rate"] = df["temp_range"] - df["temp_range"].shift(1)

    # Summer-only temp_range (correlation is strongest in summer: -0.46)
    is_summer = df["ds"].dt.month.isin([6, 7, 8]).astype(int)
    df["temp_range_summer"] = df["temp_range"] * is_summer

    # Fill midday temps, daylight hours (available after midday)
    # Fill rate_sunrise_to_midday (available after midday)
    # Fill rate_midday_to_twilight (available after twilight)
    if midday_data:
        for md in midday_data:
            # After midday: midday temp, daylight hours, rate_sunrise_to_midday
            mask_midday = df["ds"] >= md["midday_time"]
            df.loc[mask_midday, "last_midday_temp"] = md["T_midday"]
            df.loc[mask_midday, "daylight_hours"] = md["daylight_hours"]
            df.loc[mask_midday, "rate_sunrise_to_midday"] = md["rate_sunrise_to_midday"]

            # After twilight: rate_midday_to_twilight
            mask_twilight = df["ds"] >= md["twilight_time"]
            df.loc[mask_twilight, "rate_midday_to_twilight"] = md[
                "rate_midday_to_twilight"
            ]

    # Fill midnight temps (available after sunrise)
    # Fill rate_twilight_to_midnight (available after midnight)
    # Fill rate_midnight_to_sunrise (available after sunrise)
    if midnight_data:
        for mn in midnight_data:
            # After midnight: rate_twilight_to_midnight
            mask_midnight = df["ds"] >= mn["midnight_time"]
            df.loc[mask_midnight, "rate_twilight_to_midnight"] = mn[
                "rate_twilight_to_midnight"
            ]

            # After sunrise: midnight temp, rate_midnight_to_sunrise
            mask_sunrise = df["ds"] >= mn["sunrise_time"]
            df.loc[mask_sunrise, "last_midnight_temp"] = mn["T_midnight"]
            df.loc[mask_sunrise, "rate_midnight_to_sunrise"] = mn[
                "rate_midnight_to_sunrise"
            ]

    return df


def add_temp_variability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature variability features from tempMax/tempMin.

    The 15-min range (tempMax - tempMin) captures intra-interval variability
    from high-frequency 30-sec sampling, which may indicate atmospheric instability.

    Args:
        df: DataFrame with tempMax and tempMin columns.

    Returns:
        DataFrame with added features:
        - temp_range: Variability rate (tempMax - tempMin) / 0.25 in °C/hour
        - temp_high_freq_rate: Change in temp_range between consecutive intervals
        - temp_range_summer: temp_range only for summer months (Jun-Aug), 0 otherwise
    """
    df = df.copy()

    # 15-min range expressed as rate (°C/hour)
    # Divide by 0.25 hours (15 min interval) to get rate
    df["temp_range"] = (df["tempMax"] - df["tempMin"]) / 0.25

    # Rate of change of variability (acceleration of high-freq fluctuations)
    df["temp_high_freq_rate"] = df["temp_range"] - df["temp_range"].shift(1)

    # Summer-only temp_range (correlation is strongest in summer: -0.46)
    is_summer = df["ds"].dt.month.isin([6, 7, 8]).astype(int)
    df["temp_range_summer"] = df["temp_range"] * is_summer

    return df
