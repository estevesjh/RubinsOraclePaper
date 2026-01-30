"""RidgeFeatureBuilder: Feature creation for Ridge correction model.

Takes NBEATSx prediction output and builds features for Ridge training/inference.

Two modes:
- "twilight": Predicts tw_slope for twilight temperature (input: twilight_offset_predictions.csv)
- "trajectory": Predicts res_base for temperature trajectory (input: trajectory predictions)
"""

import numpy as np
import pandas as pd


class RidgeFeatureBuilder:
    """Build features for Ridge correction model.

    Ridge corrects NBEATSx predictions using residual-based features.
    Input is NBEATSx prediction output DataFrame, not raw temperature data.

    Two modes:
    - "twilight": Target is tw_slope = tw_temp - T_tw_last (twilight forecast)
    - "trajectory": Target is res_base = temp_actual - T_pred_base (trajectory correction)

    Example:
        >>> # Twilight mode (default)
        >>> rfb = RidgeFeatureBuilder(mode="twilight")
        >>> df = rfb.build(df, sun_data_path='data/temp_history.csv')

        >>> # Trajectory mode
        >>> rfb = RidgeFeatureBuilder(mode="trajectory")
        >>> df = rfb.build(df)
    """

    # h_to_tw values for residual features (hours before twilight)
    # Range: 23 (just after prev_tw) to 3 (minimum lead time)
    DEFAULT_HTW_HOURS = list(range(23, 2, -1))  # 23, 22, 21, ..., 3

    # Feature columns for Ridge Twilight
    # res_htw_XX = residual at XX hours before twilight
    TWILIGHT_FEATURE_COLS = [
        "res",  # Current residual
        # Residuals at specific h_to_tw values (odd hours for efficiency)
        "res_htw_23", "res_htw_21", "res_htw_19", "res_htw_17", "res_htw_15",
        "res_htw_13", "res_htw_11", "res_htw_9", "res_htw_7", "res_htw_5", "res_htw_3",
        # Diurnal rates
        "rate_sunrise_to_midday", "rate_midday_to_twilight",
        "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
        # Other features
        "temp_since_sunrise",
        "day_length", "twilight_cos",
        "doy_sin", "doy_cos",
        "temp_trend_3d",
    ]

    # Feature columns for Ridge Trajectory (unchanged for now)
    TRAJECTORY_FEATURE_COLS = [
        "T_pred_base",  # Base temperature prediction
        "res_1h", "res_3h", "res_5h", "res_7h", "res_9h",  # Lagged residuals
        "res_11h", "res_13h", "res_15h", "res_17h", "res_19h", "res_21h",
        "temp_since_sunrise",
        "day_length", "twilight_cos",
        "doy_sin", "doy_cos",
        "temp_trend_3d",
    ]

    # Columns that can have NaN (filled with 0)
    FILLABLE_COLS = (
        [f"res_htw_{h}" for h in range(3, 24)] +  # h_to_tw based
        [f"res_{h}h" for h in range(1, 25)] +      # lag based (for trajectory)
        [f"temp_{h}h" for h in range(1, 25)]
    )

    def __init__(self, mode: str = "twilight", lag_hours: list = None, htw_hours: list = None):
        """Initialize RidgeFeatureBuilder.

        Args:
            mode: "twilight" (predicts tw_slope) or "trajectory" (predicts res_base)
            lag_hours: List of lag hours for trajectory mode (default: 1-24)
            htw_hours: List of h_to_tw hours for twilight mode (default: 23, 21, ..., 3)
        """
        if mode not in ("twilight", "trajectory"):
            raise ValueError(f"mode must be 'twilight' or 'trajectory', got '{mode}'")
        self.mode = mode
        self.lag_hours = lag_hours or list(range(1, 25))
        self.htw_hours = htw_hours or [23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3]

    def build(
        self,
        df: pd.DataFrame,
        sun_data_path: str = None,
        sun_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Build all Ridge features.

        Args:
            df: NBEATSx predictions DataFrame with columns:
                Twilight mode:
                - tw_time, forecast_time, target_time
                - res (or res_tw): NBEATSx residual
                - T_tw_last, tw_temp: twilight temperatures
                - rate_* features, temp_since_sunrise, temp_trend_3d
                Trajectory mode:
                - tw_time, forecast_time, target_time
                - T_pred_base, temp_actual: base prediction and actual temp
                - temp_since_sunrise, trend_temp_3d
            sun_data_path: Path to temperature data with sunrise/twilight times
            sun_df: Pre-loaded sun data DataFrame (alternative to path)

        Returns:
            DataFrame with Ridge features added
        """
        df = df.copy()

        # Ensure timestamps are datetime
        for col in ["tw_time", "forecast_time", "target_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Mode-specific residual column handling
        if self.mode == "twilight":
            # Rename res_tw to res if needed
            if "res_tw" in df.columns and "res" not in df.columns:
                df = df.rename(columns={"res_tw": "res"})
        else:
            # Trajectory mode: compute res_base if missing
            if "res_base" not in df.columns and "temp_actual" in df.columns:
                df["res_base"] = df["temp_actual"] - df["T_pred_base"]

        # Step 1: Add prediction horizon if missing
        df = self._add_pred_horizon(df)

        # Step 2: Build nowcast lookup and add lagged residuals
        df = self._add_lagged_residuals(df)

        # Step 3: Add target
        df = self._add_target(df)

        # Step 4: Add seasonal features (doy_sin, doy_cos)
        df = self._add_seasonal_features(df)

        # Step 5: Add twilight_cos and day_length (requires sun data)
        if sun_data_path is not None or sun_df is not None:
            df = self._add_twilight_features(df, sun_data_path, sun_df)

        # Step 6: Fill NaN in lagged residuals with 0
        for col in self.FILLABLE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _add_pred_horizon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction horizon if missing."""
        if "pred_horizon" not in df.columns:
            df["pred_horizon"] = (
                df["target_time"] - df["forecast_time"]
            ).dt.total_seconds() / 3600
        return df

    def _add_lagged_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add residual features based on h_to_tw (hours to twilight).

        For twilight mode: res_htw_XX = residual at XX hours before twilight
        For trajectory mode: res_Xh = residual at X hours before current time

        No future data leakage: only use res_htw values >= current h_to_tw.
        """
        res_col = "res" if self.mode == "twilight" else "res_base"

        if self.mode == "twilight":
            # Build lookup: (tw_time, h_to_tw_rounded) → res
            # Round h_to_tw to nearest integer for lookup
            df["h_to_tw_rounded"] = df["h_to_tw"].round().astype(int)

            # Create lookup DataFrame for each twilight
            res_by_htw = df.groupby(["tw_time", "h_to_tw_rounded"])[res_col].first().reset_index()
            res_by_htw = res_by_htw.rename(columns={res_col: "res_value"})

            # Add res_htw_XX features (only past values, h_to_tw >= current)
            for htw in self.htw_hours:
                col_name = f"res_htw_{htw}"
                # Get res at this h_to_tw for each twilight
                htw_df = res_by_htw[res_by_htw["h_to_tw_rounded"] == htw][["tw_time", "res_value"]]
                htw_df = htw_df.rename(columns={"res_value": col_name})
                df = df.merge(htw_df, on="tw_time", how="left")
                # Mask future values (where htw < current h_to_tw)
                df.loc[df["h_to_tw_rounded"] > htw, col_name] = np.nan

        else:
            # Trajectory mode: use time-based lagged residuals
            idx_nowcast = df.groupby(["tw_time", "forecast_time"])["pred_horizon"].idxmin()
            nowcast_df = df.loc[idx_nowcast].copy()

            nowcast_df["target_time_rounded"] = nowcast_df["target_time"].dt.round("15min")
            res_lookup = dict(zip(nowcast_df["target_time_rounded"], nowcast_df[res_col]))

            base_time_col = "forecast_time"
            for lag_h in self.lag_hours:
                col_name = f"res_{lag_h}h"
                lagged_time = (df[base_time_col] - pd.Timedelta(hours=lag_h)).dt.round("15min")
                df[col_name] = lagged_time.map(res_lookup)

            # Also build lagged actual temperatures
            temp_lookup = dict(zip(nowcast_df["target_time_rounded"], nowcast_df["temp_actual"]))
            for lag_h in self.lag_hours:
                col_name = f"temp_{lag_h}h"
                lagged_time = (df[base_time_col] - pd.Timedelta(hours=lag_h)).dt.round("15min")
                df[col_name] = lagged_time.map(temp_lookup)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ridge target based on mode.

        Twilight: tw_slope = tw_temp - T_tw_last
        Trajectory: res_base = temp_actual - T_pred_base (already computed in build)
        """
        if self.mode == "twilight":
            df["tw_slope"] = df["tw_temp"] - df["T_tw_last"]
        # Trajectory mode: res_base already computed in build()
        return df

    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day-of-year seasonal encoding.

        Uses tw_time_local if available (for proper local date),
        otherwise converts tw_time from UTC to Chile local time.
        """
        if "tw_time_local" in df.columns:
            # Use local time column if provided
            tw_local = df["tw_time_local"]
        else:
            # Convert UTC tw_time to Chile local time for day-of-year
            tw_utc = pd.to_datetime(df["tw_time"]).dt.tz_localize("UTC")
            tw_local = tw_utc.dt.tz_convert("America/Santiago").dt.tz_localize(None)

        df["doy"] = tw_local.dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
        return df

    def _add_twilight_features(
        self,
        df: pd.DataFrame,
        sun_data_path: str = None,
        sun_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Add twilight_cos and day_length from sun data.

        Twilight cycle encoding:
        - At sunrise: progress=0, twilight_cos=+1
        - At midday: progress=0.5, twilight_cos=0
        - At twilight: progress=1.0, twilight_cos=-1
        - At midnight: progress=1.5, twilight_cos=0
        - At next sunrise: progress=2.0, twilight_cos=+1

        Args:
            df: DataFrame with tw_time, target_time, prev_tw_time columns
            sun_data_path: Path to CSV with sunrise_temp, twilight_temp columns
            sun_df: Pre-loaded DataFrame (alternative to path)
        """
        # Load sun data
        if sun_df is None:
            if sun_data_path is None:
                return df  # Skip if no sun data provided
            sun_df = pd.read_csv(sun_data_path)

        # Ensure timestamp column exists
        sun_df = sun_df.copy()
        if "ds" in sun_df.columns and "timestamp" not in sun_df.columns:
            sun_df["timestamp"] = sun_df["ds"]

        # Extract sunrise times
        sunrise_times = sun_df[sun_df["sunrise_temp"].notna()][["timestamp"]].copy()
        sunrise_times = sunrise_times.rename(columns={"timestamp": "sunrise_time"})
        sunrise_times["sunrise_time"] = sunrise_times["sunrise_time"].values.astype('datetime64[ns]')

        # Ensure df timestamps are timezone-free
        df = df.copy()
        df["tw_time"] = df["tw_time"].values.astype('datetime64[ns]')
        df["target_time"] = df["target_time"].values.astype('datetime64[ns]')
        if "prev_tw_time" in df.columns:
            df["prev_tw_time"] = df["prev_tw_time"].values.astype('datetime64[ns]')

        # Build lookup for each twilight: sunrise_time, day_length, night_length
        tw_info = []
        for tw_time in df["tw_time"].unique():
            # Find sunrise within 18h before twilight (same day)
            mask_sr = (sunrise_times["sunrise_time"] < tw_time) & (
                sunrise_times["sunrise_time"] >= tw_time - pd.Timedelta(hours=18)
            )
            nearby_sr = sunrise_times[mask_sr]

            if len(nearby_sr) > 0:
                sr_time = nearby_sr["sunrise_time"].max()
                day_len = (tw_time - sr_time).total_seconds() / 3600

                # Find next sunrise (for night duration)
                mask_next_sr = sunrise_times["sunrise_time"] > tw_time
                next_sr = sunrise_times[mask_next_sr]
                if len(next_sr) > 0:
                    next_sr_time = next_sr["sunrise_time"].min()
                    night_len = (next_sr_time - tw_time).total_seconds() / 3600
                else:
                    night_len = 24 - day_len
                    next_sr_time = tw_time + pd.Timedelta(hours=night_len)

                tw_info.append({
                    "tw_time": tw_time,
                    "sunrise_time": sr_time,
                    "next_sunrise_time": next_sr_time,
                    "day_length": day_len,
                    "night_length": night_len,
                })
            else:
                tw_info.append({
                    "tw_time": tw_time,
                    "sunrise_time": pd.NaT,
                    "next_sunrise_time": pd.NaT,
                    "day_length": np.nan,
                    "night_length": np.nan,
                })

        tw_info_df = pd.DataFrame(tw_info)
        df = df.merge(tw_info_df, on="tw_time", how="left")

        # Compute twilight_cos based on day/night progress
        # Three cases:
        # 1. Daytime: sunrise <= target_time < tw_time
        # 2. Nighttime after twilight: target_time >= tw_time
        # 3. Nighttime before sunrise: target_time < sunrise (use prev_tw_time)

        is_daytime = (df["target_time"] >= df["sunrise_time"]) & (df["target_time"] < df["tw_time"])
        is_night_after_tw = df["target_time"] >= df["tw_time"]
        is_night_before_sunrise = df["target_time"] < df["sunrise_time"]

        # Day progress: 0 at sunrise → 1 at twilight
        day_elapsed = (df["target_time"] - df["sunrise_time"]).dt.total_seconds() / 3600
        day_progress = (day_elapsed / df["day_length"]).clip(0, 1)

        # Night progress after twilight: 1 at twilight → 2 at next sunrise
        night_elapsed_after = (df["target_time"] - df["tw_time"]).dt.total_seconds() / 3600
        night_progress_after = 1 + (night_elapsed_after / df["night_length"]).clip(0, 1)

        # Night progress before sunrise (use prev_tw_time if available)
        # This handles nowcasts that are between prev_tw and sunrise
        if "prev_tw_time" in df.columns:
            # For night before sunrise, prev_tw_time is the reference
            # Need to find the night_length from prev_tw to sunrise
            night_elapsed_before = (df["target_time"] - df["prev_tw_time"]).dt.total_seconds() / 3600
            # Night length from prev_tw to current sunrise
            night_len_before = (df["sunrise_time"] - df["prev_tw_time"]).dt.total_seconds() / 3600
            night_len_before = night_len_before.replace(0, np.nan)  # Avoid division by zero
            night_progress_before = 1 + (night_elapsed_before / night_len_before).clip(0, 1)
        else:
            # Fallback: use negative day_elapsed to estimate night progress
            # This is approximate but better than -1
            night_progress_before = 1 + (1 + day_elapsed / df["night_length"]).clip(0, 1)

        # Combine progress based on time of day
        progress = np.where(
            is_daytime,
            day_progress,
            np.where(
                is_night_before_sunrise,
                night_progress_before,
                night_progress_after
            )
        )

        df["twilight_cos"] = np.cos(np.pi * progress)

        return df

    def get_feature_cols(self) -> list:
        """Return list of feature columns for Ridge model based on mode."""
        if self.mode == "twilight":
            return self.TWILIGHT_FEATURE_COLS.copy()
        else:
            return self.TRAJECTORY_FEATURE_COLS.copy()

    def get_target_col(self) -> str:
        """Return target column name based on mode."""
        return "tw_slope" if self.mode == "twilight" else "res_base"

    def get_fillable_cols(self) -> list:
        """Return list of columns that can have NaN (filled with 0)."""
        return self.FILLABLE_COLS.copy()


# =============================================================================
# Standalone functions for DataFrame building (used by trainer and forecaster)
# =============================================================================

def extract_nowcasts(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Extract nowcast rows from NBEATSx predictions.

    Nowcast = prediction where target_time == forecast_time (lead = 0).
    Used for Ridge-Tw training.

    Args:
        pred_df: DataFrame with NBEATSx predictions (must have target_time, forecast_time)

    Returns:
        DataFrame with nowcast rows only
    """
    if "target_time" not in pred_df.columns or "forecast_time" not in pred_df.columns:
        raise ValueError("pred_df must have 'target_time' and 'forecast_time' columns")

    nowcasts = pred_df[pred_df["target_time"] == pred_df["forecast_time"]].copy()
    return nowcasts


def build_ridge_df(
    nowcasts_df: pd.DataFrame,
    sun_df: pd.DataFrame,
    mode: str = "twilight",
) -> pd.DataFrame:
    """Build Ridge features on nowcasts DataFrame.

    Adds lagged residuals, twilight_cos, seasonal features.
    Used for Ridge-Tw training.

    Args:
        nowcasts_df: DataFrame with nowcast predictions
        sun_df: DataFrame with sunrise/twilight features (from FeatureBuilder)
        mode: "twilight" or "trajectory"

    Returns:
        DataFrame with Ridge features added
    """
    ridge_fb = RidgeFeatureBuilder(mode=mode)
    ridge_df = ridge_fb.build(nowcasts_df, sun_df=sun_df)

    # Add tw_slope target for twilight mode
    if mode == "twilight" and "tw_temp" in ridge_df.columns and "T_tw_last" in ridge_df.columns:
        ridge_df["tw_slope"] = ridge_df["tw_temp"] - ridge_df["T_tw_last"]

    return ridge_df


def build_traj_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Build trajectory predictions DataFrame.

    Contains all predictions at various lead times (0.5h to 12h).
    Used for Ridge-Traj training.

    Args:
        pred_df: DataFrame with NBEATSx predictions

    Returns:
        Copy of pred_df (unchanged, for consistency)
    """
    return pred_df.copy()


def get_traj_features_for_lead(lead: int, base_features: list, htw_hours: list) -> list:
    """Get feature columns available at a specific lead time for trajectory correction.

    At lead time L, only res_htw_XX features where XX >= L are available
    (no future data leakage).

    Args:
        lead: Lead time in hours (e.g., 3, 6, 9, 12)
        base_features: List of non-residual feature columns
        htw_hours: List of h_to_tw hours for residual features (e.g., [23, 21, 19, ...])

    Returns:
        List of feature columns available at this lead time
    """
    available_htw = [h for h in htw_hours if h >= lead]
    htw_features = [f"res_htw_{h}" for h in available_htw]
    return base_features + htw_features


def compute_ridge_tw_rmse(
    ridge_df: pd.DataFrame,
    models: dict,
    scalers: dict,
    lead_hours: list,
    feature_cols: list = None,
) -> dict:
    """Compute RMSE at each lead time for Ridge-Tw (twilight prediction).

    Ridge-Tw predicts tw_slope = T_tw - T_tw_last
    T_tw_pred = T_tw_last + slope_pred

    This is a standalone function for DRY - can be used by trainer and forecaster.

    Args:
        ridge_df: DataFrame with Ridge features (from build_ridge_df)
        models: Dict of {lead: Ridge model}
        scalers: Dict of {lead: StandardScaler}
        lead_hours: List of lead times to evaluate
        feature_cols: List of feature columns (default: TWILIGHT_FEATURE_COLS)

    Returns:
        Dict of {lead: {"rmse": float, "bias": float, "n": int}}
    """
    if feature_cols is None:
        feature_cols = RidgeFeatureBuilder.TWILIGHT_FEATURE_COLS

    # Filter to features that exist in ridge_df
    available_features = [c for c in feature_cols if c in ridge_df.columns]

    # Ensure tw_slope is computed
    if "tw_slope" not in ridge_df.columns:
        ridge_df = ridge_df.copy()
        ridge_df["tw_slope"] = ridge_df["tw_temp"] - ridge_df["T_tw_last"]

    results = {}

    for lead in lead_hours:
        if lead not in models:
            continue

        # Filter by Ridge decision lead time: h_to_tw >= lead
        lead_df = ridge_df[ridge_df["h_to_tw"] >= lead].copy()

        if len(lead_df) == 0:
            continue

        # Select ONE row per twilight (most recent target_time)
        idx_recent = lead_df.groupby("tw_time")["target_time"].idxmax()
        tw_df = lead_df.loc[idx_recent].copy()

        # Filter valid features
        valid_mask = tw_df[available_features].notna().all(axis=1)
        valid_df = tw_df[valid_mask].copy()

        if len(valid_df) < 5:
            continue

        # Predict tw_slope
        model = models[lead]
        scaler = scalers[lead]

        X = valid_df[available_features].values
        X_scaled = scaler.transform(X)
        slope_pred = model.predict(X_scaled)

        # T_tw_pred = T_tw_last + slope_pred
        T_tw_pred = valid_df["T_tw_last"].values + slope_pred
        errors = valid_df["tw_temp"].values - T_tw_pred

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))

        results[lead] = {"rmse": rmse, "bias": bias, "n": len(errors)}

    return results
