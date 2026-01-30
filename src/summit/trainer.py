"""NbeatsxTrainer: Train and cache NBEATSx and Ridge models.

Trains:
1. NBEATSx - Delta T prediction model
2. Ridge Twilight - Correction at 3h lead time
3. Ridge Trajectory - Correction for all lead times
"""

import os
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Constants for batch prediction
FREQ_MINUTES = 15
FREQ_SECONDS = FREQ_MINUTES * 60

from .features import FeatureBuilder
from .ridge_features import (
    RidgeFeatureBuilder,
    extract_nowcasts,
    build_ridge_df,
    build_traj_df,
    compute_ridge_tw_rmse,
    get_traj_features_for_lead,
)
from .sun_utils import get_sun_altitude

# Import predict_batch from models for consistent predictions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.nbeats import predict_batch

# Default configuration (used if no YAML provided)
DEFAULT_CONFIG = {
    "data": {
        "freq": "15min",
        "test_start_date": None,  # None = use all data for training
    },
    "nbeats": {
        "horizon": 52,
        "input_size": 96,
        "max_steps": 500,
        "activation": "SELU",
        "scaler_type": "robust",
        "learning_rate": 0.01,
        "stack_types": ["trend", "seasonality", "identity", "exogenous"],
        "n_blocks": [1, 1, 1, 1],
        "mlp_units": [[32, 32], [32, 32], [32, 32], [32, 32]],
        "hist_exog": [
            "temp_raw",
            "temp_last_sunrise",
            "trend_2h",
            "temp_trend_3d",
            "rate_sunrise_to_midday",
            "rate_midday_to_twilight",
            "rate_twilight_to_midnight",
            "rate_midnight_to_sunrise",
        ],
        "futr_exog": ["twilight_cos"],
        "cache_dir": "results/paper_nbeats_cache",
        "cache_prefix": "NBEATSx_deltaT_v2",
    },
    "ridge_twilight": {
        "alpha": 1.0,
        "lead_hours": [h / 2 for h in range(1, 25)],  # 0.5 to 12.0 (any time after sunrise)
        "target": "tw_slope",
        # 20 features for Ridge correction
        "features": [
            # Core (1)
            "res",
            # Lagged Residuals (9)
            "res_1h",
            "res_3h",
            "res_5h",
            "res_7h",
            "res_9h",
            "res_11h",
            "res_13h",
            "res_15h",
            "res_17h",
            # Diurnal Rates (4)
            "rate_sunrise_to_midday",
            "rate_midday_to_twilight",
            "rate_twilight_to_midnight",
            "rate_midnight_to_sunrise",
            # Temperature (1)
            "temp_since_sunrise",
            # Seasonal (4)
            "day_length",
            "twilight_cos",
            "doy_sin",
            "doy_cos",
            # Trend (1)
            "temp_trend_3d",
        ],
        "cache_dir": "results/paper_model_cache/sklearn_hourly_temps_with_trend",
        "cache_pattern": "linear_lead_{lead}h.joblib",
    },
    "ridge_trajectory": {
        "alpha": 1.0,
        "lead_hours": [h / 2 for h in range(1, 25)],  # 0.5 to 12.0
        "target": "tw_slope",
        # Same 20 features as ridge_twilight
        "features": [
            "res",
            "res_1h",
            "res_3h",
            "res_5h",
            "res_7h",
            "res_9h",
            "res_11h",
            "res_13h",
            "res_15h",
            "res_17h",
            "rate_sunrise_to_midday",
            "rate_midday_to_twilight",
            "rate_twilight_to_midnight",
            "rate_midnight_to_sunrise",
            "temp_since_sunrise",
            "day_length",
            "twilight_cos",
            "doy_sin",
            "doy_cos",
            "temp_trend_3d",
        ],
        "cache_dir": "results/paper_model_cache/traj_ridge",
        "cache_pattern": "traj_ridge_lead_{lead}h.joblib",
    },
}


class NbeatsxTrainer:
    """Train NBEATSx and Ridge models for twilight forecasting.

    Example:
        >>> trainer = NbeatsxTrainer(
        ...     'data/temp_history_jan2026.csv',
        ...     config_path='configs/trainer_config.yaml'
        ... )
        >>> trainer.train_all(use_cache=False)
        >>> trainer.print_summary()
    """

    def __init__(self, data_path: str, config_path: str = None):
        """Initialize trainer.

        Args:
            data_path: Path to CSV data file
            config_path: Path to YAML config file (uses defaults if None)
        """
        self.data_path = Path(data_path)
        self.root_path = Path(
            __file__
        ).parent.parent.parent  # src/summit -> src -> root
        self.config = self._load_config(config_path)

        # Will be populated by load_data()
        self.df = None
        self.cutoff_date = None

        # Feature builder
        self.feature_builder = FeatureBuilder()

        # Cached DataFrames (populated during training)
        self.df_features = None      # df with NBEATSx features added
        self.nowcasts_df = None      # Nowcasts only (target_time == forecast_time)
        self.ridge_df = None         # Nowcasts with Ridge features
        self.traj_df = None          # All predictions for trajectory

        # Test data DataFrames (populated by compute_test_rmse_by_lead)
        self.test_predictions = None
        self.test_nowcasts_df = None
        self.test_ridge_df = None

        # Cache paths for predictions
        self._cache_dir = self.root_path / "results/paper_nbeats_cache"
        self._insample_cache_path = self._cache_dir / "insample_predictions.parquet"
        self._test_cache_path = self._cache_dir / "test_predictions.parquet"

        # Trained models
        self._nbeats_model = None
        self._ridge_model = None
        self._ridge_scaler = None
        # Ridge trajectory models (predict res)
        self._traj_ridge_models = {}
        self._traj_ridge_scalers = {}
        self._traj_ridge_features = {}
        # Ridge twilight models (predict tw_slope)
        self._tw_ridge_models = {}
        self._tw_ridge_scalers = {}

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file or use defaults.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            return DEFAULT_CONFIG.copy()

        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Merge with defaults (user config overrides defaults)
        config = DEFAULT_CONFIG.copy()
        for section in ["data", "nbeats", "ridge_twilight", "ridge_trajectory"]:
            if section in user_config:
                config[section] = {**config.get(section, {}), **user_config[section]}

        return config

    def load_data(self) -> pd.DataFrame:
        """Load data, skip alt_sun computation if column exists.

        Returns:
            DataFrame with columns: ds, y, alt_sun
        """
        print(f"\n[Data] Loading from {self.data_path.name}...")
        df = pd.read_csv(self.data_path)

        # Standardize column names
        # ds: UTC timestamps (tz-naive) for model training
        # ds_local: Chilean local time (tz-naive) for twilight event grouping
        if "timestamp" in df.columns:
            ts_utc = pd.to_datetime(df["timestamp"], utc=True)
        elif "ds" in df.columns:
            ts_utc = pd.to_datetime(df["ds"], utc=True)
        else:
            raise ValueError("Data must have 'timestamp' or 'ds' column")

        df["ds"] = ts_utc.dt.tz_localize(None)  # UTC, tz-naive
        df["ds_local"] = ts_utc.dt.tz_convert("America/Santiago").dt.tz_localize(None)  # Local, tz-naive

        if "mean" in df.columns:
            df["y"] = df["mean"]
        elif "y" not in df.columns:
            raise ValueError("Data must have 'mean' or 'y' column")

        # Check for pre-computed alt_sun
        if "alt_sun" in df.columns:
            print("  Using pre-computed alt_sun")
        else:
            print("  Computing sun altitude...")
            timestamps = pd.DatetimeIndex(df["ds"])
            df["alt_sun"] = get_sun_altitude(timestamps)

        # Sort by time
        df = df.sort_values("ds").reset_index(drop=True)

        # df["ds"] is now tz-naive in Chilean local time
        # Set cutoff date (last timestamp in data)
        self.cutoff_date = df["ds"].max()
        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")

        self.df = df
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features required for NBEATSx and Ridge training.

        Uses shared FeatureBuilder which computes sunrise/twilight masks
        from alt_sun column (computing alt_sun if not present).
        """
        return self.feature_builder.build(df)

    def _make_nowcasts_df(self) -> pd.DataFrame:
        """Extract nowcasts from insample predictions.

        Uses extract_nowcasts() from ridge_features module.
        """
        if self._insample_predictions is None:
            raise RuntimeError("Insample predictions not generated. Call _generate_insample_predictions() first.")

        self.nowcasts_df = extract_nowcasts(self._insample_predictions)
        print(f"\n[Nowcasts] Extracted {len(self.nowcasts_df)} nowcast rows from {len(self._insample_predictions)} predictions")
        return self.nowcasts_df

    def _make_ridge_df(self) -> pd.DataFrame:
        """Build Ridge features on nowcasts_df.

        Uses build_ridge_df() from ridge_features module.
        """
        if self.nowcasts_df is None:
            raise RuntimeError("Nowcasts not extracted. Call _make_nowcasts_df() first.")

        if self.df_features is None:
            self.df_features = self._add_features(self.df)

        self.ridge_df = build_ridge_df(self.nowcasts_df, self.df_features, mode="twilight")
        print(f"[Ridge DF] Built {len(self.ridge_df)} rows with Ridge features")
        return self.ridge_df

    def _make_traj_df(self) -> pd.DataFrame:
        """Build trajectory predictions DataFrame.

        Uses build_traj_df() from ridge_features module.
        """
        if self._insample_predictions is None:
            raise RuntimeError("Insample predictions not generated. Call _generate_insample_predictions() first.")

        self.traj_df = build_traj_df(self._insample_predictions)
        print(f"\n[Traj DF] Using {len(self.traj_df)} predictions for trajectory training")
        return self.traj_df

    def _save_predictions(self, predictions: pd.DataFrame, cache_path: Path) -> None:
        """Save predictions to parquet cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(cache_path)
        print(f"  Saved predictions to {cache_path}")

    def _load_predictions(self, cache_path: Path) -> pd.DataFrame:
        """Load predictions from parquet cache."""
        if not cache_path.exists():
            return None
        predictions = pd.read_parquet(cache_path)
        print(f"  Loaded {len(predictions)} predictions from {cache_path.name}")
        return predictions

    def compute_rmse_by_lead(
        self,
        ridge_df: pd.DataFrame = None,
        lead_hours: list = None,
        label: str = "",
    ) -> dict:
        """Compute Ridge-Tw RMSE at each lead time.

        Uses standalone compute_ridge_tw_rmse() function for DRY.

        Args:
            ridge_df: DataFrame with Ridge features (default: self.ridge_df)
            lead_hours: List of lead times to evaluate (default: config lead_hours)
            label: Label for printing (e.g., "Insample", "Test")

        Returns:
            Dict of {lead: {"rmse": float, "bias": float, "n": int}}
        """
        if ridge_df is None:
            ridge_df = self.ridge_df
        if ridge_df is None:
            raise RuntimeError("No ridge_df available. Build it first.")

        if lead_hours is None:
            lead_hours = self.config["ridge_twilight"]["lead_hours"]

        if not self._tw_ridge_models:
            raise RuntimeError("No Ridge-Tw models trained. Call train_ridge_twilight() first.")

        # Get feature columns
        ridge_fb = RidgeFeatureBuilder(mode="twilight")
        feature_cols = ridge_fb.get_feature_cols()

        # Use standalone function
        results = compute_ridge_tw_rmse(
            ridge_df=ridge_df,
            models=self._tw_ridge_models,
            scalers=self._tw_ridge_scalers,
            lead_hours=lead_hours,
            feature_cols=feature_cols,
        )

        # Print summary if label provided
        if label:
            print(f"\n  {label} RMSE by Lead Time:")
            print(f"  {'Lead':<8} {'RMSE':<10} {'Bias':<10} {'N':<8}")
            print(f"  {'-'*36}")
            for lead in sorted(results.keys()):
                r = results[lead]
                print(f"  {lead:<8.1f} {r['rmse']:<10.3f} {r['bias']:<10.3f} {r['n']:<8}")

        return results

    def train_nbeats(self, use_cache: bool = True):
        """Train NBEATSx model using config settings.

        Args:
            use_cache: If True, load from cache if exists

        Returns:
            Trained NeuralForecast model
        """
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import HuberLoss
        from neuralforecast.models import NBEATSx

        cfg = self.config["nbeats"]
        data_cfg = self.config["data"]

        # Build cache path - use test_start_date if specified
        if data_cfg.get("test_start_date"):
            cutoff_str = pd.Timestamp(data_cfg["test_start_date"]).strftime("%Y%m%d")
        elif self.cutoff_date:
            cutoff_str = self.cutoff_date.strftime("%Y%m%d")
        else:
            cutoff_str = "default"
        cache_dir = self.root_path / cfg["cache_dir"]
        cache_path = cache_dir / f"{cfg['cache_prefix']}_{cutoff_str}"

        if use_cache and cache_path.exists():
            print(f"\n[NBEATSx] Loading from cache: {cache_path.name}")
            self._nbeats_model = NeuralForecast.load(str(cache_path))
            self._nbeats_model.models[0].inference_windows_batch_size = 64
            return self._nbeats_model

        print(f"\n[NBEATSx] Training new model (cutoff={cutoff_str})...")

        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Add features
        df = self._add_features(self.df)

        # Filter to training data (before test start, if specified)
        # Use ds_local for date-based filtering since dates are in local time
        if data_cfg.get("test_start_date"):
            test_start = pd.Timestamp(data_cfg["test_start_date"])
            train_df = df[df["ds_local"] < test_start].copy()
            print(f"  test_start_date cutoff: {test_start} (local time)")
            print(f"  Training data: {train_df['ds_local'].min()} to {train_df['ds_local'].max()}")
        else:
            train_df = df.copy()
        print(f"  Training samples: {len(train_df)}")

        # Prepare data for NeuralForecast
        hist_exog = cfg["hist_exog"]
        futr_exog = cfg["futr_exog"]
        nf_train = train_df[["ds", "y", "delta_T"] + hist_exog + futr_exog].copy()
        nf_train = nf_train.dropna(subset=["delta_T"] + hist_exog)
        nf_train["unique_id"] = "temp"
        nf_train["y"] = nf_train["delta_T"]

        print(f"  Training rows after dropna: {len(nf_train)}")

        # Create model with config settings
        model = NBEATSx(
            h=cfg["horizon"],
            input_size=cfg["input_size"],
            max_steps=cfg["max_steps"],
            loss=HuberLoss(),
            activation=cfg["activation"],
            scaler_type=cfg["scaler_type"],
            learning_rate=cfg["learning_rate"],
            stack_types=cfg["stack_types"],
            n_blocks=cfg["n_blocks"],
            mlp_units=cfg["mlp_units"],
            hist_exog_list=hist_exog,
            futr_exog_list=futr_exog,
        )

        # Train
        nf = NeuralForecast(models=[model], freq=data_cfg["freq"])
        nf.fit(nf_train)

        # Save
        cache_dir.mkdir(parents=True, exist_ok=True)
        nf.save(str(cache_path), overwrite=True)
        print(f"  Saved to {cache_path.name}")

        self._nbeats_model = nf
        return nf

    def _generate_insample_predictions(self, pred_times_hours: list = None, use_cache: bool = True) -> pd.DataFrame:
        """Generate in-sample NBEATSx predictions for Ridge training.

        Uses predict_batch from summit.nbeats for efficient batched inference.
        Generates predictions at regular intervals throughout each day.

        Args:
            pred_times_hours: List of hours from last twilight to generate predictions
                              (default: 1 to 21 hours in 1h increments)
            use_cache: If True, load from cache if exists

        Returns:
            DataFrame with columns needed by RidgeFeatureBuilder
        """
        # Check cache first
        if use_cache:
            cached = self._load_predictions(self._insample_cache_path)
            if cached is not None:
                self._insample_predictions = cached
                return cached

        from .nbeats import predict_batch

        if pred_times_hours is None:
            pred_times_hours = list(range(1, 22))

        if self._nbeats_model is None:
            raise RuntimeError("NBEATSx model not trained. Call train_nbeats() first.")

        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        print("\n[In-sample Predictions] Generating NBEATSx predictions...")
        print(f"  Prediction times: {len(pred_times_hours)} (from {min(pred_times_hours)}h to {max(pred_times_hours)}h from last tw)")

        cfg = self.config["nbeats"]
        data_cfg = self.config["data"]
        INPUT_SIZE = cfg["input_size"]

        # Use FeatureBuilder to compute delta_T and delta_T_approx
        df = self._add_features(self.df)

        # Filter to training data (if test_start_date specified)
        # Use ds_local for date-based filtering since dates are in local time
        if data_cfg.get("test_start_date"):
            test_start = pd.Timestamp(data_cfg["test_start_date"])
            train_df = df[df["ds_local"] < test_start].copy()
            print(f"  test_start_date cutoff: {test_start} (local time)")
            print(f"  Training data: {train_df['ds_local'].min()} to {train_df['ds_local'].max()}")
        else:
            train_df = df.copy()

        # Get twilight events (one per day, using local time for date grouping)
        twilight_events = train_df[train_df["twilight_temp"].notna()].copy()
        twilight_events["tw_date"] = twilight_events["ds_local"].dt.date
        twilight_events = twilight_events.groupby("tw_date").first().reset_index()

        # Add PREVIOUS twilight info
        twilight_events["T_tw_prev"] = twilight_events["twilight_temp"].shift(1)
        twilight_events["prev_tw_time"] = twilight_events["ds"].shift(1)

        print(f"  Found {len(twilight_events)} twilight events")

        # Build prediction requests
        prediction_requests = []
        request_metadata = []
        skipped = 0

        for _, tw_row in twilight_events.iterrows():
            tw_time = tw_row["ds"]  # UTC
            tw_time_local = tw_row["ds_local"]  # Local time
            tw_ts = pd.Timestamp(tw_time)
            if tw_ts.tz is not None:
                tw_ts = tw_ts.tz_localize(None)

            T_tw_prev = tw_row.get("T_tw_prev", np.nan)
            prev_tw_time = tw_row.get("prev_tw_time")
            tw_temp = tw_row["twilight_temp"]

            if pd.isna(T_tw_prev) or pd.isna(prev_tw_time):
                skipped += 1
                continue

            prev_tw_ts = pd.Timestamp(prev_tw_time)
            if prev_tw_ts.tz is not None:
                prev_tw_ts = prev_tw_ts.tz_localize(None)

            for h_from_last_tw in pred_times_hours:
                forecast_time = prev_tw_ts + pd.Timedelta(hours=h_from_last_tw)

                prediction_requests.append({
                    "forecast_time": forecast_time,
                    "tw_time": tw_ts,
                    "last_tw_time": prev_tw_ts,
                })
                request_metadata.append({
                    "tw_time": tw_ts,
                    "tw_time_local": tw_time_local,
                    "tw_temp": tw_temp,
                    "T_tw_last": T_tw_prev,
                    "prev_tw_time": prev_tw_ts,
                    "h_from_last_tw": h_from_last_tw,
                    "tw_row": tw_row,
                })

        print(f"  Requests: {len(prediction_requests)} (skipped {skipped} early twilights)")

        if len(prediction_requests) == 0:
            self._insample_predictions = pd.DataFrame()
            return self._insample_predictions

        # Call predict_batch once for all requests
        print("  Running batched NBEATSx inference...")
        results = predict_batch(self._nbeats_model, train_df, prediction_requests)
        print(f"  Got {len(results)} prediction results")

        # Build lookup for actual temperatures (vectorized)
        print("  Processing results (vectorized)...")
        df_sorted = train_df.sort_values("ds").reset_index(drop=True)
        t0 = df_sorted["ds"].iloc[0]
        y_values = df_sorted["y"].values

        # Concatenate all predictions into single DataFrame
        all_preds = []
        for req_idx, result in results.items():
            meta = request_metadata[req_idx]
            pred_df = result["predictions"].copy()
            pred_df["req_idx"] = req_idx
            pred_df["tw_time"] = meta["tw_time"]
            pred_df["tw_temp"] = meta["tw_temp"]
            pred_df["T_tw_last"] = meta["T_tw_last"]
            pred_df["prev_tw_time"] = meta["prev_tw_time"]
            pred_df["forecast_time"] = result["forecast_time"]
            pred_df["rate_sunrise_to_midday"] = meta["tw_row"].get("rate_sunrise_to_midday", np.nan)
            pred_df["rate_midday_to_twilight"] = meta["tw_row"].get("rate_midday_to_twilight", np.nan)
            pred_df["rate_twilight_to_midnight"] = meta["tw_row"].get("rate_twilight_to_midnight", np.nan)
            pred_df["rate_midnight_to_sunrise"] = meta["tw_row"].get("rate_midnight_to_sunrise", np.nan)
            pred_df["temp_since_sunrise"] = meta["tw_row"].get("temp_since_sunrise", np.nan)
            pred_df["temp_trend_3d"] = meta["tw_row"].get("temp_trend_3d", np.nan)
            all_preds.append(pred_df)

        pred_df = pd.concat(all_preds, ignore_index=True)
        pred_df = pred_df.rename(columns={"ds": "target_time"})

        # Vectorized time calculations
        pred_df["target_time"] = pd.to_datetime(pred_df["target_time"])
        pred_df["forecast_time"] = pd.to_datetime(pred_df["forecast_time"])
        pred_df["prev_tw_time"] = pd.to_datetime(pred_df["prev_tw_time"])

        # Compute target indices for temperature lookup
        target_idx = ((pred_df["target_time"] - t0).dt.total_seconds() // FREQ_SECONDS).astype(int)

        # Filter valid indices
        valid_mask = (target_idx >= 0) & (target_idx < len(y_values))
        pred_df = pred_df[valid_mask].copy()
        target_idx = target_idx[valid_mask]

        # Lookup actual temperatures
        pred_df["temp_actual"] = y_values[target_idx.values]

        # Filter out NaN temperatures
        pred_df = pred_df[~pred_df["temp_actual"].isna()].copy()

        # Compute derived columns (vectorized)
        pred_df["h_from_last_tw"] = (pred_df["target_time"] - pred_df["prev_tw_time"]).dt.total_seconds() / 3600
        pred_df["h_to_tw"] = (pred_df["tw_time"] - pred_df["target_time"]).dt.total_seconds() / 3600
        pred_df["lead_hours"] = (pred_df["target_time"] - pred_df["forecast_time"]).dt.total_seconds() / 3600

        # Apply exponential boundary gate to delta_T_pred
        # Gate = 1 - exp(-|h_to_tw| / tau), tau = 1 hour
        # This ensures delta_T -> 0 near twilight boundaries
        TAU = 1.0  # decay time constant in hours
        gate = 1.0 - np.exp(-np.abs(pred_df["h_to_tw"].values) / TAU)
        pred_df["delta_T_gated"] = pred_df["delta_T_pred"] * gate

        # Use gated delta_T for temperature approximation
        pred_df["temp_approx"] = pred_df["delta_T_gated"] + pred_df["T_tw_last"]
        pred_df["res"] = pred_df["temp_actual"] - pred_df["temp_approx"]

        # Clean up columns
        pred_df = pred_df.drop(columns=["req_idx"], errors="ignore")
        print(f"  Generated {len(pred_df)} predictions")

        # Save to cache
        self._save_predictions(pred_df, self._insample_cache_path)

        self._insample_predictions = pred_df
        return pred_df

    def _compute_insample_metrics(self, lead_times: list = None) -> dict:
        """Compute in-sample metrics for each lead time.

        Computes RMSE for three models:
        1. NBEATSx-Oracle: T_pred = delta_T_pred + Twilight_Trend(h)
           where Twilight_Trend(h) = T_tw_last + (T_tw - T_tw_last) * h/24
        2. NBEATSx-Ridge: T_pred = T_tw_last + slope_pred (twilight only)
        3. NBEATSx-Ridge-Traj: T_pred = temp_approx + slope_pred * h/24

        Args:
            lead_times: List of lead times in hours (default: [3, 6, 12])

        Returns:
            Dict with metrics for each model at each lead time
        """
        if lead_times is None:
            lead_times = [3, 6, 12]

        if not hasattr(self, "_insample_predictions") or self._insample_predictions is None:
            return {}

        # Ensure ridge_df (nowcasts with features) is built
        if self.ridge_df is None:
            if self.nowcasts_df is None:
                self._make_nowcasts_df()
            self._make_ridge_df()

        pred_df = self._insample_predictions.copy()
        ridge_df = self.ridge_df.copy()

        if len(pred_df) == 0:
            return {}

        print(f"\n[In-Sample Metrics] Computing from {len(ridge_df)} nowcasts...")

        # Get feature columns for Ridge-Tw
        ridge_fb = RidgeFeatureBuilder(mode="twilight")
        feature_cols = ridge_fb.get_feature_cols()
        available_features = [c for c in feature_cols if c in ridge_df.columns]

        # Ensure tw_slope is computed
        if "tw_slope" not in ridge_df.columns:
            ridge_df["tw_slope"] = ridge_df["tw_temp"] - ridge_df["T_tw_last"]

        metrics = {"oracle": {}, "ridge": {}, "ridge_traj": {}}

        # Debug: show data ranges
        print(f"  h_to_tw range: {ridge_df['h_to_tw'].min():.2f} to {ridge_df['h_to_tw'].max():.2f}")
        print(f"  h_from_last_tw range: {ridge_df['h_from_last_tw'].min():.2f} to {ridge_df['h_from_last_tw'].max():.2f}")
        print(f"  tw_slope range: {ridge_df['tw_slope'].min():.2f} to {ridge_df['tw_slope'].max():.2f}")

        # Compute metrics for each lead time
        for lead in lead_times:
            lead_key = f"{lead}h"

            # Filter by Ridge decision lead time: h_to_tw >= lead
            # This means target_time <= tw_time - lead (no future data)
            lead_df = ridge_df[ridge_df["h_to_tw"] >= lead].copy()

            if len(lead_df) == 0:
                print(f"  @ {lead}h: No data found")
                continue

            # Filter to targets within twilight cycle (h_from_last_tw <= 24)
            lead_df = lead_df[lead_df["h_from_last_tw"] <= 24].copy()

            # Debug: show h_from_last_tw distribution at this lead time
            print(f"\n  @ {lead}h lead: {len(lead_df)} samples")
            print(f"    h_from_last_tw: {lead_df['h_from_last_tw'].min():.1f} to {lead_df['h_from_last_tw'].max():.1f}")
            print(f"    h_to_tw: {lead_df['h_to_tw'].min():.1f} to {lead_df['h_to_tw'].max():.1f}")

            # =====================================================================
            # NBEATSx-Oracle: uses true Twilight_Trend (future info - cheating)
            # T_pred_oracle = delta_T_gated + Twilight_Trend(h)
            # where Twilight_Trend(h) = T_tw_last + tw_slope * h/24
            # Note: delta_T_gated has boundary gate applied (delta_T -> 0 near twilight)
            # =====================================================================
            oracle_valid = lead_df[["temp_actual", "delta_T_gated", "T_tw_last", "tw_slope", "h_from_last_tw"]].dropna()
            if len(oracle_valid) > 0:
                h = oracle_valid["h_from_last_tw"].values
                twilight_trend = oracle_valid["T_tw_last"].values + oracle_valid["tw_slope"].values * h / 24.0
                T_pred_oracle = oracle_valid["delta_T_gated"].values + twilight_trend
                errors_oracle = oracle_valid["temp_actual"].values - T_pred_oracle
                rmse_oracle = float(np.sqrt(np.mean(errors_oracle ** 2)))
                bias_oracle = float(np.mean(errors_oracle))
                metrics["oracle"][lead_key] = {"rmse": rmse_oracle, "bias": bias_oracle, "n": len(errors_oracle)}

            # =====================================================================
            # NBEATSx-Ridge-Traj: direct residual correction (no slope)
            # T_pred_ridge = temp_approx + res_pred
            # =====================================================================
            # Use trajectory model for each lead time (skip if features not in ridge_df)
            lead_float = float(lead)
            if lead_float in self._traj_ridge_models and lead_float in self._traj_ridge_features:
                ridge_model = self._traj_ridge_models[lead_float]
                ridge_scaler = self._traj_ridge_scalers[lead_float]
                traj_features = self._traj_ridge_features[lead_float]

                # Check if features exist in ridge_df
                missing_features = [f for f in traj_features if f not in lead_df.columns]
                if missing_features:
                    # Skip Ridge-Traj if features don't exist (different feature naming)
                    pass
                else:
                    valid_mask = lead_df[traj_features].notna().all(axis=1)
                    valid_df = lead_df[valid_mask].copy()

                    if len(valid_df) > 0:
                        X = valid_df[traj_features].values
                        X_scaled = ridge_scaler.transform(X)
                        res_pred = ridge_model.predict(X_scaled)

                        # Direct correction: T_pred = temp_approx + res_pred
                        T_pred_ridge = valid_df["temp_approx"].values + res_pred
                        errors_ridge = valid_df["temp_actual"].values - T_pred_ridge
                        rmse_ridge = float(np.sqrt(np.mean(errors_ridge ** 2)))
                        bias_ridge = float(np.mean(errors_ridge))
                        metrics["ridge_traj"][lead_key] = {"rmse": rmse_ridge, "bias": bias_ridge, "n": len(errors_ridge)}

            # =====================================================================
            # NBEATSx-Ridge (twilight prediction) - uses tw_slope models
            # T_tw_pred = T_tw_last + slope_pred
            # Select ONE row per twilight: most recent target_time
            # =====================================================================
            if lead_float in self._tw_ridge_models:
                # Select one row per twilight (most recent target_time)
                idx_recent = lead_df.groupby("tw_time")["target_time"].idxmax()
                tw_df = lead_df.loc[idx_recent].copy()

                if len(tw_df) > 0:
                    valid_mask = tw_df[available_features].notna().all(axis=1)
                    valid_df = tw_df[valid_mask].copy()

                    if len(valid_df) > 0:
                        ridge_model = self._tw_ridge_models[lead_float]
                        ridge_scaler = self._tw_ridge_scalers[lead_float]

                        X = valid_df[available_features].values
                        X_scaled = ridge_scaler.transform(X)
                        slope_pred = ridge_model.predict(X_scaled)

                        T_tw_pred = valid_df["T_tw_last"].values + slope_pred
                        errors_tw = valid_df["tw_temp"].values - T_tw_pred
                        rmse_tw = float(np.sqrt(np.mean(errors_tw ** 2)))
                        bias_tw = float(np.mean(errors_tw))
                        metrics["ridge"][lead_key] = {"rmse": rmse_tw, "bias": bias_tw, "n": len(errors_tw)}

        # Print summary
        print("\n  Model Performance Summary:")
        print(f"  {'Lead':<6} {'Oracle':<12} {'Ridge-Traj':<12} {'Ridge-TW':<12} {'N':<8}")
        print(f"  {'-'*50}")
        for lead in lead_times:
            lead_key = f"{lead}h"
            oracle_rmse = metrics["oracle"].get(lead_key, {}).get("rmse", np.nan)
            ridge_traj_rmse = metrics["ridge_traj"].get(lead_key, {}).get("rmse", np.nan)
            ridge_tw_rmse = metrics["ridge"].get(lead_key, {}).get("rmse", np.nan)
            n = metrics["oracle"].get(lead_key, {}).get("n", 0)
            print(f"  {lead_key:<6} {oracle_rmse:<12.3f} {ridge_traj_rmse:<12.3f} {ridge_tw_rmse:<12.3f} {n:<8}")

        self._insample_metrics = metrics
        return metrics

    def train_ridge_twilight(self, use_cache: bool = True):
        """Train Ridge twilight correction for multiple lead times.

        Uses nowcasts only (target_time == forecast_time) for training.
        Trains separate model for each lead time (0.5h to 12h).

        Args:
            use_cache: If True, load from cache if exists

        Returns:
            Dict of {lead: (model, scaler)} pairs
        """
        cfg = self.config["ridge_twilight"]
        lead_hours = cfg["lead_hours"]
        if not isinstance(lead_hours, list):
            lead_hours = [lead_hours]

        cache_dir = self.root_path / cfg["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Ridge Twilight] Training for {len(lead_hours)} lead times...")

        if self._nbeats_model is None:
            raise RuntimeError("NBEATSx model not trained. Call train_nbeats() first.")

        # Generate in-sample predictions if not available
        if not hasattr(self, "_insample_predictions") or self._insample_predictions is None:
            self._generate_insample_predictions()

        # Build nowcasts_df and ridge_df using helper methods
        if self.nowcasts_df is None:
            self._make_nowcasts_df()
        if self.ridge_df is None:
            self._make_ridge_df()

        # Get feature columns from RidgeFeatureBuilder
        ridge_fb = RidgeFeatureBuilder(mode="twilight")
        feature_cols = ridge_fb.get_feature_cols()
        target_col = "tw_slope"

        # Filter to features that exist in ridge_df
        available_features = [c for c in feature_cols if c in self.ridge_df.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"  Warning: Some features missing: {list(missing)[:5]}")

        loaded_count = 0
        trained_count = 0

        for lead in lead_hours:
            cache_path = cache_dir / f"ridge_tw_{lead}h.joblib"

            if use_cache and cache_path.exists():
                data = joblib.load(cache_path)
                if isinstance(data, tuple) and len(data) == 2:
                    self._tw_ridge_models[lead], self._tw_ridge_scalers[lead] = data
                    loaded_count += 1
                    continue

            # Filter to ALL nowcasts where h_to_tw >= lead
            # This includes nighttime (h_to_tw ~23) through decision point (h_to_tw = lead)
            # Each nowcast is a training sample with target = tw_slope
            df_lead = self.ridge_df[self.ridge_df["h_to_tw"] >= lead].copy()

            if len(df_lead) == 0:
                continue

            # Keep ALL nowcasts (not just one per twilight)
            # Drop rows with NaN in features or target
            df_lead = df_lead.dropna(subset=available_features + [target_col])

            if len(df_lead) < 10:
                print(f"  Skipping {lead}h: only {len(df_lead)} samples")
                continue

            # Debug: show sample counts for key lead times
            if lead in [3.0, 6.0, 12.0]:
                n_twilights = df_lead["tw_time"].nunique()
                samples_per_tw = len(df_lead) / n_twilights if n_twilights > 0 else 0
                h_to_tw_range = f"{df_lead['h_to_tw'].min():.1f} to {df_lead['h_to_tw'].max():.1f}"
                print(f"  Training {lead}h: {len(df_lead)} samples, {n_twilights} twilights, ~{samples_per_tw:.1f} samples/tw, h_to_tw: {h_to_tw_range}")

            # Train
            X = df_lead[available_features].values
            y = df_lead[target_col].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=cfg["alpha"])
            model.fit(X_scaled, y)

            # Save
            joblib.dump((model, scaler), cache_path)
            self._tw_ridge_models[lead] = model
            self._tw_ridge_scalers[lead] = scaler
            trained_count += 1

        print(f"  Loaded from cache: {loaded_count}")
        print(f"  Newly trained: {trained_count}")

        # For backward compatibility, set _ridge_model to 3h model if exists
        if 3.0 in self._tw_ridge_models:
            self._ridge_model = self._tw_ridge_models[3.0]
            self._ridge_scaler = self._tw_ridge_scalers[3.0]

        return self._tw_ridge_models

    def train_ridge_trajectory(self, use_cache: bool = True):
        """Train Ridge trajectory correction for multiple lead times.

        Trains on forecasts with specific lead times (not nowcasts).
        Uses simple features: rates, seasonal encoding, and optionally tw_slope.
        No res_htw features needed - rate features capture error patterns better.

        Args:
            use_cache: If True, load from cache if exists

        Returns:
            Dict of {lead: (model, scaler)} pairs
        """
        cfg = self.config["ridge_trajectory"]

        lead_hours = cfg["lead_hours"]
        cache_dir = self.root_path / cfg["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Ridge Trajectory] Training for {len(lead_hours)} lead times...")

        if self._nbeats_model is None:
            raise RuntimeError("NBEATSx model not trained. Call train_nbeats() first.")

        # Generate predictions if not available
        if not hasattr(self, "_insample_predictions") or self._insample_predictions is None:
            self._generate_insample_predictions()

        # Prepare prediction DataFrame with features
        pred_df = self._insample_predictions.copy()

        # Add seasonal features
        pred_df["doy"] = pd.to_datetime(pred_df["forecast_time"]).dt.dayofyear
        pred_df["doy_sin"] = np.sin(2 * np.pi * pred_df["doy"] / 365)
        pred_df["doy_cos"] = np.cos(2 * np.pi * pred_df["doy"] / 365)
        pred_df["hour_of_day"] = pd.to_datetime(pred_df["forecast_time"]).dt.hour
        pred_df["hour_sin"] = np.sin(2 * np.pi * pred_df["hour_of_day"] / 24)
        pred_df["hour_cos"] = np.cos(2 * np.pi * pred_df["hour_of_day"] / 24)

        # Add tw_slope (can be used as feature with noise to simulate prediction)
        pred_df["tw_slope"] = pred_df["tw_temp"] - pred_df["T_tw_last"]

        # Add lead_hours column if not present
        if "lead_hours" not in pred_df.columns:
            pred_df["lead_hours"] = (
                pred_df["target_time"] - pred_df["forecast_time"]
            ).dt.total_seconds() / 3600
        pred_df["lead_hours_int"] = pred_df["lead_hours"].round().astype(int)

        # Feature columns (simple, no res_htw)
        feature_cols = cfg.get("features", [
            "delta_T_pred", "temp_approx",
            "rate_sunrise_to_midday", "rate_midday_to_twilight",
            "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
            "temp_since_sunrise", "temp_trend_3d",
            "doy_sin", "doy_cos", "hour_sin", "hour_cos",
        ])

        # Optionally add tw_slope with noise as feature
        if cfg.get("use_tw_slope", False):
            noise_std = cfg.get("tw_slope_noise", 0.3)
            np.random.seed(42)
            pred_df["tw_slope_noisy"] = pred_df["tw_slope"] + np.random.normal(0, noise_std, len(pred_df))
            feature_cols = feature_cols + ["tw_slope_noisy"]

        target_col = cfg.get("target", "res")

        # Filter to available features
        feature_cols = [c for c in feature_cols if c in pred_df.columns]

        loaded_count = 0
        trained_count = 0

        for lead in lead_hours:
            cache_path = cache_dir / cfg["cache_pattern"].format(lead=lead)

            if use_cache and cache_path.exists():
                data = joblib.load(cache_path)
                if isinstance(data, tuple) and len(data) == 3:
                    self._traj_ridge_models[lead], self._traj_ridge_scalers[lead], self._traj_ridge_features[lead] = data
                    loaded_count += 1
                    continue

            # Filter to this lead time
            lead_int = int(lead)
            df_lead = pred_df[pred_df["lead_hours_int"] == lead_int].copy()

            if len(df_lead) == 0:
                print(f"  Skipping {lead}h: no valid samples")
                continue

            # Drop rows with NaN
            df_lead = df_lead.dropna(subset=feature_cols + [target_col, "temp_actual", "temp_approx"])

            if len(df_lead) < 50:
                print(f"  Skipping {lead}h: only {len(df_lead)} samples")
                continue

            # Train/test split (80/20 by time)
            df_lead = df_lead.sort_values("forecast_time")
            n_train = int(len(df_lead) * 0.8)
            train_df = df_lead.iloc[:n_train]

            X = train_df[feature_cols].values
            y = train_df[target_col].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=cfg["alpha"])
            model.fit(X_scaled, y)

            # Compute train RMSE
            y_pred = model.predict(X_scaled)
            train_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

            # Compute NBEATSx baseline and Ridge improvement on test set
            test_df = df_lead.iloc[n_train:]
            X_test = scaler.transform(test_df[feature_cols].values)
            res_pred = model.predict(X_test)
            temp_pred = test_df["temp_approx"].values + res_pred
            ridge_rmse = float(np.sqrt(np.mean((test_df["temp_actual"].values - temp_pred) ** 2)))
            nbeats_rmse = float(np.sqrt(np.mean((test_df["temp_actual"] - test_df["temp_approx"]) ** 2)))
            improvement = (nbeats_rmse - ridge_rmse) / nbeats_rmse * 100

            print(f"  Lead {lead}h: n={len(train_df)}, NBEATSx={nbeats_rmse:.3f}, Ridge={ridge_rmse:.3f} ({improvement:+.1f}%)")

            # Save model, scaler, and feature columns
            joblib.dump((model, scaler, feature_cols), cache_path)
            self._traj_ridge_models[lead] = model
            self._traj_ridge_scalers[lead] = scaler
            self._traj_ridge_features[lead] = feature_cols
            trained_count += 1

        print(f"  Loaded from cache: {loaded_count}")
        print(f"  Newly trained: {trained_count}")

        return self._traj_ridge_models

    def _generate_test_predictions(self, use_cache: bool = True) -> pd.DataFrame:
        """Generate test predictions (data >= test_start_date).

        Args:
            use_cache: If True, load from cache if exists

        Returns:
            DataFrame with test predictions
        """
        # Check cache first
        if use_cache:
            cached = self._load_predictions(self._test_cache_path)
            if cached is not None:
                self.test_predictions = cached
                return cached

        if self._nbeats_model is None:
            raise RuntimeError("NBEATSx model not trained. Call train_nbeats() first.")
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        data_cfg = self.config["data"]
        if not data_cfg.get("test_start_date"):
            raise ValueError("test_start_date not set in config")

        test_start = pd.Timestamp(data_cfg["test_start_date"])
        print(f"\n[Test Predictions] Generating for data >= {test_start}")

        # Add features to full dataset
        df = self._add_features(self.df)

        # Get twilight events in test period
        test_df = df[df["ds_local"] >= test_start].copy()
        twilight_events = test_df[test_df["twilight_temp"].notna()].copy()
        twilight_events["tw_date"] = twilight_events["ds_local"].dt.date
        twilight_events = twilight_events.groupby("tw_date").first().reset_index()
        twilight_events["T_tw_prev"] = twilight_events["twilight_temp"].shift(1)
        twilight_events["prev_tw_time"] = twilight_events["ds"].shift(1)

        print(f"  Twilight events: {len(twilight_events)}")

        # Build prediction requests
        pred_times_hours = list(range(1, 22))
        prediction_requests = []
        request_metadata = []

        for _, tw_row in twilight_events.iterrows():
            tw_ts = pd.Timestamp(tw_row["ds"])
            if tw_ts.tz is not None:
                tw_ts = tw_ts.tz_localize(None)

            T_tw_prev = tw_row.get("T_tw_prev", np.nan)
            prev_tw_time = tw_row.get("prev_tw_time")
            if pd.isna(T_tw_prev) or pd.isna(prev_tw_time):
                continue

            prev_tw_ts = pd.Timestamp(prev_tw_time)
            if prev_tw_ts.tz is not None:
                prev_tw_ts = prev_tw_ts.tz_localize(None)

            for h in pred_times_hours:
                forecast_time = prev_tw_ts + pd.Timedelta(hours=h)
                prediction_requests.append({
                    "forecast_time": forecast_time,
                    "tw_time": tw_ts,
                    "last_tw_time": prev_tw_ts,
                })
                request_metadata.append({
                    "tw_time": tw_ts,
                    "tw_temp": tw_row["twilight_temp"],
                    "T_tw_last": T_tw_prev,
                    "prev_tw_time": prev_tw_ts,
                    "tw_row": tw_row,
                })

        print(f"  Prediction requests: {len(prediction_requests)}")

        # Run NBEATSx predictions
        print("  Running NBEATSx inference...")
        results = predict_batch(self._nbeats_model, df, prediction_requests)

        # Process results (same as _generate_insample_predictions)
        df_sorted = df.sort_values("ds").reset_index(drop=True)
        t0 = df_sorted["ds"].iloc[0]
        y_values = df_sorted["y"].values

        all_preds = []
        for req_idx, result in results.items():
            meta = request_metadata[req_idx]
            pred_row = result["predictions"].copy()
            pred_row["tw_time"] = meta["tw_time"]
            pred_row["tw_temp"] = meta["tw_temp"]
            pred_row["T_tw_last"] = meta["T_tw_last"]
            pred_row["prev_tw_time"] = meta["prev_tw_time"]
            pred_row["forecast_time"] = result["forecast_time"]
            for col in ["rate_sunrise_to_midday", "rate_midday_to_twilight",
                        "rate_twilight_to_midnight", "rate_midnight_to_sunrise",
                        "temp_since_sunrise", "temp_trend_3d"]:
                pred_row[col] = meta["tw_row"].get(col, np.nan)
            all_preds.append(pred_row)

        pred_df = pd.concat(all_preds, ignore_index=True)
        pred_df = pred_df.rename(columns={"ds": "target_time"})
        pred_df["target_time"] = pd.to_datetime(pred_df["target_time"])
        pred_df["forecast_time"] = pd.to_datetime(pred_df["forecast_time"])
        pred_df["prev_tw_time"] = pd.to_datetime(pred_df["prev_tw_time"])

        # Lookup actual temperatures
        target_idx = ((pred_df["target_time"] - t0).dt.total_seconds() // FREQ_SECONDS).astype(int)
        valid_mask = (target_idx >= 0) & (target_idx < len(y_values))
        pred_df = pred_df[valid_mask].copy()
        target_idx = target_idx[valid_mask]
        pred_df["temp_actual"] = y_values[target_idx.values]
        pred_df = pred_df[~pred_df["temp_actual"].isna()].copy()

        # Compute derived columns
        pred_df["h_from_last_tw"] = (pred_df["target_time"] - pred_df["prev_tw_time"]).dt.total_seconds() / 3600
        pred_df["h_to_tw"] = (pred_df["tw_time"] - pred_df["target_time"]).dt.total_seconds() / 3600

        # Apply boundary gate
        TAU = 1.0
        gate = 1.0 - np.exp(-np.abs(pred_df["h_to_tw"].values) / TAU)
        pred_df["delta_T_gated"] = pred_df["delta_T_pred"] * gate
        pred_df["temp_approx"] = pred_df["delta_T_gated"] + pred_df["T_tw_last"]
        pred_df["res"] = pred_df["temp_actual"] - pred_df["temp_approx"]

        print(f"  Generated {len(pred_df)} predictions")

        # Save to cache
        self._save_predictions(pred_df, self._test_cache_path)

        self.test_predictions = pred_df
        return pred_df

    def _make_test_ridge_df(self) -> pd.DataFrame:
        """Build Ridge features on test nowcasts.

        Returns:
            DataFrame with Ridge features for test data
        """
        if self.test_predictions is None:
            raise RuntimeError("Test predictions not generated. Call _generate_test_predictions() first.")

        if self.df_features is None:
            self.df_features = self._add_features(self.df)

        # Extract nowcasts
        self.test_nowcasts_df = extract_nowcasts(self.test_predictions)
        print(f"[Test Nowcasts] Extracted {len(self.test_nowcasts_df)} rows")

        # Build Ridge features
        self.test_ridge_df = build_ridge_df(self.test_nowcasts_df, self.df_features, mode="twilight")
        print(f"[Test Ridge DF] Built {len(self.test_ridge_df)} rows")

        return self.test_ridge_df

    def compute_test_rmse(self, lead_times: list = None, use_cache: bool = True) -> dict:
        """Compute RMSE on test data using Ridge-Tw models.

        Args:
            lead_times: List of lead times in hours (default: all config lead_hours)
            use_cache: If True, load predictions from cache

        Returns:
            Dict of {lead: {"rmse": float, "bias": float, "n": int}}
        """
        if lead_times is None:
            lead_times = self.config["ridge_twilight"]["lead_hours"]

        # Generate test predictions if needed
        if self.test_predictions is None:
            self._generate_test_predictions(use_cache=use_cache)

        # Build test ridge_df if needed
        if self.test_ridge_df is None:
            self._make_test_ridge_df()

        # Use the reusable compute_rmse_by_lead method
        results = self.compute_rmse_by_lead(
            ridge_df=self.test_ridge_df,
            lead_hours=lead_times,
            label="Test",
        )

        self._test_metrics = {"ridge": {f"{lead}h": results.get(lead, {}) for lead in lead_times}}
        return results

    # Backward compatibility aliases
    def train_ridge(self, use_cache: bool = True):
        """Alias for train_ridge_twilight()."""
        return self.train_ridge_twilight(use_cache=use_cache)

    def train_traj_ridge(self, use_cache: bool = True):
        """Alias for train_ridge_trajectory()."""
        return self.train_ridge_trajectory(use_cache=use_cache)

    def train_all(self, use_cache: bool = True):
        """Execute full training pipeline.

        Args:
            use_cache: If True, load from cache where available
        """
        print("=" * 60)
        print("NbeatsxTrainer: Full Training Pipeline")
        print("=" * 60)

        self.load_data()
        self.train_nbeats(use_cache=use_cache)
        self.train_ridge_twilight(use_cache=use_cache)
        self.train_ridge_trajectory(use_cache=use_cache)

        # Compute in-sample metrics at multiple lead times
        self._compute_insample_metrics(lead_times=[3, 6, 12])

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    def print_summary(self):
        """Print summary of trained models."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if self.df is not None:
            print("\nData:")
            print(f"  Source: {self.data_path}")
            print(f"  Rows: {len(self.df)}")
            print(f"  Cutoff: {self.cutoff_date}")

        cfg = self.config

        if self._nbeats_model is not None:
            nbeats_cfg = cfg["nbeats"]
            cutoff_str = (
                self.cutoff_date.strftime("%Y%m%d") if self.cutoff_date else "default"
            )
            print("\nNBEATSx:")
            print(
                f"  Cache: {nbeats_cfg['cache_dir']}/{nbeats_cfg['cache_prefix']}_{cutoff_str}"
            )
            print(f"  Horizon: {nbeats_cfg['horizon']} steps")
            print(f"  Stacks: {nbeats_cfg['stack_types']}")

        if self._ridge_model is not None:
            ridge_cfg = cfg["ridge_twilight"]
            print("\nRidge Twilight:")
            print(f"  Lead time: {ridge_cfg['lead_hours']}h")
            print(f"  Alpha: {ridge_cfg['alpha']}")
            print(f"  Features: {len(ridge_cfg['features'])}")

        if self._traj_ridge_models:
            traj_cfg = cfg["ridge_trajectory"]
            print("\nRidge Trajectory:")
            print(f"  Lead times: {len(self._traj_ridge_models)}")
            print(f"  Alpha: {traj_cfg['alpha']}")
            print(f"  Cache: {traj_cfg['cache_dir']}/")

        # In-sample performance metrics
        if hasattr(self, "_insample_metrics") and self._insample_metrics:
            print("\nIn-Sample Performance @ 3h Lead Time:")
            print(f"  {'Model':<20} {'RMSE':>8} {'N':>8}")
            print(f"  {'-'*36}")

            m = self._insample_metrics
            if "nbeats" in m and "3h" in m["nbeats"]:
                print(f"  {'NBEATSx':<20} {m['nbeats']['3h']['rmse']:>8.3f} {m['nbeats']['3h']['n']:>8}")
            if "ridge_tw" in m and "3h" in m["ridge_tw"]:
                print(f"  {'NBEATSx-Ridge':<20} {m['ridge_tw']['3h']['rmse']:>8.3f} {m['ridge_tw']['3h']['n']:>8}")
            if "ridge_traj" in m and "3h" in m["ridge_traj"]:
                print(f"  {'NBEATSx-Ridge-Traj':<20} {m['ridge_traj']['3h']['rmse']:>8.3f} {m['ridge_traj']['3h']['n']:>8}")

        # Test performance metrics
        if hasattr(self, "_test_metrics") and self._test_metrics:
            print("\nTest Performance @ 3h Lead Time:")
            print(f"  {'Model':<20} {'RMSE':>8} {'N':>8}")
            print(f"  {'-'*36}")

            m = self._test_metrics
            if "oracle" in m and "3h" in m["oracle"]:
                print(f"  {'NBEATSx-Oracle':<20} {m['oracle']['3h']['rmse']:>8.3f} {m['oracle']['3h']['n']:>8}")
            if "ridge_traj" in m and "3h" in m["ridge_traj"]:
                print(f"  {'NBEATSx-Ridge-Traj':<20} {m['ridge_traj']['3h']['rmse']:>8.3f} {m['ridge_traj']['3h']['n']:>8}")

        print("=" * 60)


def main():
    """Run trainer from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Train NBEATSx-Ridge models")
    parser.add_argument("data_path", help="Path to training data CSV")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument(
        "--no-cache", action="store_true", help="Force retrain (ignore cache)"
    )
    args = parser.parse_args()

    trainer = NbeatsxTrainer(args.data_path, config_path=args.config)
    trainer.train_all(use_cache=not args.no_cache)
    trainer.print_summary()


if __name__ == "__main__":
    main()
