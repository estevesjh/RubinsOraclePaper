"""Central configuration for twilight temperature forecasting pipeline.

All constants and hyperparameters in one place.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data" / "temp_history_all_dec2025_sunrise_sunset.csv"
RESULTS_PATH = ROOT_PATH / "results"

# Cache paths
SKLEARN_CACHE_PATH = (
    RESULTS_PATH / "paper_model_cache" / "sklearn_hourly_temps_with_trend"
)
NBEATS_CACHE_PATH = RESULTS_PATH / "paper_nbeats_cache"
TWILIGHT_OFFSET_CACHE = RESULTS_PATH / "twilight_offset_cache"

# Output files
PAPER_RESULTS_FILE = RESULTS_PATH / "paper_results.csv"
PAPER_RESULTS_V2_FILE = RESULTS_PATH / "paper_results_v2.csv"
TWILIGHT_OFFSET_FILE = RESULTS_PATH / "twilight_offset_predictions.csv"
METRICS_3H_FILE = RESULTS_PATH / "metrics_3h.csv"
FIGURES_PATH = ROOT_PATH / "figures"

# Iterative slope correction
# Set to None for flat baseline (iteration 0), or path to slope CSV for iterations 1+
SLOPE_FILE = (
    RESULTS_PATH / "slope_predictions_10am.csv"
)  # Iteration 1: use slope_v1 from 10am
# SLOPE_FILE = None  # iteration 0

# =============================================================================
# Time Series Configuration
# =============================================================================

FREQ = "15min"
SAMPLES_PER_HOUR = 4

# =============================================================================
# Train/Test Split
# =============================================================================

TEST_START_DATE = pd.Timestamp("2025-01-01")

# =============================================================================
# Lead Times for Evaluation
# =============================================================================

# Lead times for final evaluation (0.5h to 12h in 0.5h steps)
LEAD_TIMES_HOURS = list(np.arange(0.5, 12.5, 0.5))

# =============================================================================
# NBEATSx Prediction Schedule
# =============================================================================

# Predictions issued every 30 minutes (0.5h to 24h before twilight)
# Format: (label, hours_before_twilight)
PRED_TIMES = [(f"{h:.1f}h", h) for h in np.arange(0.5, 24.5, 0.5)]

# Maximum prediction horizon for Ridge correction
# Only use predictions where (target_time - forecast_time) <= this value
MAX_PREDICTION_HORIZON = 18.0  # hours

# =============================================================================
# NBEATSx Model Hyperparameters
# =============================================================================

NBEATS_HORIZON = 48  # 12 hours at 15-min resolution
NBEATS_INPUT_SIZE = 96  # 24 hours at 15-min resolution
NBEATS_MAX_STEPS = 500

# Architecture
NBEATS_CONFIG = {
    "activation": "SELU",
    "scaler_type": "robust",
    "stack_types": ["trend", "seasonality", "exogenous"],
    "mlp_units": 3 * [[32, 32]],
    "n_blocks": [1, 1, 1],
    "learning_rate": 0.01,
}

# Features (optimized v1 - Jan 2025)
# 4 rate pairs are highly predictive (51% importance in RF)
# trend_2h captures temperature dynamics effectively
NBEATS_HIST_EXOG = [
    "temp_raw",
    "temp_last_sunrise",
    "trend_2h",
    "temp_trend_3d",
    "rate_sunrise_to_midday",
    "rate_midday_to_twilight",
    "rate_twilight_to_midnight",
    "rate_midnight_to_sunrise",
]
NBEATS_FUTR_EXOG = ["twilight_cos"]

# =============================================================================
# Baseline Model Configuration
# =============================================================================

# Maximum hours of temperature history to use as features
MAX_HOURS_BEFORE = 12

# Random Forest
RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}

# MLP
MLP_CONFIG = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "max_iter": 500,
    "early_stopping": True,
    "random_state": 42,
}

# Ridge
RIDGE_ALPHA = 1.0

# =============================================================================
# Helper Functions
# =============================================================================


def get_pred_times_subset(max_hours: float = 24.0, step: float = 0.5):
    """Get prediction times up to max_hours with given step."""
    return [(f"{h:.1f}h", h) for h in np.arange(step, max_hours + step, step)]


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Test start: {TEST_START_DATE}")
    print(f"Lead times: {LEAD_TIMES_HOURS[0]}h to {LEAD_TIMES_HOURS[-1]}h")
    print(
        f"Prediction times: {len(PRED_TIMES)} ({PRED_TIMES[0][0]} to {PRED_TIMES[-1][0]})"
    )
    print(
        f"NBEATSx horizon: {NBEATS_HORIZON} steps ({NBEATS_HORIZON / SAMPLES_PER_HOUR}h)"
    )
    print(
        f"NBEATSx input: {NBEATS_INPUT_SIZE} steps ({NBEATS_INPUT_SIZE / SAMPLES_PER_HOUR}h)"
    )
    print("=" * 60)


if __name__ == "__main__":
    print_config()
