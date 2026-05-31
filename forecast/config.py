"""Configuration for the causal NBEATSx-Diff twilight forecasting pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = Path("/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast/data/temp_history_jan2026_with_weather.csv")
RESULTS_PATH = ROOT_PATH / "results"
MODEL_CACHE_PATH = RESULTS_PATH / "nbeats_diff_cache"
FIGURES_PATH = ROOT_PATH / "figures"

# Train/test split
TEST_START_DATE = pd.Timestamp("2025-01-01")

# Solar grid
SOLAR_GRID_STEP = 1.0 / 48.0  # 48 points per solar day (30-min equivalent)
STEPS_PER_DAY = 48
HALFDAY_LAG_STEPS = 24        # 0.5 solar days = target differencing lag

# Synthetic clock: step * 24h = 0.5h = 1800s per grid step
SOLAR_GRID_FREQ = "1800s"

# NBEATSx model
NBEATS_INPUT_SIZE = 48        # 1 solar day lookback
NBEATS_HORIZON = 24           # 0.5 solar day ahead (~12h)
NBEATS_MAX_STEPS = 500

# Evaluation (max 6h lead)
LEAD_TIMES_HOURS = list(np.arange(0.5, 6.5, 0.5))

# Ridge correction
RIDGE_ALPHA = 1.0
