"""Configuration for the causal NBEATSx-Diff twilight forecasting pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT_PATH = Path(__file__).parent.parent
_SDF_DATA = Path("/sdf/home/e/esteves/sitcom-analysis/rubin-twilight-forecast/data/temp_history_jan2026_with_weather.csv")
_LOCAL_DATA = ROOT_PATH / "data" / "temp_history_jan2026_with_weather.csv"
DATA_PATH = _SDF_DATA if _SDF_DATA.exists() else _LOCAL_DATA
RESULTS_PATH = ROOT_PATH / "results"
MODEL_CACHE_PATH = RESULTS_PATH / "nbeats_diff_cache"
FIGURES_PATH = ROOT_PATH / "figures"

# Train/test split
TEST_START_DATE = pd.Timestamp("2025-01-01")

# Solar grid
SOLAR_GRID_STEP = 1.0 / 48.0  # 48 points per solar day (30-min equivalent)
STEPS_PER_DAY = 48
HALFDAY_LAG_STEPS = 24        # 0.5 solar days = target differencing lag (long-range model)
SHORT_LAG_STEPS = 6           # 3h lag = 6 steps (short-range model)
SHORT_LEAD_CUTOFF = 1.5       # hours: use short-lag model below this lead time

# Synthetic clock: step * 24h = 0.5h = 1800s per grid step
SOLAR_GRID_FREQ = "1800s"

# NBEATSx model
NBEATS_INPUT_SIZE = 48        # 1 solar day lookback
NBEATS_HORIZON = 26           # covers up to 12h lead
NBEATS_MAX_STEPS = 700        # width=16 optimum (small net resists overfit; RMSE flat past ~700)

# Evaluation lead times (0.5h steps, max 12h due to 12h differencing anchor)
LEAD_TIMES_HOURS = list(np.arange(0.5, 12.5, 0.5))

# Ridge correction
RIDGE_ALPHA = 1.0
