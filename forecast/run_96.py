"""Run NBEATSx-Diff with 1/96 solar grid (15-min equivalent).

Same pipeline as run.py but with:
- SOLAR_GRID_STEP = 1/96 (96 pts/day)
- HALFDAY_LAG_STEPS = 48 (0.5 solar day)
- NBEATS_INPUT_SIZE = 96 (1 solar day)
- NBEATS_HORIZON = 48 (0.5 solar day)
- SOLAR_GRID_FREQ = "900s" (15 min)

Usage:
    /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/run_96.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Override config before importing run
import config
config.SOLAR_GRID_STEP = 1.0 / 96.0
config.STEPS_PER_DAY = 96
config.HALFDAY_LAG_STEPS = 48
config.SHORT_LAG_STEPS = 12       # 3h = 12 steps at 96/day
config.SOLAR_GRID_FREQ = "900s"
config.NBEATS_INPUT_SIZE = 96     # 1 solar day
config.NBEATS_HORIZON = 48        # 12h

# Reimport run with patched config
from run import main
main()
