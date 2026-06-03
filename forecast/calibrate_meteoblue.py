"""Fit per-solar-time bias-correction for the MeteoBlue NWP forecast.

Writes results/meteoblue_correction_per_solartime.csv. The output supersedes
any earlier version of that file in the repo (which was produced from an
older interactive run on a slightly different summit-temperature dataset and
is not bit-reproducible; the current script reflects the calibration actually
used by forecast/run.py).

Algorithm (matches the inline calibration in forecast/run.py):
  1. Load raw MeteoBlue forecasts; keep only causal samples (lead_hours > 0)
     and the shortest available lead per valid_time.
  2. Interpolate raw MeteoBlue temperature onto the solar-time grid.
  3. For each of N_BINS uniform SolarTime bins (default 48), fit a 1-D linear
     regression T_truth ~ slope * T_meteoblue + bias on PRE-2025 data only.
  4. Apply the per-bin correction across the full grid; report the residual
     RMSE per bin.

The output CSV has one row per bin actually populated by training data:
    bin, solar_time, slope, bias, N, rmse_corrected

Usage:
    python forecast/calibrate_meteoblue.py
"""

import os
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "data", "rubin-twilight-forecast"))
sys.path.insert(0, os.path.dirname(__file__))

from config import RESULTS_PATH, TEST_START_DATE
from run import load_and_prepare


N_BINS = 48
MIN_BIN_SAMPLES = 30  # need at least this many train samples to fit a bin


def main():
    print("=" * 60)
    print("Calibrating MeteoBlue per-SolarTime bias correction")
    print("=" * 60)

    # Build the solar-time grid (same as in run.py)
    grid, _ = load_and_prepare()
    y_arr = grid["y"].values
    solar_time = grid["SolarTime"].values
    ds_real = pd.to_datetime(grid["ds_real"].values)

    # Load raw MeteoBlue forecasts
    mb_file = Path(_REPO_ROOT) / "data" / "meteo_blue_weather_station.csv"
    if not mb_file.exists():
        raise FileNotFoundError(f"MeteoBlue file not found: {mb_file}")
    mb = pd.read_csv(mb_file)
    valid_col = "valid_time_utc" if "valid_time_utc" in mb.columns else "valid_time"
    mb["valid_time"] = pd.to_datetime(mb[valid_col], utc=True).dt.tz_localize(None)
    mb = mb[mb["lead_hours"] > 0].copy()  # causal only
    mb_latest = (
        mb.sort_values(["valid_time", "lead_hours"])
          .drop_duplicates("valid_time", keep="first")
          [["valid_time", "temperature"]]
          .set_index("valid_time")
          .sort_index()
    )

    # Interpolate raw MeteoBlue onto the solar-time grid; mark out-of-range as NaN
    mb_raw = np.interp(
        ds_real.astype("int64"),
        mb_latest.index.astype("int64"),
        mb_latest["temperature"].values,
    )
    out = (ds_real < mb_latest.index.min()) | (ds_real > mb_latest.index.max())
    mb_raw[out] = np.nan

    # Fit per-bin slope + bias on TRAIN (pre-2025) only
    train_mask = ds_real < TEST_START_DATE
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)

    rows = []
    for b in range(N_BINS):
        in_bin = (solar_time >= bin_edges[b]) & (solar_time < bin_edges[b + 1])
        usable = in_bin & ~np.isnan(mb_raw) & ~np.isnan(y_arr) & train_mask
        n = int(usable.sum())
        if n <= MIN_BIN_SAMPLES:
            continue
        slope, bias = np.polyfit(mb_raw[usable], y_arr[usable], 1)
        # Residual RMSE on the same train-set bin after correction
        corrected = slope * mb_raw[usable] + bias
        rmse = float(np.sqrt(np.mean((y_arr[usable] - corrected) ** 2)))
        # Bin centre as a representative SolarTime
        st_centre = float(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        rows.append({
            "bin": b,
            "solar_time": st_centre,
            "slope": float(slope),
            "bias": float(bias),
            "N": n,
            "rmse_corrected": rmse,
        })

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_PATH / "meteoblue_correction_per_solartime.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} / {N_BINS} bins -> {out_path}")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
