"""Fit per-solar-time bias-correction for the MeteoBlue NWP forecast.

Writes results/meteoblue_correction_per_solartime.csv. The output supersedes
any earlier version of that file in the repo (which was produced from an
older interactive run on a slightly different summit-temperature dataset and
is not bit-reproducible; the current script reflects the calibration actually
used by forecast/run.py).

Algorithm (mirrors forecast/run.py:126-178 exactly):
  1. Load raw MeteoBlue forecasts; keep only causal samples (lead_hours > 0)
     and, for each valid_time, the shortest-lead causal forecast.
  2. Linearly interpolate the kept (valid_time, temperature) series onto the
     solar-time grid (np.interp on int64 timestamps); mark grid points
     outside the MeteoBlue temporal range as NaN.
  3. For each of N_BINS uniform SolarTime bins (default 48), fit a 1-D
     ordinary-least-squares regression T_truth ~ slope * T_meteoblue + bias
     on PRE-2025 (training) data only, requiring strictly more than 30
     samples in the bin.
  4. Report the residual RMSE per bin (T_truth - corrected) on the same
     training subset used for the fit.

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
MIN_BIN_SAMPLES = 30  # require strictly MORE than this; matches run.py's '> 30'


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

    # For each valid_time we may have several causal forecasts (different
    # issue_times, hence different lead_hours). Keep the one with the
    # SHORTEST lead, i.e. the most recent forecast that still respects
    # causality. After this we have exactly one row per valid_time.
    mb_latest = (
        mb.sort_values(["valid_time", "lead_hours"])  # ascending lead within each valid_time
          .drop_duplicates("valid_time", keep="first")  # keep min(lead_hours)
          [["valid_time", "temperature"]]
          .set_index("valid_time")
          .sort_index()
    )
    assert mb_latest.index.is_unique, "mb_latest must have one row per valid_time"

    # Interpolate MeteoBlue temperature onto the solar-time grid in time:
    # for each ds_real (int64 ns), linearly interpolate between adjacent
    # MeteoBlue valid_times. Grid points outside the MeteoBlue temporal
    # range are set to NaN so they are not used in the fits.
    mb_raw = np.interp(
        ds_real.astype("int64"),
        mb_latest.index.astype("int64"),
        mb_latest["temperature"].values,
    )
    out = (ds_real < mb_latest.index.min()) | (ds_real > mb_latest.index.max())
    mb_raw[out] = np.nan

    # Sanity: exactly one MB temperature value per solar-time grid step.
    assert mb_raw.shape == y_arr.shape, (
        f"mb_raw must have one value per grid step "
        f"(expected {y_arr.shape}, got {mb_raw.shape})"
    )

    # Fit per-bin slope + bias on TRAIN (pre-2025) only.
    #
    # NOTE: FeatureBuilder produces SolarTime values that are quantized to
    # roughly i/N_BINS but with floating-point noise. Half-open bin edges
    # `[lo, hi)` from np.linspace would silently drop ~1/3 of all samples
    # into the wrong bin (we observed bins 2/5/8/... ending up with N=0).
    # Compute bin assignment by rounding instead, which is FP-robust.
    train_mask = ds_real < TEST_START_DATE
    bin_idx_all = np.round(solar_time * N_BINS).astype(int) % N_BINS

    rows = []
    for b in range(N_BINS):
        in_bin = (bin_idx_all == b)
        usable = in_bin & ~np.isnan(mb_raw) & ~np.isnan(y_arr) & train_mask
        n = int(usable.sum())
        st_centre = b / N_BINS  # canonical SolarTime for this bin

        # run.py's behaviour: under-populated bins fall back to identity
        # correction (slope=1, bias=0) so that mb_corrected == mb_raw there.
        # We still emit a row so the CSV has all 48 bins.
        if n <= MIN_BIN_SAMPLES:
            rows.append({
                "bin": b,
                "solar_time": st_centre,
                "slope": 1.0,
                "bias": 0.0,
                "N": n,
                "rmse_corrected": np.nan,
            })
            continue

        slope, bias = np.polyfit(mb_raw[usable], y_arr[usable], 1)
        corrected = slope * mb_raw[usable] + bias
        rmse = float(np.sqrt(np.mean((y_arr[usable] - corrected) ** 2)))
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
