"""
Fill gaps in temperature time series data.

Order of operations:
1. Fill tiny gaps (<3h) with linear interpolation
2. Fill small/medium gaps (3h-72h) with Prophet
3. Fill 8-day gap (Jan 2024) from Jan 2025 data + boundary matching

The 16-day gap (Feb 2025) is used as train/test split boundary and NOT filled.

Output: data/temp_history_all_dec2025_filled_gaps.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from prophet import Prophet
from tqdm import tqdm

# Suppress Prophet logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# =============================================================================
# Config
# =============================================================================

DATA_PATH = Path("data/temp_history_all_dec2025.csv")
OUTPUT_PATH = Path("data/temp_history_all_dec2025_filled_gaps.csv")
CUTOFF = pd.Timestamp("2023-10-01", tz="UTC")

# Known gaps
GAP_8DAY_START = pd.Timestamp("2024-01-17 06:00:00", tz="UTC")
GAP_8DAY_END = pd.Timestamp("2024-01-25 19:00:00", tz="UTC")
GAP_16DAY_START = pd.Timestamp(
    "2025-02-04 16:00:00+00:00", tz="UTC"
)  # Train/test boundary
GAP_16DAY_END = pd.Timestamp(
    "2025-02-20 22:30:00+00:00", tz="UTC"
)  # Train/test boundary

FREQ = 4  # 4 samples per hour

# =============================================================================
# Gap Identification
# =============================================================================


def identify_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Identify gaps (NaN regions) in the dataframe."""
    nan_mask = df["y"].isna()
    if not nan_mask.any():
        return pd.DataFrame()

    gap_groups = (nan_mask != nan_mask.shift()).cumsum()
    gap_info = (
        df[nan_mask]
        .groupby(gap_groups[nan_mask])
        .agg(
            start=("ds", "min"),
            end=("ds", "max"),
        )
    )
    gap_info["hours"] = (
        (gap_info["end"] - gap_info["start"]).dt.total_seconds() / 60.0 / 60
    )
    gap_info["days"] = gap_info["hours"] / 24
    return gap_info


# =============================================================================
# Prophet Gap Filling
# =============================================================================


def fill_gap_prophet(
    df: pd.DataFrame,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    context_days: int = 11,
) -> pd.DataFrame | None:
    """Fill gap using Prophet fit on window centered around gap."""
    context_start = gap_start - pd.Timedelta(days=context_days + 1)
    context_end = gap_start + pd.Timedelta(hours=72)

    context = df[((df["ds"] >= context_start) & (df["ds"] <= context_end))].copy()

    if len(context) < 48:
        return None

    context_naive = context.copy()
    context_naive["ds"] = context_naive["ds"].dt.tz_localize(None)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.85,
        n_changepoints=int(2 * context_days),
    )
    model.fit(context_naive[["ds", "y"]])

    gap_times = pd.date_range(gap_start, gap_end, freq="15min", tz="UTC")
    future = pd.DataFrame({"ds": gap_times.tz_localize(None)})
    forecast = model.predict(future)

    return pd.DataFrame({"ds": gap_times, "y": forecast["yhat"].values})


# =============================================================================
# 8-Day Gap Filling (from 2025 data)
# =============================================================================


def fill_gap_from_next_year(
    df: pd.DataFrame,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    context_days: int = 7,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Fill gap using data from same calendar dates next year.

    Strategy:
    1. Get source data from Jan 2025 (same calendar dates)
    2. Interpolate any NaNs in the source data
    3. Match boundaries to 2024 context using linear offset blending
    """
    print(f"  Gap: {gap_start} to {gap_end}")

    # Get source from 2025
    source_start_2025 = gap_start + pd.DateOffset(years=1)
    source_end_2025 = gap_end + pd.DateOffset(years=1)

    source = df[(df["ds"] >= source_start_2025) & (df["ds"] <= source_end_2025)].copy()

    if len(source) == 0:
        print("  ERROR: No source data found in 2025")
        return df, None

    print(f"  Source: {len(source)} hours from Jan 2025")

    # Handle NaNs in source
    n_source_nan = source["y"].isna().sum()
    if n_source_nan > 0:
        print(f"  Interpolating {n_source_nan} NaN values in source")
        source["y"] = source["y"].interpolate(method="linear", limit_direction="both")

    # Get context from 2024
    before = df[
        (df["ds"] >= gap_start - pd.Timedelta(days=context_days))
        & (df["ds"] < gap_start)
        & df["y"].notna()
    ]
    after = df[
        (df["ds"] > gap_end)
        & (df["ds"] <= gap_end + pd.Timedelta(days=context_days))
        & df["y"].notna()
    ]

    if len(before) < 24 or len(after) < 24:
        print(f"  ERROR: Not enough context (before={len(before)}, after={len(after)})")
        return df, None

    # Boundary values
    val_before_2024 = before["y"].iloc[-1]
    val_after_2024 = after["y"].iloc[0]
    source_start_val = source["y"].iloc[0]
    source_end_val = source["y"].iloc[-1]

    print(f"  2024 boundaries: start={val_before_2024:.2f}, end={val_after_2024:.2f}")
    print(f"  2025 source:     start={source_start_val:.2f}, end={source_end_val:.2f}")

    # Calculate offsets
    start_offset = val_before_2024 - source_start_val
    end_offset = val_after_2024 - source_end_val

    # Shift timestamps and apply linear offset blend
    source["ds"] = source["ds"] - pd.DateOffset(years=1)
    n_hours = len(source)
    blend = np.linspace(0, 1, n_hours)
    offset = start_offset * (1 - blend) + end_offset * blend
    source_adjusted = source["y"].values + offset

    print(f"  Final: first={source_adjusted[0]:.2f}, last={source_adjusted[-1]:.2f}")

    source["y"] = source_adjusted
    source["filled"] = True

    # Replace gap
    df = df.copy()
    df_filled = (
        pd.concat(
            [
                df[df["ds"] < gap_start],
                source[["ds", "y", "filled"]],
                df[df["ds"] > gap_end],
            ]
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    return df_filled, {"n_hours": n_hours}


def fill_gap_from_previous_year(
    df: pd.DataFrame,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    context_days: int = 7,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Fill gap using data from same calendar dates next year.

    Strategy:
    1. Get source data from Jan 2025 (same calendar dates)
    2. Interpolate any NaNs in the source data
    3. Match boundaries to 2024 context using linear offset blending
    """
    print(f"  Gap: {gap_start} to {gap_end}")

    # Get source from 2024
    source_start_2024 = gap_start - pd.DateOffset(years=1)
    source_end_2024 = gap_end - pd.DateOffset(years=1)

    source = df[(df["ds"] >= source_start_2024) & (df["ds"] <= source_end_2024)].copy()
    # print(f"Deubg {source['ds'].max()} -  {source['ds'].min()}")

    if len(source) == 0:
        print("  ERROR: No source data found in 2024")
        return df, None

    print(f"  Source: {len(source)} hours from Feb 2024")

    # Handle NaNs in source
    n_source_nan = source["y"].isna().sum()
    if n_source_nan > 0:
        print(f"  Interpolating {n_source_nan} NaN values in source")
        source["y"] = source["y"].interpolate(method="linear", limit_direction="both")

    # Get context from 2024
    before = df[
        (df["ds"] >= gap_start - pd.Timedelta(days=context_days))
        & (df["ds"] < gap_start)
        & df["y"].notna()
    ]
    after = df[
        (df["ds"] > gap_end)
        & (df["ds"] <= gap_end + pd.Timedelta(days=context_days))
        & df["y"].notna()
    ]

    if len(before) < 24 or len(after) < 24:
        print(f"  ERROR: Not enough context (before={len(before)}, after={len(after)})")
        return df, None

    # Boundary values
    val_before_2025 = before["y"].iloc[-1]
    val_after_2025 = after["y"].iloc[0]
    source_start_val = source["y"].iloc[0]
    source_end_val = source["y"].iloc[-1]

    print(f"  2025 boundaries: start={val_before_2025:.2f}, end={val_after_2025:.2f}")
    print(f"  2024 source:     start={source_start_val:.2f}, end={source_end_val:.2f}")

    # Calculate offsets
    start_offset = val_before_2025 - source_start_val
    end_offset = val_after_2025 - source_end_val

    # Shift timestamps and apply linear offset blend
    source["ds"] = source["ds"] + pd.DateOffset(years=1)
    n_hours = len(source)
    blend = np.linspace(0, 1, n_hours)
    offset = start_offset * (1 - blend) + end_offset * blend
    source_adjusted = source["y"].values + offset

    print(f"  Final: first={source_adjusted[0]:.2f}, last={source_adjusted[-1]:.2f}")

    source["y"] = source_adjusted
    source["filled"] = True

    # Replace gap
    df = df.copy()
    df_filled = (
        pd.concat(
            [
                df[df["ds"] < gap_start],
                source[["ds", "y", "filled"]],
                df[df["ds"] > gap_end],
            ]
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )
    print(f"Deubg {source['ds'].max()} -  {source['ds'].min()}")
    print(f"Debug {source['y'].isna().sum()}")
    return df_filled, {"n_hours": n_hours}


# =============================================================================
# Main Pipeline
# =============================================================================


def fill_all_gaps(split: str = "train") -> pd.DataFrame:
    """
    Fill gaps in temperature data.

    Args:
        split: "train" (up to Feb 2025), "test" (after Feb 2025), or "all"

    Returns:
        DataFrame with filled gaps
    """
    print("=" * 60)
    print("Gap Filling Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw = df_raw[df_raw["timestamp"] >= CUTOFF].copy()
    df_raw["ds"] = df_raw["timestamp"]
    df_raw["y"] = df_raw["tempMean"]
    print(f"  Raw: {len(df_raw)} records")

    # Create full range
    full_range = pd.date_range(
        start=df_raw["timestamp"].min(),
        end=df_raw["timestamp"].max(),
        freq="15min",
        tz="UTC",
    )
    df = pd.DataFrame({"ds": full_range})
    df = df.merge(df_raw, on="ds", how="left")
    df["filled"] = False

    n_missing = df["y"].isna().sum()
    df.set_index("timestamp", inplace=True, drop=True)
    print(
        f"  Full range: {len(df)} hours, {n_missing} missing ({100 * n_missing / len(df):.2f}%)"
    )

    # Apply split
    if split == "train":
        df = df[df["ds"] < GAP_16DAY_START].copy()
        print(f"  Train split: {len(df)} hours (up to {GAP_16DAY_START})")
    elif split == "test":
        df = df[df["ds"] > GAP_16DAY_START - pd.Timedelta(days=400)].copy()
        print(f"  Test split: {len(df)} hours (after {GAP_16DAY_START})")

    # Identify gaps
    gap_info = identify_gaps(df)
    print(f"\n  Found {len(gap_info)} gaps")

    # Step 1: Linear interpolation for tiny gaps
    print("\n[2] Step 1: Linear interpolation (<3h)...")
    n_before = df["y"].isna().sum()
    # df["ds"] = pd.to_timedelta(df["ds"])
    df = (
        df.set_index("ds")
        .interpolate(method="time", limit=2 * FREQ, direction="both")
        .reset_index("ds")
    )
    n_after = df["y"].isna().sum()
    print(f"  Filled: {n_before - n_after} values")

    # Re-identify gaps
    gap_info = identify_gaps(df)

    # Step 2: Prophet for small/medium gaps (3h-72h)
    print("\n[3] Step 2: Prophet (3h-72h)...")
    GAP_8DAY_HOURS = int(8 * 24)
    GAP_16DAY_HOURS = int(16 * 24)
    prophet_gaps = gap_info[
        (gap_info["hours"] >= 2)
        & (gap_info["hours"] <= 72)
        & (gap_info["hours"] != GAP_8DAY_HOURS)
        & (gap_info["hours"] != GAP_16DAY_HOURS)
    ]
    print(f"  Found {len(prophet_gaps)} gaps to fill")

    for idx, gap in tqdm(
        prophet_gaps.iterrows(), total=len(prophet_gaps), desc="  Prophet"
    ):
        gap_start = gap["start"] - pd.Timedelta(hours=1)
        gap_end = gap["end"] + pd.Timedelta(hours=1)

        filled = fill_gap_prophet(df, gap_start, gap_end)
        if filled is not None:
            mask = (df["ds"] >= filled["ds"].min()) & (df["ds"] <= filled["ds"].max())
            df.loc[mask, "y"] = filled["y"].values
            df.loc[mask, "filled"] = True

    print(f"  After Step 2: {df['y'].isna().sum()} NaN remaining")

    df = df.set_index("ds").interpolate(method="time", limit=3 * FREQ).reset_index("ds")
    # Step 3: 8-day gap from 2025 data (only for train split)
    if split in ["train", "all"]:
        # Check if 8-day gap exists by looking for NaNs in that date range
        gap_mask = (df["ds"] >= GAP_8DAY_START) & (df["ds"] <= GAP_8DAY_END)
        n_gap_nan = df.loc[gap_mask, "y"].isna().sum()

        if n_gap_nan > 0:
            print(f"\n[4] Step 3: 8-day gap from Jan 2025 ({n_gap_nan} NaN hours)...")
            df, _ = fill_gap_from_next_year(df, GAP_8DAY_START, GAP_8DAY_END)
    if split in ["test"]:
        df, _ = fill_gap_from_previous_year(
            df, GAP_16DAY_START, GAP_16DAY_END, context_days=3
        )
        print(f"  After Step 3: {df['y'].isna().sum()} NaN remaining")
        df = df[df["ds"] >= GAP_16DAY_START]

    print(f"  After Step 3: {df['y'].isna().sum()} NaN remaining")

    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total hours: {len(df)}")
    print(f"  Filled hours: {df['filled'].sum()}")
    print(f"  Remaining NaN: {df['y'].isna().sum()}")

    return df


def fill_gaps_max_min(df_input):
    median_range = np.nanmedian((df_input["tempMax"] - df_input["tempMin"]).to_numpy())
    nanmask = (df_input["tempMin"].isna()) | (df_input["tempMax"].isna())
    df_input.loc[nanmask, "tempMin"] = df_input["y"] - median_range
    df_input.loc[nanmask, "tempMax"] = df_input["y"] + median_range
    df_input.loc[nanmask, "filled"] = True
    return df_input


def main():
    """Generate filled gaps CSV file."""
    # Fill train split (up to Feb 2025 gap)
    df_train = fill_all_gaps(split="train")

    # Fill test split (after Feb 2025 gap)
    df_test = fill_all_gaps(split="test")

    # Combine
    df_all = pd.concat([df_train, df_test]).sort_values("ds").reset_index(drop=True)
    df_all = (
        df_all.set_index("ds")
        .interpolate(method="time", limit=2 * FREQ, direction="both")
        .reset_index("ds")
    )
    df_all["timestamp"] = Time(df_all["ds"]).iso

    # fill gaps on min/max
    df_all = fill_gaps_max_min(df_all)

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df_all[["ds", "y", "tempMean", "tempMax", "tempMin", "filled", "timestamp"]].to_csv(
        OUTPUT_PATH, index=False
    )
    print(f"Done! {len(df_all)} hours saved.")
    print(f"Date ranges are {df_all['ds'].min()} and {df_all['ds'].max()}")
    print(f"Number of filled gaps is {df_all['filled'].sum()}")
    print(identify_gaps(df_all))


if __name__ == "__main__":
    main()
