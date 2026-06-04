"""Sweep NBEATSx max_steps at fixed width=256 and benchmark M5 (MPS).

Reuses run.py pipeline. Early stopping (patience=10, val_check_steps=50)
stays active, so higher max_steps may not train proportionally longer.

Usage:
    uv run --no-sync python forecast/sweep_maxsteps.py [--mb] [--accelerator mps|cpu]
"""

import os
import sys
import time
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # forecast/ on path
import run  # noqa: E402
from config import (  # noqa: E402
    NBEATS_INPUT_SIZE, NBEATS_HORIZON,
    SOLAR_GRID_FREQ, TEST_START_DATE,
)

WIDTH = int(os.environ.get("MS_WIDTH", 16))
MAX_STEPS_GRID = [100, 150, 200, 250, 300, 400, 500, 700, 900, 1200, 1500, 2000]
KEY_LEADS = [0.5, 1.0, 3.0, 6.0, 12.0]


def build_model(max_steps, accelerator):
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist_exog, futr_exog = run.get_hist_futr_exog(None)
    return NBEATSx(
        h=NBEATS_HORIZON,
        input_size=NBEATS_INPUT_SIZE,
        max_steps=max_steps,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=0.001,
        batch_size=48,
        scaler_type="identity",
        enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[WIDTH, WIDTH]],
        n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10,
        val_check_steps=50,
        accelerator=accelerator,
        devices=1,
    )


def train_timed(grid, model):
    from neuralforecast import NeuralForecast

    hist_exog, futr_exog = run.get_hist_futr_exog(None)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    # Train strictly pre-2025 (test set is all 2025; avoid leakage).
    train_mask = pd.to_datetime(grid["ds_real"]) < TEST_START_DATE
    nf_train = grid.loc[train_mask, ["ds", "D"] + all_exog].dropna().copy()
    nf_train["y"] = nf_train["D"]
    nf_train["unique_id"] = "temp"
    val_size = int(len(nf_train) * 0.1)

    nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
    t0 = time.perf_counter()
    nf.fit(nf_train, val_size=val_size)
    fit_s = time.perf_counter() - t0
    return nf, fit_s, len(nf_train)


def rmse_at(results, lead):
    s = results[results["lead_time_hours"] == lead]
    if len(s) == 0:
        return np.nan
    return float(np.sqrt((s["error"] ** 2).mean()))


def main(accelerator):
    print("=" * 78)
    print(f"NBEATSx max_steps SWEEP  |  width={WIDTH}  |  accelerator={accelerator}")
    print("=" * 78)

    grid, cfg = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  grid={len(grid)}, test twilights={len(tw_test)}\n")

    rows = []
    for ms in MAX_STEPS_GRID:
        print("-" * 78)
        print(f">>> max_steps = {ms}")
        model = build_model(ms, accelerator)
        nf, fit_s, n_train = train_timed(grid, model)

        t0 = time.perf_counter()
        results = run.predict_and_evaluate(nf, nf, grid, tw_test, cfg)
        pred_s = time.perf_counter() - t0

        row = {"max_steps": ms, "fit_s": fit_s, "pred_s": pred_s, "n_train": n_train}
        for lead in KEY_LEADS:
            row[f"rmse_{lead}h"] = rmse_at(results, lead)
        rows.append(row)
        print(f"    fit={fit_s:7.1f}s  predict={pred_s:6.1f}s  "
              f"RMSE@3h={row['rmse_3.0h']:.3f}  RMSE@12h={row['rmse_12.0h']:.3f}")

    df = pd.DataFrame(rows)
    out = run.RESULTS_PATH / "sweep_maxsteps.csv"
    run.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    hdr = f"{'steps':>6} {'fit(s)':>8} {'pred(s)':>8}"
    for lead in KEY_LEADS:
        hdr += f" {'R@'+str(lead):>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        line = f"{int(r['max_steps']):>6} {r['fit_s']:>8.1f} {r['pred_s']:>8.1f}"
        for lead in KEY_LEADS:
            line += f" {r[f'rmse_{lead}h']:>8.3f}"
        print(line)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", action="store_true")
    parser.add_argument("--accelerator", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    run.USE_MB = args.mb
    run.__dict__["USE_MB"] = args.mb

    main(args.accelerator)
