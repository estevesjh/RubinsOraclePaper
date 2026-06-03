"""Sweep NBEATSx mlp_units width and benchmark M5 (MPS) performance.

Reuses the run.py pipeline (load_and_prepare, find_twilight_targets,
predict_and_evaluate) but retrains the long-lag model for each mlp_units
config. Reports fit wall-time, predict wall-time, and RMSE at key leads.

Usage:
    uv run --no-sync python forecast/sweep_mlp.py [--mb] [--accelerator mps|cpu]
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

sys.path.insert(0, os.path.dirname(__file__))
import run  # noqa: E402  (run.py sets up the twilight import path)
from config import (  # noqa: E402
    NBEATS_INPUT_SIZE, NBEATS_HORIZON, NBEATS_MAX_STEPS,
    SOLAR_GRID_FREQ, TEST_START_DATE, LEAD_TIMES_HOURS,
)

# Widths to sweep — each is the per-block hidden layer [w, w] x 4 stacks.
WIDTHS = [32, 256, 512, 1024, 2048]
KEY_LEADS = [0.5, 1.0, 3.0, 6.0, 12.0]


def build_model(width, accelerator):
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist_exog, futr_exog = run.get_hist_futr_exog(None)

    kwargs = dict(
        h=NBEATS_HORIZON,
        input_size=NBEATS_INPUT_SIZE,
        max_steps=NBEATS_MAX_STEPS,
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
        mlp_units=4 * [[width, width]],
        n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10,
        val_check_steps=50,
        accelerator=accelerator,
        devices=1,
    )
    return NBEATSx(**kwargs)


def train_timed(grid, model):
    from neuralforecast import NeuralForecast

    hist_exog, futr_exog = run.get_hist_futr_exog(None)
    hist_exog = [c for c in hist_exog if c in grid.columns]
    futr_exog = [c for c in futr_exog if c in grid.columns]
    all_exog = hist_exog + futr_exog

    nf_train = grid[["ds", "D"] + all_exog].dropna().copy()
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
    print(f"NBEATSx mlp_units SWEEP  |  accelerator={accelerator}  |  "
          f"max_steps={NBEATS_MAX_STEPS}")
    print("=" * 78)

    grid, cfg = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  Train rows feed grid={len(grid)}, test twilights={len(tw_test)}\n")

    rows = []
    for w in WIDTHS:
        print("-" * 78)
        print(f">>> mlp_units = 4 x [{w}, {w}]")
        model = build_model(w, accelerator)
        nf, fit_s, n_train = train_timed(grid, model)

        t0 = time.perf_counter()
        results = run.predict_and_evaluate(nf, nf, grid, tw_test, cfg)
        pred_s = time.perf_counter() - t0

        row = {"width": w, "fit_s": fit_s, "pred_s": pred_s, "n_train": n_train}
        for lead in KEY_LEADS:
            row[f"rmse_{lead}h"] = rmse_at(results, lead)
        rows.append(row)
        print(f"    fit={fit_s:7.1f}s  predict={pred_s:6.1f}s  "
              f"RMSE@3h={row['rmse_3.0h']:.3f}  RMSE@12h={row['rmse_12.0h']:.3f}")

    df = pd.DataFrame(rows)
    out = run.RESULTS_PATH / "sweep_mlp.csv"
    run.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    hdr = f"{'width':>6} {'fit(s)':>8} {'pred(s)':>8}"
    for lead in KEY_LEADS:
        hdr += f" {'R@'+str(lead):>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        line = f"{int(r['width']):>6} {r['fit_s']:>8.1f} {r['pred_s']:>8.1f}"
        for lead in KEY_LEADS:
            line += f" {r[f'rmse_{lead}h']:>8.3f}"
        print(line)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", action="store_true",
                        help="Include MeteoBlue corrected forecast as futr_exog")
    parser.add_argument("--accelerator", default="mps", choices=["mps", "cpu"],
                        help="Lightning accelerator (default: mps for M5 GPU)")
    args = parser.parse_args()

    run.USE_MB = args.mb
    globals_run = run.__dict__
    globals_run["USE_MB"] = args.mb

    main(args.accelerator)
