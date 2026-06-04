"""Validate persistence by itself (no NBEATSx) on the 2025 test twilights.

Persistence at lead h_steps issued at index `iss`:

    Pers(iss, h)        = y(iss) + ( y(iss + h - 48) - y(iss - 48) )
    Pers_sm(iss, h)     = y(iss) + ( y_sm(iss + h - 48) - y_sm(iss - 48) )

where y_sm = 6-step (3 h) centered rolling mean of y. For h <= 24 (<=12 h
lead), both lookups land at indices <= iss - 24 (causal).

Reports RMSE / MAE / pct<1C at 3 h, 6 h, 12 h on the 2025 test set, plus
the existing build_final 'Persistence' for sanity (it should match P_raw).
"""

import os
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run  # noqa: E402
from config import TEST_START_DATE, STEPS_PER_DAY  # noqa: E402

LEADS = [3.0, 6.0, 12.0]


def main():
    print("=" * 72)
    print("PERSISTENCE BASELINE  |  2025 test twilights, no NBEATSx")
    print("=" * 72)

    grid, _ = run.load_and_prepare()
    tw = run.find_twilight_targets(grid)
    tw_test = tw[tw["ds_real"] >= TEST_START_DATE].copy()
    print(f"  test twilights: {len(tw_test)}")

    y = grid["y"].values
    y_sm = pd.Series(y).rolling(6, min_periods=3, center=True).mean().values

    rows = []
    for L in LEADS:
        h_steps = run.lead_hours_to_steps(L)
        for variant, ysrc in [("P_raw", y), ("P_smooth3h", y_sm)]:
            errs = []
            for _, ev in tw_test.iterrows():
                tgt = int(ev["grid_idx"])
                iss = tgt - h_steps
                if iss - STEPS_PER_DAY < 0 or iss + h_steps - STEPS_PER_DAY < 0:
                    continue
                # Pers(iss, h)
                level = y[iss]
                yesterday_now = ysrc[iss - STEPS_PER_DAY]
                yesterday_target = ysrc[iss + h_steps - STEPS_PER_DAY]
                if any(np.isnan(v) for v in [level, yesterday_now, yesterday_target]):
                    continue
                pred = level + (yesterday_target - yesterday_now)
                errs.append(ev["y_actual"] - pred)
            e = np.array(errs)
            rmse = float(np.sqrt((e ** 2).mean())) if e.size else np.nan
            mae = float(np.abs(e).mean()) if e.size else np.nan
            p1 = float((np.abs(e) < 1.0).mean() * 100) if e.size else np.nan
            rows.append((L, variant, rmse, mae, p1, e.size))
            print(f"  lead={L:>4}h  {variant:>10}  RMSE={rmse:.3f}  "
                  f"MAE={mae:.3f}  <1C={p1:.0f}%  N={e.size}")

    print("\nDelta P_smooth3h vs P_raw at each lead:")
    for L in LEADS:
        rraw = next(r[2] for r in rows if r[0] == L and r[1] == "P_raw")
        rsm = next(r[2] for r in rows if r[0] == L and r[1] == "P_smooth3h")
        d = 100 * (rsm - rraw) / rraw
        print(f"  {L:>4}h:  {rraw:.3f} -> {rsm:.3f}   ({d:+.1f}%)")


if __name__ == "__main__":
    main()
