# Persistence-as-future-exogenous experiment — plan

## 1. Persistence definition

Issued at `now`, predicting time `now + h_steps` (h_steps in solar-grid steps,
1 step = 30 min, 48 steps = 1 solar day):

```
Pers(h) = y_raw_now + smooth( y_raw[now + h_steps − 48]  −  y_raw[now − 48] )
```

i.e. today's level at issue plus yesterday's *change* over the same
issue→target interval. For any h_steps ∈ [1, 24] (≤ 12 h lead) both lookups
are strictly in the past (≥ 24 steps before `now`), so the feature is fully
causal.

Two variants:
- `P_raw(h)`         — `smooth = identity`
- `P_smooth3h(h)`    — `smooth = centered 3 h rolling mean (6 steps)`,
  applied to `y_raw` *before* the day-lag, so the smoothing footprint is
  also entirely in the past of `now − 24` and remains causal.

The `now` row index is the **issue index**, not the target. For each
forecast request we therefore build a per-request future trajectory
of length `NBEATS_HORIZON`.

## 2. Decoupling check (why `y_prevday` alone is NOT the same)

`y_prevday(t) = y(t − 48)` is a *static* solar-grid column. Feeding it as
a futr_exog gives the network `y(t−48)` for every horizon step `t`. Combined
with the existing `y_diff_24h` hist feature evaluated at the issue row
(`y(now) − y(now−48)`), the network *can* reconstruct `Pers(h)` as:

```
Pers(h) = y_prevday(t) + [y(now) − y(now−48)] = y(t−48) + y_diff_24h_at_now
```

But that requires the network to **learn the addition**. That's a free win
asymptotically and a tax in low-data regimes. Because we're capacity-limited
(width=16, ~20k pre-2025 rows), feeding the **fully-formed** `Pers(h)`
trajectory directly is more sample-efficient.

We therefore feed the per-request `Pers(h)` series itself as a futr_exog.

## 3. Per-request future-exog construction

For each request `(issue_idx, target_idx, lead_h)`:

```python
horizon_idx = issue_idx + np.arange(NBEATS_HORIZON)         # the future steps
P_raw_traj    = y_arr[issue_idx]      \
              + (y_arr[horizon_idx - 48] - y_arr[issue_idx - 48])
P_smooth_traj = y_arr[issue_idx]      \
              + (y_smooth[horizon_idx - 48] - y_smooth[issue_idx - 48])
```

Both trajectories are appended as futr_exog columns to the per-request
`futr_df`. They are causal at every horizon step.

## 4. Variants compared

NBEATSx predicts **absolute T** directly (the differenced target
underperformed in the prior diagnostic; ↓7% raw RMSE @3h vs current).
Architecture: width=16, max_steps=700, 4 hist features
`[y_raw, y_lag_24, trend_solar_2h, y_diff_24h]`, det futr
`[solar_sin, solar_cos, doy_sin, doy_cos]`. Pre-2025 training, test = 2025.

| Variant            | Hist                | Futr (NBEATSx)                          |
|--------------------|---------------------|------------------------------------------|
| `base`             | 4 features          | 4 deterministic                          |
| `+P_raw`           | 4 features          | 4 det + `P_raw_traj`                     |
| `+P_smooth3h`      | 4 features          | 4 det + `P_smooth_traj`                  |

`P_*_traj` is a **per-request** futr_exog (custom column on `futr_df`).
**Crucially: it is a FUTR exog, never a hist exog.**

## 5. Reference baselines

- **Persistence-by-itself (no NBEATSx).** Same formula, evaluated directly
  at twilight for each twilight event in 2025. Two flavours: raw and 3h-smoothed.
  Reported at 3 h, 6 h, 12 h. *(This is what we compute first and gate on
  before any NBEATSx run.)*
- **Linear** baseline (already in build_final): per-lead Ridge on engineered
  solar-grid features.
- **Existing NBEATSx-Ridge** (Linear-blended): from current paper_results_final.csv.

## 6. Decision gate

After the experiment, look at 3 h / 6 h / 12 h RMSE:

- If `+P_smooth3h` (or `+P_raw`) beats `base` by **>7 %** at the
  *operationally critical* 3 h lead → escalate to **linear-as-exog**.
- If gain is in (0, 7 %] → keep persistence-exog as a paper variant but do
  not pursue linear-as-exog yet.
- If no gain → publish honest negative result; do not add the extra channel.

Long-lead (12 h) is reported but not the gate, because that's where the
local model has nearly run out of signal and any modest exog helps.

## 7. What is NOT in this experiment

- No leakage (training strictly pre-2025; futr_exog uses only `y[t]` with
  `t < issue_idx − 24`).
- No re-tuning of width / max_steps / feature set within this experiment;
  that was already done in the prior sweeps.
- No MeteoBlue NWP variant; that was de-scoped.
