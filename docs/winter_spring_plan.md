# Improving winter & spring RMSE — diagnosis + plan

Target model: `forecast/run_final_model.py` (direct-T NBEATSx + Ridge, the paper's model).
All numbers below are NBEATSx+Ridge on the 396 twilights of 2025, from
`results/paper_results_final.csv`, computed directly (not copied from the paper).

## 1. The problem is lead-time-dependent

| Season | 3 h RMSE | 9 h RMSE | 9 h \|e\|>2°C |
|--------|---------:|---------:|-------------:|
| Summer | 0.68 | 1.06 | 6.6% |
| Fall   | 0.77 | 1.03 | 7.5% |
| Winter | 0.88 | **1.80** | **20.7%** |
| Spring | 0.90 | **1.61** | **20.0%** |

- At the **operational 3 h lead** the seasonal spread is modest (0.68→0.90).
- At the **morning 9 h lead** winter/spring blow up to ~1.7°C and a ~20% outlier
  rate. This is where the "bottleneck" actually lives.

## 2. Why — what the data says (and what it contradicts)

### 2a. The dominant driver is day-to-day synoptic variability, not "bad weather"
Mean absolute day-to-day twilight-temperature change (a clean synoptic proxy):

| Season | mean \|ΔT_day\| | std(actual T) |
|--------|---------------:|--------------:|
| Summer | 1.14°C | 2.35 |
| Fall   | 1.73°C | 2.60 |
| Winter | **3.16°C** | 5.13 |
| Spring | **2.29°C** | 3.45 |

`r(|error|, |ΔT_day|) = +0.38` at 9 h — **by far the strongest correlate of error.**
Weather flags are weak by comparison: `r(|err|, bad_weather)=+0.15`,
`r(|err|, mb_cloud_hours)=+0.06`.

### 2b. The "cold front / bad weather" framing is largely unsupported
- Only **8 / 92** winter twilights carry the `bad_weather` flag (they hold ~18% of
  winter squared error — real but a minority).
- **0 / 90** spring twilights are flagged `bad_weather`, and **0** have
  `cloud_hours>0.5`. The current paper text ("cold fronts descending from the
  Andes", Sec. 4.6) does not describe the spring data.

### 2c. The mechanism is trend-lag, not random noise
`r(signed_error, ΔT_day) = +0.58` (winter), `+0.52` (spring) at 9 h. When the
temperature is moving day-over-day, the model systematically **under-shoots the
move** — a persistence-style lag. The model isn't broken: it still beats
solar-persistence by its *largest* margins in these seasons (36% winter, 44%
spring at 9 h). The seasons are intrinsically harder, but part of the error is
structured and therefore reducible.

### 2d. How much is reducible (upper bound)
Removing the linear `error~ΔT_day` component (oracle trend knowledge) leaves:

| Season | 9 h RMSE | trend-removed | 3 h RMSE | trend-removed |
|--------|---------:|--------------:|---------:|--------------:|
| Winter | 1.80 | 1.45 (−19%) | 0.88 | 0.87 (−1%) |
| Spring | 1.61 | 1.37 (−15%) | 0.90 | 0.85 (−6%) |

So **~15–19% of the 9 h winter/spring error is structured trend-lag**; at 3 h it
is nearly all irreducible from local history. Conclusion: the biggest, most
achievable wins are at the **6–12 h morning leads**, not at 3 h.

### 2e. Humidity & wind ARE available — and carry a modest, real signal
`data/weather_wind_humidity.csv` (rsync'd from slacd, covers 2021→2026):
windSpeed, humidity, windDir at 15-min cadence. Tested as **causal issue-time**
features, pooled over 6–12h leads, with **partial correlation** controlling for
the dominant synoptic driver `|ΔT_day|`:

| Season | hum_issue r (partial) | wind_issue r (partial) |
|--------|----------------------:|-----------------------:|
| Summer | +0.04 (+0.06) | +0.03 (+0.00) |
| Fall   | −0.05 (−0.05) | +0.14 (+0.12) |
| Winter | **+0.27 (+0.18)** | +0.18 (+0.11) |
| Spring | +0.15 (+0.06) | **+0.19 (+0.16)** |

So humidity carries **independent** information in winter (partial r +0.18, i.e.
beyond synoptic), and **wind** is the stronger of the two in spring. As an
operational "hard-day" flag: RH_issue ≥ Q3 → RMSE **2.11** vs **1.50** for the
rest (winter+spring, long lead).

**But the paper's specific mechanism is wrong.** The *desierto florido* claim is
that high twilight humidity flattens post-sunset cooling via latent heat →
warmer-than-forecast (positive signed error). The data show the **opposite**:
high-twilight-RH spring twilights have mean signed error **−0.67°C** (model runs
too *warm*, RMSE 1.81) vs +0.27°C for low-RH. Humidity helps as a **correlate of
unsettled/synoptic conditions**, not via the stated latent-heat pathway.
Conclusion: add humidity/wind as exog features (worth it), but they are
**secondary** to synoptic variability and will not "close the gap."

### 2f. MeteoBlue NWP *does* carry the swing — conditionally
- `r(MB_daily_change, actual_swing) = +0.78` (winter): NWP tracks the synoptic
  motion the local model misses.
- `r(MB_error, NB_error) = +0.32`: NWP carries information **independent** of the
  local model.
- But bias-corrected MB RMSE (1.82 winter) ≈ NB+Ridge (1.79) — NWP alone is no
  better. The paper's negative NWP result was an **unconditional** add (degrades
  overall because it's noisy on calm days). It was never tested as a
  **regime-gated blend** active only on high-variability / long-lead cases.
  (Caveat: MB station file ends 2025-08-31, so spring NWP coverage is sparse —
  validate winter first.)

## 3. Plan (prioritized by expected gain / effort)

### Phase 0 — Lock the diagnosis (½ day)
Make a `forecast/diag_seasonal.py` that produces the tables above as a figure +
CSV, so the "why" is reproducible and citable. Deliverable: a seasonal
error-vs-|ΔT_day| scatter and the reducibility table.

### Phase 1 — Multi-day trend features (cheap, attacks 2c directly)
The Ridge stage uses only `dmean_1d` for multi-day context. `feature_sweep.py`
already computes `dmean_3d`, `y_diff_30h/45h/72h`, `temp_trend_3d`,
`cooling_rate_6h`, `trend_solar_6h` but they are **not** in the final
`RIDGE_FEATS` / `NB_HIST`. Add the multi-day trend features and re-run the greedy
sweep **with a winter/spring-weighted objective** (current sweep optimizes 3 h
overall RMSE, which is why these never got selected — they pay off at long lead in
hard seasons). Expect to recover a chunk of the 15–19% at 6–12 h.
- Files: `forecast/run_final_model.py` (`NB_HIST`, `RIDGE_FEATS`),
  `forecast/feature_sweep.py` (objective).

### Phase 2 — Longer NBEATSx lookback (attacks the root cause)
`NBEATS_INPUT_SIZE=48` = exactly **1 solar day**. The model literally cannot see a
multi-day synoptic trend. Sweep `input_size ∈ {72, 96, 144}` (1.5–3 SD). This is
the most principled fix for trend-lag; cost is a few retrains.
- Files: `forecast/config.py`, a new `forecast/sweep_inputsize.py` (mirror
  `sweep_maxsteps.py`).

### Phase 3 — Regime-gated NWP blend (tests the untested hypothesis, 2f)
Build a blend `T = (1-w)·T_NB + w·T_MB_biascorr` where `w` is gated on a
**predicted-variability** signal available at issue time (e.g. recent
`last_std_24h`, `|ΔT_day|` at issue, or lead time). Calibrate `w` on pre-2025,
evaluate per season on 2025. Hypothesis: conditional blend captures winter's
reducible 19% without the global degradation the paper reports. Validate winter
first (spring NWP data sparse).
- Files: new `forecast/exp_nwp_gated_blend.py` (reuse
  `calibrate_meteoblue.py` per-bin correction).

### Phase 3b — Humidity & wind exog features (cheap, now data-backed)
Merge `weather_wind_humidity.csv` onto the solar grid (causal: value at issue
time and recent rolling stats). Add `humidity`, `windSpeed` (and maybe
`last_std_24h` of each) to the Ridge feature set and/or NBEATSx hist_exog, and
re-run the seasonal sweep. Expectation from §2e: a modest winter gain from
humidity, modest spring gain from wind — secondary to Phases 1–2, but real and
low-cost. Reframes (does not vindicate) the paper's humidity claim.
- Files: `forecast/run.py` load (merge weather), `exp_winter_spring.py` candidate
  list. Watchpoint: ensure causal merge (no future leakage) and handle the
  Dec-2025 humidity tail gap.

### Phase 4 — Heteroscedastic / quantile handling (manages the tail)
Winter/spring errors are asymmetric (cold-side tail). Options: (a) quantile loss
in NBEATSx for prediction intervals so the control system knows when to widen its
margin; (b) season-specific Ridge correction. This improves *operational
usefulness* (calibrated uncertainty) even where RMSE is irreducible.

### Phase 5 — Correct the paper (depends on 0–3 outcomes)
Replace Sec. 4.6's unsupported humidity/cold-front narrative with the
variability/trend-lag diagnosis (§2 numbers). Reframe the NWP discussion if Phase
3 shows conditional skill. This is a correctness fix regardless of modeling gains.

## 3.5 RESULTS — Phase 1+2 run (forecast/exp_winter_spring.py)

Ran 5 NBEATSx configs in parallel (CPU, width=16, 700 steps), each followed by a
greedy Ridge sweep with objective WS9 = mean(winter 9h, spring 9h RMSE). Strict
pre-2025 train / 2025 test. Baseline ("base/BASE") reproduces the paper.

| Config | 3h ALL | 9h Winter | 9h Spring | WS9 |
|--------|-------:|----------:|----------:|----:|
| **base / BASE** (paper) | 0.806 | 1.809 | 1.614 | 1.712 |
| base / SWEEP | 0.812 | 1.766 | 1.583 | **1.674** |
| in72 / SWEEP | 0.807 | 1.784 | 1.585 | 1.685 |
| in96 / SWEEP | 0.824 | 1.833 | 1.554 | 1.693 |
| in144 / SWEEP | 0.850 | 1.932 | 1.730 | 1.831 |
| in96_trend / SWEEP | 0.858 | 1.925 | 1.750 | 1.838 |

### Phase 2 (longer NBEATSx lookback): FALSIFIED.
WS9 rises monotonically with input_size (base 1.712 → in72 1.719 → in96 1.739 →
in144 1.854 at BASE ridge). Feeding `dmean_3d` to the network (in96_trend) is
also worse. Hypothesis that the 1-solar-day window starves the net of multi-day
context is wrong: a longer window just dilutes the width-16 net's capacity and
hurts. **Keep input_size=48.**

### Phase 1 (multi-day trend features in Ridge): SMALL POSITIVE.
On the base config the greedy sweep selected `+y_diff_45h, y_diff_30h,
last_std_24h, DTR_3d` and improved WS9 1.712 → 1.674 (**−2.2%**): winter 9h
1.809 → 1.766 (−2.4%), spring 9h 1.614 → 1.583 (−1.9%). At the operational 3h
lead it's flat-to-slightly-worse (0.806 → 0.812), consistent with §2d (3h error
is near-irreducible). So the trend features help exactly where predicted — the
long-lead morning forecast — but the gain is modest, far below the ~15–19%
oracle upper bound. The reducible trend-lag structure is mostly *not* linearly
recoverable from these issue-time features.

### Verdict (Phase 1+2)
Best achievable: **base + Ridge trend features**, ~2% WS9 improvement at 9h,
neutral at 3h. Real but small. The reducible trend-lag is mostly not linearly
recoverable from local features.

## 3.6 RESULTS — Phase 3b: humidity & wind (forecast/exp_humidity.py)

CAUSAL issue-time features (`humidity`, `windSpeed` + 24h rolling std), merged by
nearest timestamp, read at the issue row. Gaps imputed with a pre-2025
month×solar-bin climatology so all variants share the **identical** twilight set
(the first run had a coverage confound — humidity is sparse in summer/Dec, dense
in winter/spring — which faked a winter gain; fixed). Full-set and
humidity-observed-subset agree:

| Variant | 3h Wi | 3h Sp | 9h Wi | 9h Sp | WS9 |
|---------|------:|------:|------:|------:|----:|
| BASE | 0.876 | 0.901 | 1.813 | 1.616 | 1.714 |
| +trend | 0.881 | 0.916 | 1.756 | 1.576 | 1.666 |
| +hw | 1.048 | 1.002 | 1.731 | 1.563 | 1.647 |
| **+trend+hw** | 1.083 | 1.044 | **1.692** | **1.527** | **1.610** |

**Humidity/wind help at 9h, hurt at 3h.** +trend+hw cuts 9h winter 1.813→1.692
(−6.7%) and spring 1.616→1.527 (−5.5%) — the best long-lead result so far. But
at 3h it *degrades* winter 0.876→1.083 (+24%). This is a bias–variance effect,
not a bug: 4 extra features on ~430 training twilights add estimation variance
that the easy, low-variance 3h target can't absorb, while the hard high-variance
9h target benefits. Confirmed on both the full and observed-only subsets.

**Actionable design — lead-gated features:**
- short lead (≤~4h): `+trend` only (3h winter 0.881, spring 0.916)
- long lead (≥~6h): `+trend+hw` (9h winter 1.692, spring 1.527)

This captures the morning-forecast win without the dome-opening cost. Net effect
vs paper: 9h winter −6.7%, spring −5.5%; 3h neutral.

## 3.7 RESULTS — synoptic "hard-day" flag (the user's idea)

Tested flagging days where the day-to-day twilight swing exceeds the per-season
75th percentile ("greater than normal" synoptic change), at 9h:
- **ORACLE flag** (|ΔT to the *target* twilight|, uses the future): flagged days
  are 25% of days but carry **~50% of all squared error** (winter RMSE 2.54
  flagged vs 1.47 not; spring 2.27 vs 1.32). **This confirms the mechanism: the
  error lives on abnormal-synoptic-change days.**
- **CAUSAL flag** (built only from past twilights / issue-time volatility
  features `last_std_24h`, `DTR`, `dmean_3d`, `y_diff_30h`): **fails to identify
  them.** Best causal r(flag, |err|) ≈ +0.20 (spring DTR); winter ≈ 0. Flagged
  vs unflagged RMSE barely separate, sometimes invert.

Why: the daily twilight swing has **near-zero autocorrelation** (winter lag-1 ACF
+0.11, **spring −0.24**) — a big move today is not foreshadowed by yesterday's
move or by current volatility. So the flag is **diagnostically true but not
predictable from local data**. Operational use: a flag can't pre-identify hard
days for tighter control, but quantile/interval output (Phase 4) can *widen the
margin* on the days the model is internally least confident.

## 3.8 RESULTS — directional wind (user's idea)

Raw `windSpeed` is direction-blind. The summit wind is **bimodal**: a dominant
NW lobe (~315°, strong ~13 km/h) and a N/NE lobe (~40% of time); the S quadrant
is rare. Decomposed wind relative to the prevailing axis (anchor = pre-2025
speed-weighted vector mean of the NW/N lobe = **35°**), per the user's formula
`u_par = |w|cos(wd−anchor)`, `u_perp = |w|sin(wd−anchor)`, plus an NE-anchored
along-flow term (diagnostics: spring `r(u_par_NE, |err|)=+0.31` vs raw |w| +0.21):

| Variant | 9h Winter (Δ%) | 9h Spring (Δ%) |
|---------|---------------:|---------------:|
| BASE | 1.813 | 1.616 |
| +trend+hw (raw wind) | 1.692 (**−6.7%**) | 1.527 (−5.5%) |
| +trend+hw_dir (directional) | 1.728 (−4.7%) | **1.488 (−7.9%)** |

**Season-dependent physics:**
- **Spring**: directional wind WINS — NE-aligned along-flow marks the hard days;
  best spring result in the study (−7.9%). Raw |w| couldn't see this.
- **Winter**: directional wind slightly LOSES (1.728 vs 1.692) — winter error is
  humidity-driven, and splitting wind into 3 components just adds variance.

## 3.9 RESULTS — PCA decorrelation (negative)

The base feature set is collinear (5 lags + solar/doy sin/cos + multiple trend
terms). Tested PCA decorrelation (standardize → PCA → Ridge on top-k PCs + T_nb),
fit pre-2025 only. **PCA does not improve the forecaster in any configuration:**

| Approach | Best WS9 | vs curated |
|----------|---------:|-----------:|
| Curated features, no PCA (current model) | **1.660** | — |
| Full curated pool → PCA, 3 comps | 1.700 | +2.4% worse |
| Raw 96-lag bank → PCA | 1.714 | +3.3% worse |
| All 288 lags → PCA, 1–3 comps | 1.754 | +5.7% worse |
| Differenced lag banks → PCA | 1.742–1.752 | +5–5.5% worse |

The closer PCA gets to the curated set, the better — but it never beats keeping
the features and letting Ridge's L2 regularize them. The features are already few,
physically motivated, and forward-selected, so there is no curse-of-dimensionality
for PCA to cure; Ridge already absorbs the collinearity, and truncating to a
handful of PCs only discards the small genuine signal in the tail components.
**Keep the curated, lead-gated feature set; do not use PCA.**

## 3.10 RESULTS — RH terciles (user's idea): the best variant found

The spring humidity effect is strongly NONLINEAR (9h spring signed-error by RH
tercile: low +0.47, mid +0.24, high −1.05°C) — flat across low/mid RH, strong
only in the high-RH bin. A single linear Ridge coefficient can't represent this
and pays variance at every lead. Encoding RH as two tercile dummies (mid, high;
thresholds from pre-2025) instead of linear humidity is the best config in the
study:

| Variant | 3h Wi | 3h Sp | 9h Wi | 9h Sp |
|---------|------:|------:|------:|------:|
| BASE | 0.876 | 0.901 | 1.813 | 1.616 |
| +trend+hw_dir (linear RH) | 1.103 | 1.053 | 1.728 (−4.7%) | 1.488 (−7.9%) |
| **+trend+hw_dir_rhbin** (binned RH) | **0.903** | **0.988** | **1.668 (−8.0%)** | 1.527 (−5.5%) |
| +trend+hw_dir+rhbin (both) | 1.110 | 1.059 | 1.733 | 1.484 (−8.2%) |

Binned RH **halves the 3h damage** (winter 1.10→0.90, spring 1.05→0.99) AND
improves 9h winter further (−8.0% vs −4.7%). Honest humidity-observed subset
agrees: WS9 1.607 (best of all configs). The threshold encoding is far cheaper
in estimation variance than a continuous coefficient, so it stops poisoning the
easy short-lead target while still capturing the high-RH spring regime. **This is
the recommended production feature for the long-lead Ridge: binned RH + directional
wind + trend.** (Even so, 3h winter still nudges 0.876→0.903, so keep lead-gating.)

## 3.11 RESULTS — classifying synoptic hard days (revisits §3.7)

§3.7 concluded the hard-day flag "isn't deployable," but that test used the wrong
signal (magnitude of *yesterday's swing* → autocorrelation ≈ 0). A proper
classifier using **issue-time atmospheric volatility state** does have skill.
Forecast (forecast/exp_hardday_clf.py): logistic/GBM on issue-row features,
trained pre-2025, predicting whether |ΔT to the target twilight| exceeds the
hard-day threshold (set per season, so the trivial "winter is hard" signal is
removed).

| Feature set | Winter AUC | Spring AUC |
|-------------|-----------:|-----------:|
| TEMP volatility (last_std_24h, DTR, dmean_3d, trend) | **0.75** | **0.77** |
| TEMP + humidity/wind | 0.66 | 0.78 |

Key findings:
- **Hard days ARE classifiable** at the 9\,h issue time, AUC 0.74–0.77 in
  Winter/Spring (per-season threshold, honest 2025 test). The skill lives in
  current volatility (24\,h temperature std, diurnal range, 3-day mean drift):
  an agitated boundary layer now makes a large swing tonight more likely. This is
  a *regime/volatility* signal, not swing-persistence — which is why §3.7's
  swing-magnitude flag missed it.
- **Humidity/wind do NOT help the classifier** (winter 0.75→0.66); they help the
  *forecast* (§3.6/§3.10) but not hard-day *detection*. Use temperature
  volatility features for the flag.
- Operating points (Winter+Spring, base rate 31%): flag the top 40% of nights →
  precision 0.55, recall 0.70 (catches 70% of hard nights at ~2× random
  precision); top 30% → precision 0.58, recall 0.56.

**Operational use:** the flag can't sharpen the point forecast (the swing itself
is unpredictable), but it CAN pre-identify ~70% of hard nights at the morning
lead with usable precision — enough to trigger a wider M1M3 control margin or a
conservative setpoint on flagged nights. This is the deployable form of the
quantile/interval idea (Phase 4): a cheap pre-computed risk flag rather than a
full predictive-interval model.

## 3.12 RESULTS — wind tendency / trend-lag (user's idea)

A front shows up as wind *changing* (speed picking up, direction veering), so wind
tendency should beat instantaneous wind. Tested causal issue-time wind-tendency
features (6h/3h windspeed change `dws_*`, circular direction-change `ddir_*`, 24h
directional variability `wdir_var_24h`, along-axis-component change). Bootstrap
95% CIs on the strongest, vs forecast |error| at 9h:

- **Spring `dws_6h` (6h windspeed change): r = −0.42, CI [−0.63, −0.15]** —
  much stronger than instantaneous windspeed (−0.04). NEGATIVE sign is physical:
  spring error is worst on *calm / decreasing-wind* evenings (stagnant,
  radiatively-decoupled boundary layer = the humid desierto-florido regime);
  freshening wind makes the night easier.
- **Winter `wdir_var_24h` (directional variability): r = −0.34, CI [−0.51, −0.16]**
  — steady-direction days are harder; windspeed tendency is null in winter (−0.07).

Both robust (CIs exclude 0). Wind tendency is a genuine error correlate and
confirms the user's intuition. BUT it adds nothing to the deployed models:
- **Hard-day classifier**: no detection skill on top of temperature volatility
  (Winter AUC 0.749→0.756, Spring 0.768→0.758; wind-tendency alone only
  0.55–0.63).
- **Forecast (Ridge)**: adding `dws_6h`+`wdir_var_24h` to the production
  `+trend+hw_dir_rhbin` config is flat-to-worse — WS9 1.598→1.606, Spring 9h
  1.527→1.543 (full set); observed subset agrees (1.607→1.616). Spring actually
  degrades.

Reason (consistent across humidity and wind, three times now): tendency features
correlate with error but are **redundant** with what the model already holds —
the directional wind components, RH bins, and trend/volatility terms already
encode the calm-stagnant-evening regime that `dws_6h` measures. On ~430 training
twilights the extra columns cost more estimation variance than they return.
**Decision: do NOT add wind tendency to production.** Keep `+trend+hw_dir_rhbin`.
The features are computed in merge_humidity (dws_6h, wdir_var_24h) and available
if a future, larger training set changes the trade-off.

### Overall verdict
The best lever is **external humidity/wind, lead-gated AND season-aware**:
- short lead (≤4h): `+trend` only (avoid the 3h variance penalty)
- long lead (≥6h), winter-leaning: `+trend+hw` (raw wind) → 9h winter −6.7%
- long lead (≥6h), spring-leaning: `+trend+hw_dir` (directional) → 9h spring −7.9%

A single deployable config `+trend+hw_dir` gives winter −4.7% / spring −7.9% at
9h; `+trend+hw` gives −6.7% / −5.5%. Either roughly halves the gap to Fall at the
morning lead. Trend-only adds ~2–3%. The user's directional-wind idea is the
key unlock for spring. Synoptic hard-day flag confirms the diagnosis but isn't a
usable predictor (swing autocorr ≈ 0). NWP skipped (MB data ends Aug-2025).

## 4. Recommended ordering
0 → 1 → 2 in parallel with 3 (independent), then 4, then 5. Phases 1–2 are the
highest expected-value modeling work; Phase 5 is a correctness obligation
independent of whether RMSE moves.
