# ICE Enforcement Media Risk Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![R](https://img.shields.io/badge/R-4.x-276DC3)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A two-stage statistical pipeline that estimates the probability of a US county
appearing in **immigration enforcement media coverage** in the following month.
Outputs an interactive animated choropleth map across all 3,222 US counties,
with a county-level lookup returning predicted probability, percentile rank,
and Low / Medium / High risk classification for up to 6 months ahead.

> **⚠️ Media attention model, not an enforcement intensity model.**
> The dependent variable is derived from news coverage (GDELT), not from
> official enforcement records. Coverage is systematically lower in
> Republican-leaning and rural areas due to media geography —
> not because enforcement is less frequent there.
> See [Known Limitations](#known-limitations).

---

## Interactive Map + County Risk Lookup

**[→ Open interactive dashboard](https://WeihaoGe1009.github.io/ice-risk-modeling/)**

[![ICE Enforcement Media Risk Map](docs/preview.png)](https://WeihaoGe1009.github.io/ice-risk-modeling/)

The dashboard page (link above) contains two sections:

1. **Animated choropleth map** — all 3,222 US counties color-coded by risk
   score, with a month slider (Jan 2025 – Mar 2026) and hover details
2. **County risk lookup** (scroll down on the page) — select State → County →
   Month to get predicted probability, 95% interval, percentile rank among all
   counties, and Low / Medium / High risk classification for up to 6 months ahead

> GitHub README cannot run JavaScript, so the interactive content lives on the
> linked page, not here.

---

## Model Summary

### Stage 1 — ICE Arrest Rate Forecast (univariate time-series)

| | |
|---|---|
| **Input** | State-level monthly ICE arrest counts (Deportation Data Project FY23–26) |
| **Model** | 3-month moving average (MA3) per state |
| **Train / Val** | Oct 2022 – Sep 2025 / Oct 2025 – Mar 2026 |
| **Val RMSE** | 4.34 arrests per 100k (vs 4.62 for naive last-value) |
| **Purpose** | Forecast the offset term for Stage 2 when future arrest data is unavailable |

Three models were compared on the 6-month hold-out (Oct 2025 – Mar 2026):
naive last-value, MA3, and Holt's double exponential smoothing (ETS).
MA3 won; ETS collapsed to naive because inter-month variance dominates any
trend signal over the ~36-month training window.

### Stage 2 — County-level Outcome Model (mixed-effects logistic regression)

| | |
|---|---|
| **Outcome** | Binary: county has ≥1 media mention of immigration enforcement next month |
| **Data source** | GDELT GKG v1, deduplicated by CAMEO event ID |
| **Model** | Mixed-effects logistic regression (`lme4::glmer`) |
| **Features** | Demographics · political · geographic · temporal momentum |
| **Offset** | State-level ICE arrest rate — **actual** during training, **MA3-forecasted** for future months |
| **Random effect** | County intercept (3,222 counties) |
| **Train period** | 2025-01 → 2025-09 |
| **Val period** | 2025-10 → 2026-03 |
| **AUC-PR** | **0.626** (baseline ≈ 0.16) |
| **F2 / Recall / Precision** | **0.303 / 76.5% / 40.1%** at threshold 0.157 |
| **Mean prediction bias** | **+1.4 pp** (vs −10.6 pp with zero offset) |

### Why two stages?

The `glmer` model uses `offset(log1p(ice_arrest_rate_state))` — a fixed-coefficient
term that anchors each county's baseline to the state's enforcement intensity.
For future months this rate is unknown, so originally it was set to zero, causing a
systematic **−10.6 percentage-point underestimation** of all probabilities (confirmed
by walk-forward backtesting). Stage 1 forecasts this rate from its own history,
reducing the bias to **+1.4 pp** and nearly matching the oracle (actual future rate)
in validation.

### Key Findings

| Feature | Coefficient | Interpretation |
|---|---|---|
| Republican vote share 2024 | **−0.85 ★★★** | Strongest predictor — media geography, not enforcement |
| Urban (vs rural) | **+0.79 ★★★** | Urban areas generate far more coverage per operation |
| Border zone (100 mi) | **+0.70 ★★★** | ~2× higher baseline log-odds |
| Suburban (vs rural) | **+0.52 ★★★** | |
| % Poverty | **−0.42 ★★★** | Wealthier immigrant enclaves are more media-visible |
| % Foreign-born | **+0.13 ★★** | Established communities → more reported incidents |
| % Hispanic | **+0.11 ★★** | |
| Pop. density | **+0.20 ★★** | |
| Lag 1 month | **+0.23 ★★** | News momentum — prior coverage predicts next month |
| Lag 3 months | **+0.38 ★★** | |
| Crime rate, HH income | ≈ 0 | No meaningful signal |

★★★ p < 0.001 · ★★ p < 0.05 · open = not significant

### Calibration

With the MA3 arrest-rate offset, the model is well-calibrated across all deciles
(e.g. lowest decile: predicted 1.0% vs observed 1.7%; highest decile: predicted 75.2%
vs observed 69.7%). The original zero-offset version overestimated in the top decile
(33.9% predicted vs 69.7% observed) and underestimated everywhere else.
Mid-range predictions (0.2–0.5) should still be treated as relative rankings rather
than precise probabilities.

### Walk-forward Validation Summary (6 masked months, propagated lags)

| Offset strategy | AUC-ROC | AUC-PR | F2 | Recall | Bias |
|---|---|---|---|---|---|
| Zero (original) | 0.843 | 0.612 | 0.279 | 40.5% | −10.6 pp |
| **MA3 predicted** | **0.849** | **0.626** | **0.303** | **76.5%** | **+1.4 pp** |
| Oracle (actual rate) | 0.859 | 0.637 | 0.304 | 79.0% | +1.9 pp |

The predicted-MA3 and oracle rows are nearly identical, confirming the MA3
arrest-rate forecast is accurate enough to act as a proxy for the true future rate.

---

## How It Works

### Dependent variable

GDELT GKG v1 daily files (Jan 2025 – Apr 2026) are filtered to articles mentioning
`IMMIGRATION` plus at least one enforcement-specific theme:

```
DISCRIMINATION_IMMIGRATION_ANTIIMMIGRANT
TAX_FNCACT_IMMIGRATION_OFFICER
TAX_FNCACT_BORDER_PATROL_AGENT
WB_2491_BORDER_SECURITY
UNREST_CLOSINGBORDER
```

Articles are deduplicated by **CAMEO event ID** — multiple outlets covering the same
ICE operation count as one event, not one per outlet. Locations are resolved to
county FIPS via Census TIGER name matching + point-in-polygon spatial join fallback.

### Stage 1: arrest rate forecast

State-level monthly ICE arrest counts (from the Deportation Data Project FY23–26)
are aggregated and a **3-month moving average (MA3)** is fit per state.
Months with arrest counts below 25% of the state's trailing 12-month median
are flagged as incomplete (file-cutoff truncation) and excluded from training.

The MA3 forecast provides the `ice_arrest_rate_state` value fed into Stage 2's
offset for future months. For historical months within the training period,
the actual recorded rate is used.

### Stage 2: outcome model offset

The forecasted (or actual) rate is entered as `offset(log1p(rate))` — a fixed
coefficient of 1 that shifts the county's log-odds baseline by the state's
enforcement intensity without letting the model absorb this signal into the
demographic predictors.

### Model formula (R)

```r
glmer(
  outcome ~ pct_hispanic + pct_black + pct_asian + pct_foreign_born
          + median_hh_income + pct_poverty + pop_density
          + rep_vote_share_2024 + crime_rate_per_100k
          + lag_1m + lag_3m + lag_6m
          + in_border_zone + urban_rural
          + offset(log1p(ice_arrest_rate_state))
          + (1 | fips),
  family  = binomial(link = "logit"),
  data    = train_scaled     # features standardised on training set
)
```

> The model was specified with `cloglog` link (theoretically appropriate for rare
> binary events under a Poisson process) but fell back to `logit` — cloglog is
> numerically more sensitive near p ≈ 0, and the state-level offset pushed some
> predictions into the unstable region. At 15% prevalence the two links produce
> nearly identical results.

---

## Running Locally

### 1. Python environment

```bash
pip install -r requirements.txt
```

### 2. R packages

The training script auto-installs on first run:
```bash
Rscript model/train.R   # installs lme4, PRROC, dplyr, readr, tibble, jsonlite
```

### 3. Census API key (for `fetch_static.py` only)

```bash
export CENSUS_API_KEY=your_key_here   # free: https://api.census.gov/data/key_signup.html
```

### 4. Manual downloads (pipeline only — not needed to run the dashboard)

Two sources require a browser:

**A. MIT Election Lab — 2024 county presidential returns**
1. Go to <https://electionlab.mit.edu/data>
2. Download **County Presidential Returns 2000–2024**
3. Save as `data/raw/election/countypres_2000-2024.tab`

**B. FBI crime data**
1. Go to <https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads>
2. Download **Offenses Known to Law Enforcement** (most recent year)
3. Save as `data/raw/crime/fbi_county_crime_2024.csv`

If absent, `crime_rate_per_100k` is median-imputed. Safe to skip for development.

### 5. Run the full pipeline

```bash
# Fetch static county features (demographics, election, crime, border zone)
python data/fetch_static.py

# Fetch GDELT news data and aggregate to county × month
# (~480 days; resumable via checkpoint)
python data/fetch_gdelt.py

# Build model panel
python data/build_panel.py

# Train Stage 2 outcome model (saves artifacts to model/model_artifacts/)
Rscript model/train.R

# Build Stage 1 arrest rate model — validates MA3 vs naive/ETS,
# generates forecasts for future months
python scripts/arrest_rate_model.py

# Regenerate the interactive map (writes docs/index.html + docs/preview.png)
python scripts/generate_map.py

# Generate forward predictions with calibrated offset (writes docs/predictions.json,
# updates docs/index.html with lookup panel)
python scripts/forward_predict.py
```

### 6. Validate the pipeline

```bash
# Walk-forward backtesting: masks last 6 months, compares
# zero / predicted / oracle offset strategies
python scripts/validate_forward.py

# Optional: test a later cutoff (3 masked steps from Dec 2025)
python scripts/validate_forward.py --cutoff 2025-12 --steps 3
```

### 7. View the map without re-running the pipeline

Pre-computed artifacts are already in the repo. Just regenerate the HTML:

```bash
pip install plotly pandas kaleido==0.2.1
python scripts/generate_map.py
python scripts/forward_predict.py
# Then open docs/index.html in your browser
```

---

## Project Structure

```
data/
  fetch_static.py          # ACS, USDA RUCC, election, FBI crime, border zone
  fetch_gdelt.py           # GDELT GKG v1 → county × month binary indicators
  build_panel.py           # Full panel + lag features + ICE arrest rate
  processed/
    static_features.csv    # 3,222 counties × 14 static features (in repo)

model/
  train.R                  # glmer fit + threshold selection + artifact export
  model_artifacts/
    fixed_effects.csv      # Coefficients, SE, p-values, 95% CI
    random_effects.csv     # County BLUPs (3,222 intercepts)
    predictions_all.csv    # Full panel predictions (train + val)
    threshold.json         # Operating threshold + validation metrics
    calibration_data.csv   # Calibration bins for plot
    scale_params.csv       # Feature means/SDs (apply to new data)
    arrest_rate_history.csv      # Stage 1: clean state × month arrest rates (FY23–26)
    arrest_rate_forecasts.csv    # Stage 1: MA3 forecasts for future feature months
    arrest_rate_val_forecasts.csv# Stage 1: MA3 forecasts for val period (used by validate_forward.py)

scripts/
  arrest_rate_model.py     # Stage 1 — builds, validates, and saves arrest rate forecasts
  generate_map.py          # Builds docs/index.html (animated choropleth)
  forward_predict.py       # Forward predictions using MA3 offset; writes docs/predictions.json
  validate_forward.py      # Walk-forward backtesting (zero / predicted / oracle offset modes)

docs/
  index.html               # Interactive animated choropleth + lookup panel (GitHub Pages)
  predictions.json         # County × month forward predictions (6 months)
  preview.png              # Static map snapshot (README embed)

requirements.txt
```

---

## Known Limitations

- **Not a law-enforcement dataset.** GDELT captures media coverage, not
  ground-truth operations.
- **Republican-area blind spot.** The model underestimates risk in deeply
  conservative, rural areas where national media coverage is sparse.
  The negative Republican vote share coefficient reflects *media geography*,
  not enforcement pattern.
- **Low-confidence counties** (< 3 positive months in training data) have
  unreliable predictions. Flagged as `low_confidence` and toggleable on the
  dashboard map.
- **Static features are 2023 vintage.** Post-2023 demographic shifts are
  not captured.
- **MA3 arrest rate forecast is a flat projection.** The 3-month average is
  held constant across all 6 forecast months (no trend extrapolation). For
  states with rapidly changing enforcement activity the offset may drift.
  Validation shows RMSE 4.34 arrests/100k with +1.4 pp mean prediction bias.
- **Lag propagation accumulates rounding error.** Walk-forward validation
  shows oracle-lags and propagated-lags produce essentially identical metrics
  (Δ F2 < 0.001), so this error is negligible in practice.

---

## Data Sources

| Source | What | Vintage |
|---|---|---|
| [GDELT GKG v1](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/) | Dependent variable | Jan 2025 – Apr 2026 |
| [Deportation Data Project](https://deportationdata.org) | ICE arrest rate (Stage 1 + Stage 2 offset) | FY2023–2026 |
| [Census ACS DP05/DP02/DP03](https://www.census.gov/programs-surveys/acs/) | Demographics | 2023 5-yr |
| [USDA RUCC](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/) | Urban/rural classification | 2023 |
| [MIT Election Lab](https://electionlab.mit.edu/data) | 2024 presidential returns | 2024 |
| [FBI UCR/NIBRS](https://cde.ucr.cjis.gov/) | Crime rate per 100k | 2022 |
| [Census TIGER](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | County shapefiles | 2023 |

---

## License

MIT
