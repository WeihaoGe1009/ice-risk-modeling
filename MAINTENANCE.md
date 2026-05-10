# Monthly Maintenance Guide

## Monthly (~15 min)

1. Download the latest ICE arrest file from [Deportation Data Project](https://deportationdata.org)
   (usually 1–3 month lag behind current date)
2. Re-run the arrest rate model to extend history and refresh MA3 forecasts:
   ```bash
   python scripts/arrest_rate_model.py
   ```
3. Re-run forward predictions to roll the window forward by one month:
   ```bash
   python scripts/forward_predict.py
   ```
4. Commit and push `docs/predictions.json`, `docs/index.html`,
   `model/model_artifacts/arrest_rate_history.csv`,
   `model/model_artifacts/arrest_rate_forecasts.csv`

---

## Quarterly (~half day, mostly waiting)

1. Fetch new GDELT data (resumable from checkpoint, takes hours):
   ```bash
   python data/fetch_gdelt.py
   ```
2. Rebuild the panel with new outcome months:
   ```bash
   python data/build_panel.py
   ```
3. Retrain the outcome model (coefficients + BLUPs update):
   ```bash
   Rscript model/train.R
   ```
4. Regenerate the choropleth map:
   ```bash
   python scripts/generate_map.py
   ```
5. Re-run arrest rate model and forward predictions:
   ```bash
   python scripts/arrest_rate_model.py
   python scripts/forward_predict.py
   ```
6. Optionally re-run walk-forward validation to check model hasn't drifted:
   ```bash
   python scripts/validate_forward.py
   ```
7. Commit and push everything in `docs/` and `model/model_artifacts/`

---

## Annually (~1 hour)

1. Refresh static county features when new ACS 5-year estimates release
   (typically late fall each year):
   ```bash
   export CENSUS_API_KEY=your_key_here
   python data/fetch_static.py
   ```
2. Re-run full quarterly pipeline above to propagate updated demographics

---

## What you can ignore unless something major changes

- **Model structure** (`model/train.R`, glmer formula) — stable unless you add new predictors
- **Election data** — next update after 2028 presidential election
- **USDA RUCC codes** — update cycle is ~5 years
- **FBI crime data** — update annually if you care about that feature
  (currently median-imputed for most counties anyway, near-zero coefficient)

---

## Key data sources to watch

| Source | Where | Lag |
|---|---|---|
| ICE arrest files | [deportationdata.org](https://deportationdata.org) | 1–3 months |
| GDELT GKG v1 | Auto-fetched by `fetch_gdelt.py` | ~1 day |
| Census ACS | [census.gov](https://www.census.gov/programs-surveys/acs/) | ~1 year |
