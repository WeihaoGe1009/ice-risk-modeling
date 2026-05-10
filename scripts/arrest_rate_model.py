#!/usr/bin/env python3
"""
scripts/arrest_rate_model.py

Univariate time-series model for state-level ICE arrest rates.

Problem
-------
forward_predict.py currently sets the ICE arrest rate offset to 0 for all
future months (data not yet available).  This systematically underestimates
probabilities by ~10 percentage points (confirmed in validate_forward.py).

Solution
--------
Build a simple univariate forecasting model per state using the full FY23-26
arrest records (Oct 2022 – present, ~42 months per state).

Models compared
---------------
  naive    — repeat last known value
  ma3      — 3-month rolling average (last 3 known months)
  ets      — Holt's double exponential smoothing (ETS with additive trend)
             via statsmodels.tsa.holtwinters.ExponentialSmoothing

Validation
----------
  Train : Oct 2022 – Sep 2025  (same cutoff as main outcome model)
  Val   : Oct 2025 – Mar 2026  (same as main model val period; 6 months)
  Metrics: RMSE and MAE per model, averaged across states

Outputs
-------
  model/model_artifacts/arrest_rate_history.csv
      state_fips, year_month, ice_arrests_state, ice_arrest_rate_state
      (all complete months, Oct 2022 – latest)

  model/model_artifacts/arrest_rate_forecasts.csv
      state_fips, year_month, rate_naive, rate_ma3, rate_ets, rate_selected
      (forecasts for May 2026 – Oct 2026, trained on full available data)

Usage
-----
    /opt/anaconda3/bin/python3 scripts/arrest_rate_model.py
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
ICE_DIR    = BASE / "data" / "raw" / "ice" / "2026-ICLI-00005_Arrests_Redacted"
STATIC     = BASE / "data" / "processed" / "static_features.csv"
ARTIFACTS  = BASE / "model" / "model_artifacts"

TRAIN_END   = pd.Timestamp("2025-09-01")   # last training month (inclusive)
VAL_END     = pd.Timestamp("2026-03-01")   # last val month (inclusive)
FORECAST_START = pd.Timestamp("2026-05-01")  # first month to forecast (outcome month)
FORECAST_END   = pd.Timestamp("2026-10-01")

# ── State name → 2-digit FIPS ──────────────────────────────────────────────────
STATE_NAME_TO_FIPS = {
    "ALABAMA":"01","ALASKA":"02","ARIZONA":"04","ARKANSAS":"05",
    "CALIFORNIA":"06","COLORADO":"08","CONNECTICUT":"09","DELAWARE":"10",
    "DISTRICT OF COLUMBIA":"11","FLORIDA":"12","GEORGIA":"13","HAWAII":"15",
    "IDAHO":"16","ILLINOIS":"17","INDIANA":"18","IOWA":"19","KANSAS":"20",
    "KENTUCKY":"21","LOUISIANA":"22","MAINE":"23","MARYLAND":"24",
    "MASSACHUSETTS":"25","MICHIGAN":"26","MINNESOTA":"27","MISSISSIPPI":"28",
    "MISSOURI":"29","MONTANA":"30","NEBRASKA":"31","NEVADA":"32",
    "NEW HAMPSHIRE":"33","NEW JERSEY":"34","NEW MEXICO":"35","NEW YORK":"36",
    "NORTH CAROLINA":"37","NORTH DAKOTA":"38","OHIO":"39","OKLAHOMA":"40",
    "OREGON":"41","PENNSYLVANIA":"42","RHODE ISLAND":"44",
    "SOUTH CAROLINA":"45","SOUTH DAKOTA":"46","TENNESSEE":"47","TEXAS":"48",
    "UTAH":"49","VERMONT":"50","VIRGINIA":"51","WASHINGTON":"53",
    "WEST VIRGINIA":"54","WISCONSIN":"55","WYOMING":"56",
}

# ── Load raw ICE arrests ───────────────────────────────────────────────────────
print("Loading raw ICE arrest files …")
dfs = []
for f in sorted(ICE_DIR.glob("*.xlsx")):
    df = pd.read_excel(f, header=6, usecols=["Apprehension Date", "State"])
    df.columns = ["arrest_date", "state"]
    df["arrest_date"] = pd.to_datetime(df["arrest_date"], errors="coerce")
    df["state"]       = df["state"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["arrest_date"])
    dfs.append(df)
    print(f"  {f.name}: {len(df):,} rows")

raw = pd.concat(dfs, ignore_index=True)
raw["state_fips"]  = raw["state"].map(STATE_NAME_TO_FIPS)
raw = raw.dropna(subset=["state_fips"])
raw["year_month"]  = raw["arrest_date"].dt.to_period("M").dt.to_timestamp()
print(f"Total valid rows: {len(raw):,}")

# ── State population for rate normalisation ────────────────────────────────────
static    = pd.read_csv(STATIC, dtype={"fips": str})
static["fips"] = static["fips"].str.zfill(5)
state_pop = (
    static.assign(state_fips=static["fips"].str[:2])
    .groupby("state_fips")["total_pop"]
    .sum()
    .reset_index()
)

# ── Aggregate to state × month counts ─────────────────────────────────────────
monthly = (
    raw.groupby(["state_fips", "year_month"])
    .size()
    .reset_index(name="ice_arrests_state")
)
monthly = monthly.merge(state_pop, on="state_fips", how="left")
monthly["ice_arrest_rate_state"] = (
    monthly["ice_arrests_state"] / monthly["total_pop"] * 100_000
)

# ── Detect and remove incomplete months ───────────────────────────────────────
# A month is "incomplete" if its arrest count is <25% of that state's median
# monthly count in the prior 12 months.  This catches file-cutoff truncation.
print("\nChecking for incomplete months …")

monthly = monthly.sort_values(["state_fips", "year_month"])

def flag_incomplete(grp):
    """Flag month if count < 25% of state's trailing 12-month median."""
    grp = grp.sort_values("year_month").copy()
    trailing_med = (
        grp["ice_arrests_state"]
        .shift(1)
        .rolling(12, min_periods=3)
        .median()
    )
    grp["incomplete"] = (
        (grp["ice_arrests_state"] < 0.25 * trailing_med) &
        trailing_med.notna()
    )
    return grp

monthly = monthly.groupby("state_fips", group_keys=False).apply(flag_incomplete)
n_incomplete = monthly["incomplete"].sum()
if n_incomplete > 0:
    bad = monthly[monthly["incomplete"]][["state_fips","year_month","ice_arrests_state"]]
    print(f"  Flagged {n_incomplete} incomplete state-months:")
    # Show unique months flagged
    bad_months = bad["year_month"].dt.strftime("%Y-%m").unique()
    print(f"  Months: {sorted(bad_months)}")

monthly_clean = monthly[~monthly["incomplete"]].copy()

print(f"\nClean monthly records: {len(monthly_clean):,}")
print(f"Date range: {monthly_clean['year_month'].min().strftime('%b %Y')} "
      f"→ {monthly_clean['year_month'].max().strftime('%b %Y')}")
print(f"States: {monthly_clean['state_fips'].nunique()}")

# Save history
hist_path = ARTIFACTS / "arrest_rate_history.csv"
monthly_clean[["state_fips","year_month","ice_arrests_state","ice_arrest_rate_state"]].to_csv(
    hist_path, index=False)
print(f"\nHistory saved → {hist_path}")

# ── Summary: arrest rate vs outcome prevalence ─────────────────────────────────
# Load panel for this relationship
print("\nArrest rate ↔ outcome prevalence (from panel):")
panel = pd.read_csv(BASE / "data" / "processed" / "panel.csv", dtype={"fips":str})
panel["year_month"] = pd.to_datetime(panel["year_month"])
state_outcome = (
    panel.assign(state_fips=panel["fips"].str[:2])
    .groupby(["state_fips","year_month"])
    .agg(outcome_rate=("outcome","mean"), rate=("ice_arrest_rate_state","first"))
    .reset_index()
)
corr_linear = state_outcome["rate"].corr(state_outcome["outcome_rate"])
corr_log    = np.log1p(state_outcome["rate"]).corr(state_outcome["outcome_rate"])
print(f"  Pearson r(rate, outcome)        = {corr_linear:.4f}")
print(f"  Pearson r(log1p(rate), outcome) = {corr_log:.4f}")
print(f"  Note: low correlation expected — state rate is a fixed-coefficient")
print(f"  offset in the model, not a free predictor. Its role is to anchor")
print(f"  the baseline rather than to explain cross-county variance.")

# ── Validation: train Oct 2022 – Sep 2025, val Oct 2025 – Mar 2026 ─────────────
print("\n" + "═"*65)
print("ARREST RATE MODEL VALIDATION  (train → Sep 2025 / val → Mar 2026)")
print("═"*65)

states = sorted(monthly_clean["state_fips"].unique())

val_months = pd.date_range(
    TRAIN_END + pd.DateOffset(months=1), VAL_END, freq="MS"
)
print(f"Val months: {[m.strftime('%b %Y') for m in val_months]}")

def fit_ets(series: pd.Series) -> ExponentialSmoothing:
    """Fit Holt's double ETS (additive trend, no seasonal)."""
    m = ExponentialSmoothing(series, trend="add", seasonal=None,
                             initialization_method="estimated")
    return m.fit(optimized=True, disp=False)

val_records = []
per_state   = []

for sf in states:
    grp = (monthly_clean[monthly_clean["state_fips"] == sf]
           .set_index("year_month")["ice_arrest_rate_state"]
           .sort_index())

    train_ser = grp[grp.index <= TRAIN_END]
    val_ser   = grp[(grp.index > TRAIN_END) & (grp.index <= VAL_END)]

    if len(train_ser) < 6 or len(val_ser) == 0:
        continue

    h = len(val_ser)

    # --- Naive: last training value repeated ---
    naive_fc = [float(train_ser.iloc[-1])] * h

    # --- MA3: mean of last 3 training values ---
    ma3_fc = [float(train_ser.iloc[-3:].mean())] * h

    # --- ETS: Holt's linear trend ---
    try:
        ets_model = fit_ets(train_ser)
        ets_fc    = ets_model.forecast(h).clip(lower=0).tolist()
    except Exception:
        ets_fc = naive_fc   # fallback

    actuals = val_ser.values.tolist()

    for i, (vm, actual) in enumerate(zip(val_ser.index, actuals)):
        val_records.append({
            "state_fips": sf, "year_month": vm,
            "actual": actual,
            "naive":  naive_fc[i],
            "ma3":    ma3_fc[i],
            "ets":    ets_fc[i],
        })

    # Per-state RMSE
    def rmse(preds, acts): return np.sqrt(np.mean((np.array(preds) - np.array(acts))**2))
    def mae(preds, acts):  return np.mean(np.abs(np.array(preds) - np.array(acts)))

    per_state.append({
        "state_fips":   sf,
        "n_train":      len(train_ser),
        "n_val":        h,
        "naive_rmse":   rmse(naive_fc, actuals),
        "ma3_rmse":     rmse(ma3_fc,   actuals),
        "ets_rmse":     rmse(ets_fc,   actuals),
        "naive_mae":    mae(naive_fc, actuals),
        "ma3_mae":      mae(ma3_fc,   actuals),
        "ets_mae":      mae(ets_fc,   actuals),
    })

val_df   = pd.DataFrame(val_records)
state_df = pd.DataFrame(per_state)

# Overall metrics
print(f"\n{'Model':<10} {'RMSE':>8} {'MAE':>8}   (per-state average)")
print("─"*30)
for model in ["naive","ma3","ets"]:
    rmse_avg = state_df[f"{model}_rmse"].mean()
    mae_avg  = state_df[f"{model}_mae"].mean()
    print(f"  {model:<8} {rmse_avg:>8.3f} {mae_avg:>8.3f}")

# Per-month averages
print("\nPer-month mean absolute error across states:")
print(f"  {'Month':<12} {'Actual':>8} {'Naive':>8} {'MA3':>8} {'ETS':>8}")
print("  " + "─"*44)
for vm in val_months:
    sub = val_df[val_df["year_month"] == vm]
    if sub.empty:
        continue
    row = sub.mean(numeric_only=True)
    print(f"  {vm.strftime('%b %Y'):<12} {row['actual']:>8.3f} "
          f"{row['naive']:>8.3f} {row['ma3']:>8.3f} {row['ets']:>8.3f}")

# National aggregate validation (sum across all states, rate per 100k national pop)
nat_val = val_df.groupby("year_month")[["actual","naive","ma3","ets"]].mean()
print("\nNational mean rate (averaged across states):")
print(nat_val.round(3).to_string())

# Pick best model
best_model = min(["naive","ma3","ets"],
                 key=lambda m: state_df[f"{m}_rmse"].mean())
print(f"\n→ Best model by RMSE: {best_model.upper()}")

# ── Forecasts: fit on ALL clean data → predict May 2026 – Oct 2026 ─────────────
print("\n" + "═"*65)
print("FORWARD FORECASTS  (fit on all clean data → May 2026 – Oct 2026)")
print("═"*65)

# The model's feature month for outcome M is M-1.
# forward_predict.py predicts outcomes May–Oct 2026.
# The offset in the linear predictor is log1p(rate at feature month T).
# Feature months for May–Oct 2026 are Apr–Sep 2026.
forecast_feature_months = pd.date_range("2026-04-01", "2026-09-01", freq="MS")

forecast_records = []

for sf in states:
    grp = (monthly_clean[monthly_clean["state_fips"] == sf]
           .set_index("year_month")["ice_arrest_rate_state"]
           .sort_index())

    if len(grp) < 6:
        continue

    h = len(forecast_feature_months)

    naive_fc = [float(grp.iloc[-1])] * h
    ma3_fc   = [float(grp.iloc[-3:].mean())] * h
    try:
        ets_m  = fit_ets(grp)
        ets_fc = ets_m.forecast(h).clip(lower=0).tolist()
    except Exception:
        ets_fc = naive_fc

    for i, fm in enumerate(forecast_feature_months):
        outcome_m = fm + pd.DateOffset(months=1)   # the outcome month
        forecast_records.append({
            "state_fips":        sf,
            "feature_month":     fm,
            "outcome_month":     outcome_m,
            "rate_naive":        round(naive_fc[i], 4),
            "rate_ma3":          round(ma3_fc[i],   4),
            "rate_ets":          round(ets_fc[i],   4),
            "rate_selected":     round({"naive":naive_fc,"ma3":ma3_fc,"ets":ets_fc}[best_model][i], 4),
        })

fc_df = pd.DataFrame(forecast_records)

# Print sample (California + Texas)
print("\nForecasts for California (06) and Texas (48):")
sample = fc_df[fc_df["state_fips"].isin(["06","48"])]
sample = sample.assign(
    feature=sample["feature_month"].dt.strftime("%b %Y"),
    outcome=sample["outcome_month"].dt.strftime("%b %Y"),
)
print(sample[["state_fips","feature","outcome","rate_naive","rate_ma3","rate_ets","rate_selected"]].to_string(index=False))

print("\nNational average forecasted rates (selected model):")
nat_fc = fc_df.groupby("outcome_month")["rate_selected"].mean()
for m, r in nat_fc.items():
    print(f"  {pd.Timestamp(m).strftime('%b %Y')}: {r:.3f}")

# Save
fc_path = ARTIFACTS / "arrest_rate_forecasts.csv"
fc_df.to_csv(fc_path, index=False)
print(f"\nForecasts saved → {fc_path}  ({fc_path.stat().st_size // 1024} KB)")

# ── Historical rates for val-period offset correction ──────────────────────────
# Also save val-period predictions (Oct 2025 – Mar 2026) for validate_forward.py
# Using the same approach: fit on data through Sep 2025
val_fc_records = []
for sf in states:
    grp = (monthly_clean[monthly_clean["state_fips"] == sf]
           .set_index("year_month")["ice_arrest_rate_state"]
           .sort_index())
    train_ser = grp[grp.index <= TRAIN_END]
    if len(train_ser) < 6:
        continue

    # val feature months: Sep 2025 + 1 = Oct 2025 ... Mar 2026
    # In the panel, feature month T → outcome T+1
    # val period in panel: Oct 2025 – Mar 2026 (feature months), outcomes Nov 2025 – Apr 2026
    val_feature_months = pd.date_range("2025-10-01", "2026-03-01", freq="MS")
    h = len(val_feature_months)

    naive_fc = [float(train_ser.iloc[-1])] * h
    ma3_fc   = [float(train_ser.iloc[-3:].mean())] * h
    try:
        ets_m  = fit_ets(train_ser)
        ets_fc = ets_m.forecast(h).clip(lower=0).tolist()
    except Exception:
        ets_fc = naive_fc

    for i, fm in enumerate(val_feature_months):
        val_fc_records.append({
            "state_fips":    sf,
            "feature_month": fm,
            "rate_naive":    round(naive_fc[i], 4),
            "rate_ma3":      round(ma3_fc[i],   4),
            "rate_ets":      round(ets_fc[i],   4),
            "rate_selected": round({"naive":naive_fc,"ma3":ma3_fc,"ets":ets_fc}[best_model][i], 4),
        })

val_fc_df = pd.DataFrame(val_fc_records)
val_fc_path = ARTIFACTS / "arrest_rate_val_forecasts.csv"
val_fc_df.to_csv(val_fc_path, index=False)
print(f"Val-period forecasts saved → {val_fc_path}")

print("\nDone. Next steps:")
print("  1. Run validate_forward.py  -- compares zero / predicted / oracle offsets")
print("  2. Run forward_predict.py   -- uses predicted offset for May–Oct 2026")
