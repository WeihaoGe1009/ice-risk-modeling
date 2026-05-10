#!/usr/bin/env python3
"""
build_panel.py

Builds the model-ready panel dataset:
  - Full county × month grid (3,222 counties × 16 months)
  - Dependent variable: has_mention_next (binary, t+1)
  - Rolling lag features: county-level GDELT mention counts over prior 1/3/6 months
  - Static features joined from static_features.csv
  - State-level ICE arrest rate (monthly, from Deportation Data Project files)
  - Low-confidence flag: counties with < N total positive months

Output: data/processed/panel.csv

Train/validate split is temporal — stored as a column 'split'
  train: 2025-01 through 2025-09  (features at t, outcome at t+1 i.e. up to 2025-10)
  val:   2025-10 through 2026-03  (features at t, outcome at t+1 i.e. up to 2026-04)
  The final month (2026-04) has no t+1 outcome and is excluded.
"""

import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED      = Path("data/processed")
GDELT_PATH     = PROCESSED / "gdelt_county_month.csv"
STATIC_PATH    = PROCESSED / "static_features.csv"
ICE_DIR        = Path("data/raw/ice/2026-ICLI-00005_Arrests_Redacted")
OUT_PATH       = PROCESSED / "panel.csv"

LOW_CONF_N     = 3   # flag counties with fewer than this many positive months
TRAIN_END      = "2025-09"   # last training month (inclusive)
VAL_END        = "2026-03"   # last validation month (inclusive, t+1 = 2026-04)

# ---------------------------------------------------------------------------
# State name → 2-digit FIPS
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 1. Load GDELT county × month mentions
# ---------------------------------------------------------------------------
def load_gdelt() -> pd.DataFrame:
    log.info("Loading GDELT county-month data ...")
    df = pd.read_csv(GDELT_PATH, dtype={"fips": str})
    df["year_month"] = pd.to_datetime(df["year_month"] + "-01")
    log.info("  %d rows, %d counties, %d months",
             len(df), df["fips"].nunique(), df["year_month"].nunique())
    return df


# ---------------------------------------------------------------------------
# 2. Build full county × month grid and fill zeros
# ---------------------------------------------------------------------------
def build_full_grid(gdelt: pd.DataFrame,
                    static: pd.DataFrame) -> pd.DataFrame:
    log.info("Building full county × month grid ...")
    all_fips   = static["fips"].unique()
    all_months = pd.date_range("2025-01-01", "2026-04-01", freq="MS")

    grid = pd.MultiIndex.from_product(
        [all_fips, all_months], names=["fips", "year_month"]
    ).to_frame(index=False)
    grid["fips"] = grid["fips"].astype(str).str.zfill(5)

    merged = grid.merge(
        gdelt[["fips", "year_month", "mention_count", "has_mention"]],
        on=["fips", "year_month"], how="left"
    )
    merged["mention_count"] = merged["mention_count"].fillna(0).astype(int)
    merged["has_mention"]   = merged["has_mention"].fillna(0).astype(int)

    log.info("  Grid shape: %s", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# 3. Rolling lag features (computed within each county, sorted by time)
# ---------------------------------------------------------------------------
def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing rolling lag features ...")
    panel = panel.sort_values(["fips", "year_month"]).copy()

    grp = panel.groupby("fips")["mention_count"]

    # Rolling sums over prior 1, 3, 6 months (min_periods=1 to avoid NaN)
    panel["lag_1m"]  = grp.transform(
        lambda x: x.shift(1).rolling(1, min_periods=1).sum()
    )
    panel["lag_3m"]  = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    panel["lag_6m"]  = grp.transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).sum()
    )

    # Fill any remaining NaN (first row per county) with 0
    for col in ["lag_1m", "lag_3m", "lag_6m"]:
        panel[col] = panel[col].fillna(0).astype(int)

    return panel


# ---------------------------------------------------------------------------
# 4. Dependent variable: has_mention at t+1
# ---------------------------------------------------------------------------
def add_outcome(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Creating t+1 outcome variable ...")
    panel = panel.sort_values(["fips", "year_month"]).copy()

    next_month = (
        panel.set_index(["fips", "year_month"])["has_mention"]
        .groupby(level="fips")
        .shift(-1)
        .reset_index()
        .rename(columns={"has_mention": "outcome"})
    )
    panel = panel.merge(next_month, on=["fips", "year_month"], how="left")

    # Drop the last month (no t+1 available)
    panel = panel[panel["outcome"].notna()].copy()
    panel["outcome"] = panel["outcome"].astype(int)
    return panel


# ---------------------------------------------------------------------------
# 5. State-level ICE arrest rate (monthly calibration covariate)
# ---------------------------------------------------------------------------
def load_ice_state_monthly(static: pd.DataFrame) -> pd.DataFrame:
    log.info("Loading ICE arrest files for state-level monthly counts ...")
    files = sorted(glob.glob(str(ICE_DIR / "*.xlsx")))
    if not files:
        log.warning("No ICE arrest files found in %s", ICE_DIR)
        return pd.DataFrame(columns=["state_fips", "year_month",
                                     "ice_arrests_state"])

    dfs = []
    for f in files:
        df = pd.read_excel(f, header=6,
                           usecols=["Apprehension Date", "State"])
        dfs.append(df)

    arrests = pd.concat(dfs, ignore_index=True)
    arrests = arrests.dropna(subset=["Apprehension Date"])
    arrests["year_month"] = pd.to_datetime(
        arrests["Apprehension Date"]
    ).dt.to_period("M").dt.to_timestamp()
    arrests["State"] = arrests["State"].astype(str).str.upper().str.strip()
    arrests["state_fips"] = arrests["State"].map(STATE_NAME_TO_FIPS)
    arrests = arrests.dropna(subset=["state_fips"])

    # Filter to panel date range
    arrests = arrests[
        (arrests["year_month"] >= "2025-01-01") &
        (arrests["year_month"] <= "2026-04-01")
    ]

    monthly = (
        arrests.groupby(["state_fips", "year_month"])
        .size()
        .reset_index(name="ice_arrests_state")
    )

    # Merge state population to compute rate per 100k
    state_pop = (
        static.assign(state_fips=static["fips"].str[:2])
        .groupby("state_fips")["total_pop"]
        .sum()
        .reset_index()
    )
    monthly = monthly.merge(state_pop, on="state_fips", how="left")
    monthly["ice_arrest_rate_state"] = (
        monthly["ice_arrests_state"] / monthly["total_pop"] * 100_000
    )
    monthly = monthly.drop(columns=["total_pop"])

    log.info("  State-month ICE arrest rows: %d", len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# 6. Low-confidence flag
# ---------------------------------------------------------------------------
def add_low_confidence(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Flagging low-confidence counties (< %d positive months) ...",
             LOW_CONF_N)
    pos_counts = (
        panel.groupby("fips")["has_mention"]
        .sum()
        .reset_index(name="total_positive_months")
    )
    panel = panel.merge(pos_counts, on="fips", how="left")
    panel["low_confidence"] = (
        panel["total_positive_months"] < LOW_CONF_N
    ).astype(int)
    return panel


# ---------------------------------------------------------------------------
# 7. Train / validation split label
# ---------------------------------------------------------------------------
def add_split(panel: pd.DataFrame) -> pd.DataFrame:
    panel["split"] = "train"
    panel.loc[panel["year_month"] > TRAIN_END, "split"] = "val"
    train_n = (panel["split"] == "train").sum()
    val_n   = (panel["split"] == "val").sum()
    log.info("  Train rows: %d | Val rows: %d", train_n, val_n)
    return panel


# ---------------------------------------------------------------------------
# 8. Final assembly
# ---------------------------------------------------------------------------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    static  = pd.read_csv(STATIC_PATH, dtype={"fips": str})
    static["fips"] = static["fips"].str.zfill(5)
    log.info("Static features: %d counties", len(static))

    gdelt   = load_gdelt()
    panel   = build_full_grid(gdelt, static)
    panel   = add_lag_features(panel)
    panel   = add_outcome(panel)

    # Join static features
    log.info("Joining static features ...")
    panel = panel.merge(static, on="fips", how="left")

    # Derive state_fips for joining ICE data
    panel["state_fips"] = panel["fips"].str[:2]

    # Join state-level ICE arrest rate
    ice_state = load_ice_state_monthly(static)
    if not ice_state.empty:
        panel = panel.merge(
            ice_state[["state_fips", "year_month",
                        "ice_arrests_state", "ice_arrest_rate_state"]],
            on=["state_fips", "year_month"], how="left"
        )
        panel["ice_arrests_state"]    = panel["ice_arrests_state"].fillna(0)
        panel["ice_arrest_rate_state"] = panel["ice_arrest_rate_state"].fillna(0)
    else:
        panel["ice_arrests_state"]    = 0
        panel["ice_arrest_rate_state"] = 0

    panel = add_low_confidence(panel)
    panel = add_split(panel)

    # Drop the last month column used only for outcome construction
    panel = panel.drop(columns=["state_fips"], errors="ignore")

    # Missing value summary
    log.info("Missing value summary:")
    missing = panel.isnull().sum()
    missing = missing[missing > 0]
    for col, n in missing.items():
        log.info("  %-35s %d (%.1f%%)", col, n, 100 * n / len(panel))

    panel.to_csv(OUT_PATH, index=False)
    log.info("Panel saved → %s", OUT_PATH)
    log.info("Shape: %s | Outcome prevalence: %.2f%%",
             panel.shape,
             100 * panel["outcome"].mean())

    # Quick sanity check
    log.info("\nSample (first 3 rows):")
    cols = ["fips", "year_month", "lag_1m", "lag_3m", "lag_6m",
            "outcome", "ice_arrest_rate_state", "split", "low_confidence"]
    log.info("\n%s", panel[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
