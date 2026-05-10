"""
Fetch and process all static (non-daily) county-level features.

Sources:
  1. US Census ACS 5-Year (2023 vintage) — demographics via REST API
  2. USDA ERS Rural-Urban Continuum Codes (2023) — direct download
  3. MIT Election Lab — 2024 county presidential returns (Harvard Dataverse)
  4. FBI Crime Data Explorer — annual county crime rates
  5. Border zone flag — computed from Census TIGER shapefiles

Output: data/processed/static_features.csv  (one row per FIPS county code)

Usage:
  export CENSUS_API_KEY=your_key_here
  python data/fetch_static.py

  Census API keys are free: https://api.census.gov/data/key_signup.html
"""

import os
import io
import time
import zipfile
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
ACS_VINTAGE = 2023


# ---------------------------------------------------------------------------
# 1. Census ACS demographics
# ---------------------------------------------------------------------------

# Variable mappings across the three ACS detail tables we use.
# Profile tables (DP*) return PE (percent estimate) and E (estimate) suffixes.
ACS_VARS = {
    # DP05 — Demographic and Housing Estimates
    "pct_hispanic":    "DP05_0071PE",   # % Hispanic or Latino
    "pct_black":       "DP05_0078PE",   # % Black or African American alone
    "pct_asian":       "DP05_0080PE",   # % Asian alone
    "total_pop":       "DP05_0001E",    # Total population (for density denominator)
    # DP03 — Selected Economic Characteristics
    "median_hh_income":"DP03_0062E",    # Median household income ($)
    "pct_poverty":     "DP03_0128PE",   # % persons below poverty level
    # DP02 — Selected Social Characteristics
    "pct_foreign_born":"DP02_0093PE",   # % foreign born
}

# The three profile tables that contain the variables above
ACS_TABLES = {
    "DP05": ["DP05_0071PE", "DP05_0078PE", "DP05_0080PE", "DP05_0001E"],
    "DP03": ["DP03_0062E", "DP03_0128PE"],
    "DP02": ["DP02_0093PE"],
}


def fetch_acs_table(table: str, variables: list) -> pd.DataFrame:
    """Query one ACS profile table for all counties."""
    var_str = ",".join(["NAME"] + variables)
    url = (
        f"https://api.census.gov/data/{ACS_VINTAGE}/acs/acs5/profile"
        f"?get={var_str}&for=county:*&key={CENSUS_API_KEY}"
    )
    log.info("Querying ACS table %s ...", table)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["fips"] = df["state"] + df["county"]
    df = df.drop(columns=["NAME", "state", "county"])
    # coerce numerics; Census encodes missing as "-" strings
    for col in variables:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_census_acs() -> pd.DataFrame:
    out_path = RAW / "census_acs" / "acs_demographics.csv"
    if out_path.exists():
        log.info("ACS file already exists, loading from cache.")
        return pd.read_csv(out_path, dtype={"fips": str})

    if not CENSUS_API_KEY:
        raise EnvironmentError(
            "Set CENSUS_API_KEY environment variable before running.\n"
            "Free key: https://api.census.gov/data/key_signup.html"
        )

    frames = []
    for table, variables in ACS_TABLES.items():
        df = fetch_acs_table(table, variables)
        frames.append(df.set_index("fips"))
        time.sleep(0.5)  # be polite to Census API

    combined = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "fips"})

    # Rename to friendly names
    rev = {v: k for k, v in ACS_VARS.items()}
    combined = combined.rename(columns=rev)

    combined.to_csv(out_path, index=False)
    log.info("ACS demographics saved → %s  (%d counties)", out_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# 2. USDA Rural-Urban Continuum Codes
# ---------------------------------------------------------------------------

RUCC_URL = "https://www.ers.usda.gov/media/5767/2023-rural-urban-continuum-codes.xlsx"

RUCC_MAP = {
    1: "urban", 2: "urban", 3: "urban",
    4: "suburban", 5: "suburban", 6: "suburban",
    7: "rural", 8: "rural", 9: "rural",
}


def fetch_rucc() -> pd.DataFrame:
    out_path = RAW / "usda_rucc" / "rucc_2023.xlsx"
    if not out_path.exists():
        log.info("Downloading USDA RUCC 2023 ...")
        resp = requests.get(RUCC_URL, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        log.info("RUCC saved → %s", out_path)

    df = pd.read_excel(out_path, dtype={"FIPS": str})
    df["fips"] = df["FIPS"].str.zfill(5)
    df["rucc_code"] = pd.to_numeric(df["RUCC_2023"], errors="coerce")
    df["urban_rural"] = df["rucc_code"].map(RUCC_MAP).fillna("unknown")
    return df[["fips", "rucc_code", "urban_rural"]]


# ---------------------------------------------------------------------------
# 3. 2024 county presidential vote share (MIT Election Lab via Harvard Dataverse)
# ---------------------------------------------------------------------------

# Harvard Dataverse permanent DOI download for MIT county presidential returns.
# If 2024 data is not yet posted, the script falls back to a stub warning.
MIT_ELECTION_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/"
    "8302168"   # datafile ID for countypres_2000-2024.csv
)


def fetch_election() -> pd.DataFrame:
    # Accept either hyphens or underscores in filename, .tab or .csv
    candidates = [
        (RAW / "election" / "countypres_2000-2024.tab", "\t"),
        (RAW / "election" / "countypres_2000_2024.tab", "\t"),
        (RAW / "election" / "countypres_2000-2024.csv", ","),
        (RAW / "election" / "countypres_2000_2024.csv", ","),
    ]
    found = [(p, s) for p, s in candidates if p.exists()]

    if found:
        out_path, sep = found[0]
    else:
        log.info("Attempting MIT Election Lab auto-download ...")
        resp = requests.get(MIT_ELECTION_URL, timeout=120)
        if resp.status_code != 200:
            fallback = RAW / "election" / "countypres_2000-2024.tab"
            log.warning(
                "MIT Election data unavailable (status %d). "
                "Download manually from https://electionlab.mit.edu/data "
                "and save to %s",
                resp.status_code, fallback,
            )
            return pd.DataFrame(columns=["fips", "rep_vote_share_2024"])
        fallback = RAW / "election" / "countypres_2000-2024.tab"
        fallback.write_bytes(resp.content)
        out_path, sep = fallback, "\t"
        log.info("Election data saved → %s", out_path)

    df = pd.read_csv(out_path, sep=sep, dtype={"county_fips": str})
    df2024 = df[df["year"] == 2024].copy()

    if df2024.empty:
        log.warning("2024 data not found in election file — using 2020 as proxy.")
        df2024 = df[df["year"] == 2020].copy()

    df2024["fips"] = df2024["county_fips"].str.zfill(5)

    # Compute Republican two-party vote share per county
    pivot = df2024.pivot_table(
        index="fips", columns="party", values="candidatevotes", aggfunc="sum"
    )
    pivot.columns = [c.upper() for c in pivot.columns]

    rep_col = next((c for c in pivot.columns if "REPUBLICAN" in c), None)
    dem_col = next((c for c in pivot.columns if "DEMOCRAT" in c), None)

    if rep_col is None:
        log.warning("Republican column not found in election data.")
        return pd.DataFrame(columns=["fips", "rep_vote_share_2024"])

    pivot["rep_vote_share_2024"] = pivot[rep_col] / (
        pivot.get(rep_col, 0) + pivot.get(dem_col, 0)
    )
    return pivot[["rep_vote_share_2024"]].reset_index()


# ---------------------------------------------------------------------------
# 4. FBI Crime Data — county-level crime rate
# ---------------------------------------------------------------------------
# Download "Offenses Known to Law Enforcement" zip from:
#   https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads
# Save the zip as-is to data/raw/crime/ (e.g. offenses-known-to-le-2024.zip).
# The script reads Table 10 (county-level) directly from the zip.

CRIME_ZIP_GLOB = "offenses-known-to-le-*.zip"
TABLE10_GLOB = "CIUS_Table_10_Offenses_Known*Counties*.xlsx"


def _build_county_fips_crosswalk() -> pd.DataFrame:
    """State name + county name → FIPS, built from the TIGER shapefile."""
    shp_candidates = list((RAW / "shapefiles").glob("tl_2023_us_county*.shp"))
    if not shp_candidates:
        log.warning("TIGER shapefile not found; crime FIPS crosswalk will be empty.")
        return pd.DataFrame(columns=["state_name", "county_name", "fips"])

    tiger = gpd.read_file(shp_candidates[0])[["STATEFP", "COUNTYFP", "NAME", "NAMELSAD"]]
    tiger["fips"] = tiger["STATEFP"] + tiger["COUNTYFP"]

    # State FIPS → state name via a small lookup (Census standard)
    state_fips = {
        "01":"ALABAMA","02":"ALASKA","04":"ARIZONA","05":"ARKANSAS","06":"CALIFORNIA",
        "08":"COLORADO","09":"CONNECTICUT","10":"DELAWARE","11":"DISTRICT OF COLUMBIA",
        "12":"FLORIDA","13":"GEORGIA","15":"HAWAII","16":"IDAHO","17":"ILLINOIS",
        "18":"INDIANA","19":"IOWA","20":"KANSAS","21":"KENTUCKY","22":"LOUISIANA",
        "23":"MAINE","24":"MARYLAND","25":"MASSACHUSETTS","26":"MICHIGAN","27":"MINNESOTA",
        "28":"MISSISSIPPI","29":"MISSOURI","30":"MONTANA","31":"NEBRASKA","32":"NEVADA",
        "33":"NEW HAMPSHIRE","34":"NEW JERSEY","35":"NEW MEXICO","36":"NEW YORK",
        "37":"NORTH CAROLINA","38":"NORTH DAKOTA","39":"OHIO","40":"OKLAHOMA",
        "41":"OREGON","42":"PENNSYLVANIA","44":"RHODE ISLAND","45":"SOUTH CAROLINA",
        "46":"SOUTH DAKOTA","47":"TENNESSEE","48":"TEXAS","49":"UTAH","50":"VERMONT",
        "51":"VIRGINIA","53":"WASHINGTON","54":"WEST VIRGINIA","55":"WISCONSIN","56":"WYOMING",
    }
    tiger["state_name"] = tiger["STATEFP"].map(state_fips)
    # Normalise county name: strip " County", " Parish", etc. to match FBI format
    tiger["county_name"] = (
        tiger["NAME"].str.upper()
        .str.replace(r"\s+(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|CITY AND BOROUGH)$",
                     "", regex=True)
        .str.strip()
    )
    return tiger[["state_name", "county_name", "fips"]]


def fetch_crime() -> pd.DataFrame:
    out_path = RAW / "crime" / "fbi_county_crime_2024.csv"
    if out_path.exists():
        log.info("FBI crime file already exists, loading from cache.")
        return pd.read_csv(out_path, dtype={"fips": str})

    zip_files = list((RAW / "crime").glob(CRIME_ZIP_GLOB))
    if not zip_files:
        log.warning(
            "No FBI crime zip found in %s. Download 'Offenses Known to Law Enforcement' "
            "from https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads "
            "and save the zip to data/raw/crime/",
            RAW / "crime",
        )
        return pd.DataFrame(columns=["fips", "crime_rate_per_100k"])

    zip_path = sorted(zip_files)[-1]  # use most recent if multiple
    log.info("Parsing FBI crime data from %s ...", zip_path.name)

    with zipfile.ZipFile(zip_path) as z:
        table10_names = [n for n in z.namelist() if "Table_10" in n and n.endswith(".xlsx")]
        if not table10_names:
            log.warning("Table 10 (county-level) not found in zip.")
            return pd.DataFrame(columns=["fips", "crime_rate_per_100k"])
        with z.open(table10_names[0]) as f:
            raw = pd.read_excel(f, header=None)

    # Row 4 is the header; data starts at row 5
    raw.columns = raw.iloc[4]
    data = raw.iloc[5:].copy().reset_index(drop=True)
    data.columns = [str(c).strip() for c in data.columns]

    # Identify the state, county, and offense count columns by position
    # Col 0 = State, col 2 = County, col 3 = Violent crime, col 8 = Property crime
    cols = list(data.columns)
    state_col, county_col = cols[0], cols[2]
    violent_col  = next((c for c in cols if "violent" in c.lower()), cols[3])
    property_col = next((c for c in cols if "property" in c.lower()), cols[8])

    data = data[[state_col, county_col, violent_col, property_col]].copy()
    data.columns = ["state_name", "county_name", "violent_crime", "property_crime"]

    # Forward-fill state name (blank in continuation rows)
    data["state_name"] = data["state_name"].replace("", np.nan).ffill()
    data = data.dropna(subset=["county_name"])
    data = data[data["county_name"].astype(str).str.strip() != ""]

    data["state_name"]   = data["state_name"].astype(str).str.upper().str.strip()
    data["county_name"]  = data["county_name"].astype(str).str.upper().str.strip()
    data["violent_crime"]  = pd.to_numeric(data["violent_crime"],  errors="coerce").fillna(0)
    data["property_crime"] = pd.to_numeric(data["property_crime"], errors="coerce").fillna(0)
    data["total_offenses"] = data["violent_crime"] + data["property_crime"]

    # Map state+county → FIPS
    xwalk = _build_county_fips_crosswalk()
    merged = data.merge(xwalk, on=["state_name", "county_name"], how="left")

    unmatched = merged["fips"].isna().sum()
    if unmatched:
        log.warning("%d rows could not be matched to a FIPS code.", unmatched)

    merged = merged.dropna(subset=["fips"])

    # Divide by ACS population to get rate per 100k
    acs_path = RAW / "census_acs" / "acs_demographics.csv"
    if acs_path.exists():
        pop = pd.read_csv(acs_path, dtype={"fips": str})[["fips", "total_pop"]]
        merged = merged.merge(pop, on="fips", how="left")
        merged["crime_rate_per_100k"] = (
            merged["total_offenses"] / merged["total_pop"] * 100_000
        )
    else:
        log.warning("ACS demographics not found; crime_rate_per_100k will be raw offense count.")
        merged["crime_rate_per_100k"] = merged["total_offenses"]

    result = merged[["fips", "crime_rate_per_100k"]].dropna()
    result.to_csv(out_path, index=False)
    log.info("FBI crime data saved → %s  (%d counties)", out_path, len(result))
    return result


# ---------------------------------------------------------------------------
# 5. Border zone flag (within 100 miles of US external boundary)
# ---------------------------------------------------------------------------

TIGER_COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
BORDER_ZONE_MILES = 100
METERS_PER_MILE = 1609.344


def _download_and_unzip(url: str, dest_dir: Path) -> Path:
    zip_path = dest_dir / Path(url).name
    if not zip_path.exists():
        log.info("Downloading %s ...", url)
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    shp_files = list(dest_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found after unzipping {zip_path}")
    return shp_files[0]


def fetch_border_zone() -> pd.DataFrame:
    out_path = RAW / "shapefiles" / "border_zone.csv"
    if out_path.exists():
        log.info("Border zone file already exists, loading from cache.")
        return pd.read_csv(out_path, dtype={"fips": str})

    shp_dir = RAW / "shapefiles"

    log.info("Fetching Census TIGER county shapefile ...")
    county_shp = _download_and_unzip(TIGER_COUNTIES_URL, shp_dir)
    counties = gpd.read_file(county_shp)
    counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]

    # Drop territories outside continental + AK/HI (non-US FIPS prefixes)
    # Keep FIPS state codes 01–56 (includes DC=11, excludes 57+)
    counties = counties[counties["STATEFP"].astype(int) <= 56]
    counties = counties.to_crs(epsg=5070)  # project to Conus Albers (meters)

    # Derive US boundary from the union of all county geometries — avoids a
    # second large shapefile download and is accurate enough for a 100-mile check.
    log.info("Deriving US border from county union ...")
    us_union = counties.geometry.unary_union
    border_line = us_union.boundary

    # Buffer border by 100 miles
    buffer_m = BORDER_ZONE_MILES * METERS_PER_MILE
    border_zone_poly = border_line.buffer(buffer_m)

    log.info("Computing border zone flag for %d counties ...", len(counties))
    counties["in_border_zone"] = counties.geometry.intersects(border_zone_poly).astype(int)

    result = counties[["fips", "in_border_zone"]].copy()
    result.to_csv(out_path, index=False)
    log.info("Border zone flags saved → %s", out_path)
    return result


# ---------------------------------------------------------------------------
# 6. Population density (from ACS total_pop + TIGER county land area)
# ---------------------------------------------------------------------------

def compute_population_density(acs: pd.DataFrame, shp_dir: Path) -> pd.Series:
    """
    Returns a Series indexed by fips with population density (persons/sq mile).
    Uses TIGER county land area (ALAND in square meters).
    """
    county_shp_candidates = list(shp_dir.glob("tl_2023_us_county*.shp"))
    if not county_shp_candidates:
        log.warning("County shapefile not found for density calculation; returning NaN.")
        return pd.Series(dtype=float, name="pop_density")

    area_df = gpd.read_file(county_shp_candidates[0])[["STATEFP", "COUNTYFP", "ALAND"]]
    area_df["fips"] = area_df["STATEFP"] + area_df["COUNTYFP"]
    area_df["land_area_sqmi"] = area_df["ALAND"] / 2_589_988.11  # m² → sq miles

    merged = acs[["fips", "total_pop"]].merge(area_df[["fips", "land_area_sqmi"]], on="fips", how="left")
    merged["pop_density"] = merged["total_pop"] / merged["land_area_sqmi"]
    return merged.set_index("fips")["pop_density"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    log.info("=== 1/5  Census ACS demographics ===")
    acs = fetch_census_acs()

    log.info("=== 2/5  USDA Rural-Urban Continuum Codes ===")
    rucc = fetch_rucc()

    log.info("=== 3/5  2024 election results ===")
    election = fetch_election()

    log.info("=== 4/5  FBI crime rates ===")
    crime = fetch_crime()

    log.info("=== 5/5  Border zone flag (TIGER shapefiles) ===")
    border = fetch_border_zone()

    # ---- Add population density ----------------------------------------
    log.info("Computing population density ...")
    shp_dir = RAW / "shapefiles"
    pop_density = compute_population_density(acs, shp_dir)
    acs["pop_density"] = acs["fips"].map(pop_density)

    # ---- Merge all sources on FIPS -------------------------------------
    log.info("Merging all static sources ...")
    static = (
        acs
        .merge(rucc, on="fips", how="left")
        .merge(election, on="fips", how="left")
        .merge(crime, on="fips", how="left")
        .merge(border, on="fips", how="left")
    )

    # ---- Document missing data ----------------------------------------
    log.info("Missing value summary:")
    missing = static.isnull().sum()
    missing_pct = (missing / len(static) * 100).round(1)
    miss_report = pd.DataFrame({"missing_n": missing, "missing_pct": missing_pct})
    miss_report = miss_report[miss_report["missing_n"] > 0]
    if not miss_report.empty:
        log.info("\n%s", miss_report.to_string())

    # ---- Imputation strategy -----------------------------------------
    # Static numeric features: median imputation (documented in README).
    # Categorical (urban_rural): mode imputation.
    # in_border_zone: 0 (conservative; unknown counties assumed not in zone).
    numeric_cols = static.select_dtypes(include=[np.number]).columns.difference(["fips"])
    medians = static[numeric_cols].median()
    static[numeric_cols] = static[numeric_cols].fillna(medians)

    if "urban_rural" in static.columns:
        mode_val = static["urban_rural"].mode().iloc[0]
        static["urban_rural"] = static["urban_rural"].fillna(mode_val)

    if "in_border_zone" in static.columns:
        static["in_border_zone"] = static["in_border_zone"].fillna(0).astype(int)

    # ---- Save --------------------------------------------------------
    out_path = PROCESSED / "static_features.csv"
    static.to_csv(out_path, index=False)
    log.info("Static features saved → %s  (%d counties, %d columns)", out_path, len(static), len(static.columns))

    # ---- Quick sanity checks ----------------------------------------
    log.info("Sample output:")
    log.info("\n%s", static.head(3).to_string())


if __name__ == "__main__":
    main()
