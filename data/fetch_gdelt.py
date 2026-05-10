#!/usr/bin/env python3
"""
fetch_gdelt.py

Downloads GDELT GKG v1 daily files for Jan 2025 – Apr 2026, filters for
DEPORTATION theme, extracts US county-level locations, and aggregates to a
binary county × month indicator.

Output: data/processed/gdelt_county_month.csv
  fips, year_month, mention_count, has_mention (binary)

Usage:
  python3 data/fetch_gdelt.py

The script is resumable: already-processed dates are recorded in
data/processed/gdelt_checkpoint.txt and skipped on re-run.
"""

import io
import hashlib
import zipfile
import logging
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE  = date(2025, 1, 1)
END_DATE    = date(2026, 4, 30)

GDELT_URL   = "http://data.gdeltproject.org/gkg/{}.gkg.csv.zip"
# Articles must contain IMMIGRATION plus at least one enforcement-specific theme.
# "DEPORTATION" is not a GDELT theme — these are the actual taxonomy equivalents.
THEME_REQUIRED  = "IMMIGRATION"
THEME_ENFORCE   = {
    "DISCRIMINATION_IMMIGRATION_ANTIIMMIGRANT",
    "TAX_FNCACT_IMMIGRATION_OFFICER",
    "TAX_FNCACT_BORDER_PATROL_AGENT",
    "WB_2491_BORDER_SECURITY",
    "UNREST_CLOSINGBORDER",
}

# DC/White House locations dominate political-news geocoding; exclude them
EXCLUDE_FULLNAMES = {
    "WASHINGTON, WASHINGTON, UNITED STATES",
    "WHITE HOUSE, DISTRICT OF COLUMBIA, UNITED STATES",
    "DISTRICT OF COLUMBIA, UNITED STATES",
    "WASHINGTON MONUMENT, DISTRICT OF COLUMBIA, UNITED STATES",
    "CAPITOL HILL, DISTRICT OF COLUMBIA, UNITED STATES",
}

PROCESSED    = Path("data/processed")
CHECKPOINT   = PROCESSED / "gdelt_checkpoint.txt"
OUT_PATH     = PROCESSED / "gdelt_county_month.csv"
EVENTS_LOG   = PROCESSED / "gdelt_events_log.csv"   # raw (ym, event_id, fips) for resumability
TIGER_SHP    = Path("data/raw/shapefiles/tl_2023_us_county.shp")

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-5s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State abbreviation → FIPS (2-digit)
# ---------------------------------------------------------------------------
STATE_ABBR_TO_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09",
    "DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17",
    "IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24",
    "MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31",
    "NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38",
    "OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46",
    "TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
    "WI":"55","WY":"56","GU":"66","PR":"72","VI":"78",
}

# ADM1CODE in GDELT looks like "USIL", "USTX", etc.
def adm1_to_state_fips(adm1: str) -> str:
    """Convert GDELT ADM1 code (e.g. 'USIL') to 2-digit state FIPS."""
    abbr = adm1[2:] if adm1.startswith("US") else ""
    return STATE_ABBR_TO_FIPS.get(abbr, "")


# ---------------------------------------------------------------------------
# Build crosswalks from TIGER shapefile + Census gazetteer
# ---------------------------------------------------------------------------
def build_crosswalk() -> pd.DataFrame:
    """county_name (upper) + state_fips → 5-digit county FIPS."""
    log.info("Building county FIPS crosswalk from TIGER shapefile ...")
    tiger = gpd.read_file(TIGER_SHP)[["STATEFP", "COUNTYFP", "NAME"]]
    tiger["fips"]         = tiger["STATEFP"] + tiger["COUNTYFP"]
    tiger["county_upper"] = tiger["NAME"].str.upper().str.strip()
    tiger["state_fips"]   = tiger["STATEFP"]
    return tiger[["county_upper", "state_fips", "fips"]]


def build_spatial_index():
    """
    Load TIGER county polygons and build a spatial index for
    point-in-polygon lookups. Returns (GeoDataFrame, sindex).
    Used as fallback when county name parsing fails.
    """
    log.info("Building county spatial index from TIGER shapefile ...")
    counties = gpd.read_file(TIGER_SHP)[["STATEFP", "COUNTYFP", "geometry"]]
    counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]
    counties = counties.to_crs("EPSG:4326")
    return counties


# ---------------------------------------------------------------------------
# Parse GDELT LOCATIONS field
# Each entry: GEO_TYPE#FULLNAME#COUNTRYCODE#ADM1CODE#LAT#LON#FEATUREID
# We want GEO_TYPE=3, COUNTRYCODE=US
# ---------------------------------------------------------------------------
def parse_locations(loc_str: str) -> list:
    """Return list of dicts for US city/county-level locations."""
    results = []
    if not loc_str or pd.isna(loc_str):
        return results
    for part in str(loc_str).split(";"):
        fields = part.split("#")
        if len(fields) < 6:
            continue
        geo_type = fields[0].strip()
        fullname = fields[1].strip().upper()
        country  = fields[2].strip()
        adm1     = fields[3].strip()
        try:
            lat = float(fields[4])
            lon = float(fields[5])
        except (ValueError, IndexError):
            lat = lon = None
        if geo_type == "3" and country == "US":
            if fullname in EXCLUDE_FULLNAMES:
                continue
            results.append({"fullname": fullname, "adm1": adm1,
                            "lat": lat, "lon": lon})
    return results


def fullname_to_county_state(fullname: str):
    """
    Try to extract (county_name, state_fips) from a GDELT fullname string.
    Examples:
      'COOK COUNTY, ILLINOIS, UNITED STATES'  → ('COOK', '17')
      'DALLAS COUNTY, TEXAS, UNITED STATES'   → ('DALLAS', '48')
    Returns (None, None) if pattern not matched.
    """
    parts = [p.strip() for p in fullname.split(",")]
    # Need at least [county/city, state, country]
    if len(parts) < 3:
        return None, None
    # Last part should be "UNITED STATES"
    if "UNITED STATES" not in parts[-1]:
        return None, None
    state_name_part = parts[-2].strip()
    first_part      = parts[0].strip()

    # Find state FIPS from state name (map full state name → abbr → fips)
    STATE_NAME_TO_ABBR = {
        "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR",
        "CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT","DELAWARE":"DE",
        "FLORIDA":"FL","GEORGIA":"GA","HAWAII":"HI","IDAHO":"ID",
        "ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA","KANSAS":"KS",
        "KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD",
        "MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS",
        "MISSOURI":"MO","MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV",
        "NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ","NEW MEXICO":"NM","NEW YORK":"NY",
        "NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK",
        "OREGON":"OR","PENNSYLVANIA":"PA","RHODE ISLAND":"RI",
        "SOUTH CAROLINA":"SC","SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX",
        "UTAH":"UT","VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA",
        "WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY",
    }
    abbr       = STATE_NAME_TO_ABBR.get(state_name_part)
    state_fips = STATE_ABBR_TO_FIPS.get(abbr, "") if abbr else ""
    if not state_fips:
        return None, None

    # Strip " COUNTY" suffix if present
    county_name = first_part
    for suffix in [" COUNTY", " PARISH", " BOROUGH", " CENSUS AREA",
                   " MUNICIPALITY", " CITY AND BOROUGH"]:
        if county_name.endswith(suffix):
            county_name = county_name[: -len(suffix)].strip()
            break

    return county_name, state_fips


# ---------------------------------------------------------------------------
# CAMEO event ID parser
# ---------------------------------------------------------------------------
def parse_cameo_ids(cameo_str) -> list:
    """
    Parse the CAMEOEVENTIDS field (comma-separated integers) into a list.
    Returns a list of ID strings, or [] if empty/null.
    """
    if pd.isna(cameo_str) or str(cameo_str).strip() == "":
        return []
    return [cid.strip() for cid in str(cameo_str).split(",") if cid.strip()]


# ---------------------------------------------------------------------------
# Download and process one day
# ---------------------------------------------------------------------------
def process_day(day: date, crosswalk: pd.DataFrame,
                counties_gdf) -> list:
    """
    Download the GKG file for `day`, filter for immigration enforcement themes,
    extract US county locations, and return a list of (event_id, fips) tuples.

    Deduplication: articles sharing a CAMEO event ID are reporting the same
    case — each (event_id, fips) pair is unique so the same event covered by
    multiple outlets counts only once per county.
    When CAMEOEVENTIDS is absent a synthetic ID (day + URL hash) is used so
    the article still contributes exactly one event per county.

    Returns [] on any failure.
    """
    url = GDELT_URL.format(day.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            log.debug("No file for %s (404)", day)
            return []
        resp.raise_for_status()
    except Exception as e:
        log.warning("Failed to fetch %s: %s", day, e)
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(
                    f, sep="\t", dtype=str, header=0,
                    names=["DATE","NUMARTS","COUNTS","THEMES","LOCATIONS",
                           "PERSONS","ORGANIZATIONS","TONE","CAMEOEVENTIDS",
                           "SOURCES","SOURCEURLS"],
                    on_bad_lines="skip",
                )
    except Exception as e:
        log.warning("Failed to parse %s: %s", day, e)
        return []

    # Filter: must have IMMIGRATION + at least one enforcement theme
    def is_enforcement(themes):
        if pd.isna(themes):
            return False
        tokens = set(str(themes).split(";"))
        return THEME_REQUIRED in tokens and bool(tokens & THEME_ENFORCE)

    df = df[df["THEMES"].apply(is_enforcement)]
    if df.empty:
        return []

    pairs = []  # (event_id, fips) tuples — deduplicated in main()

    for idx, row in df.iterrows():
        # ── CAMEO event IDs ────────────────────────────────────────────────
        event_ids = parse_cameo_ids(row.get("CAMEOEVENTIDS", ""))
        if not event_ids:
            # Synthetic article-unique ID — stable via SHA-1 of source URLs
            src = str(row.get("SOURCEURLS", "")) + str(row.get("SOURCES", ""))
            synth = hashlib.sha1(
                src.encode("utf-8", errors="replace")
            ).hexdigest()[:10]
            event_ids = [f"S{day.strftime('%Y%m%d')}_{synth}"]

        # ── Resolve county FIPS for this article's locations ───────────────
        locs = parse_locations(str(row.get("LOCATIONS", "")))
        article_fips = set()
        unresolved   = []

        for loc in locs:
            fips = _resolve_fips_name(loc["fullname"], loc["adm1"], crosswalk)
            if fips:
                article_fips.add(fips)
            elif loc["lat"] is not None and loc["lon"] is not None:
                unresolved.append((loc["lat"], loc["lon"]))

        if unresolved:
            article_fips.update(_spatial_lookup(unresolved, counties_gdf))

        # ── Emit (event_id, fips) for every combination ────────────────────
        for event_id in event_ids:
            for fips in article_fips:
                pairs.append((event_id, fips))

    return pairs


def _resolve_fips_name(fullname: str, adm1: str,
                       crosswalk: pd.DataFrame):
    """County name → FIPS via TIGER crosswalk. Returns None if not found."""
    county_name, state_fips = fullname_to_county_state(fullname)
    if not county_name or not state_fips:
        return None
    hits = crosswalk[
        (crosswalk["county_upper"] == county_name) &
        (crosswalk["state_fips"]   == state_fips)
    ]
    return hits["fips"].iloc[0] if not hits.empty else None


def _spatial_lookup(points: list, counties_gdf) -> list:
    """
    Batch point-in-polygon lookup. points = list of (lat, lon).
    Returns list of FIPS strings for matched points.
    """
    from shapely.geometry import Point
    pts_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(lon, lat) for lat, lon in points]},
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts_gdf, counties_gdf[["fips", "geometry"]],
                       how="left", predicate="within")
    return joined["fips"].dropna().tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (already-processed dates)
    processed_dates = set()
    if CHECKPOINT.exists():
        processed_dates = set(CHECKPOINT.read_text().splitlines())

    crosswalk    = build_crosswalk()
    counties_gdf = build_spatial_index()

    # monthly_events: year_month → set of (event_id, fips) — globally deduplicated
    monthly_events: dict = defaultdict(set)

    # Reload previously accumulated events for true cross-day deduplication
    log_exists = EVENTS_LOG.exists() and EVENTS_LOG.stat().st_size > 0
    if log_exists:
        log_df = pd.read_csv(EVENTS_LOG, dtype={"fips": str, "event_id": str})
        for _, r in log_df.iterrows():
            monthly_events[r["year_month"]].add((r["event_id"], r["fips"]))
        log.info("Resuming: loaded %d event-county pairs from events log",
                 len(log_df))

    # Open events log for appending
    events_fh = open(EVENTS_LOG, "a", newline="")
    if not log_exists:
        events_fh.write("year_month,event_id,fips\n")

    day    = START_DATE
    n_days = (END_DATE - START_DATE).days + 1
    n_done = 0

    try:
        while day <= END_DATE:
            day_str = day.strftime("%Y-%m-%d")
            ym      = day.strftime("%Y-%m")

            if day_str not in processed_dates:
                pairs = process_day(day, crosswalk, counties_gdf)
                for event_id, fips in pairs:
                    key = (event_id, fips)
                    if key not in monthly_events[ym]:
                        monthly_events[ym].add(key)
                        events_fh.write(f"{ym},{event_id},{fips}\n")
                events_fh.flush()

                with open(CHECKPOINT, "a") as f:
                    f.write(day_str + "\n")
                processed_dates.add(day_str)

                n_done += 1
                if n_done % 30 == 0:
                    log.info("Progress: %d / %d days processed", n_done, n_days)
                    _save(monthly_events)

            day += timedelta(days=1)
    finally:
        events_fh.close()

    _save(monthly_events)
    log.info("Done. Output → %s", OUT_PATH)


def _save(monthly_events: dict):
    """Aggregate unique (event_id, fips) pairs → mention_count per county-month."""
    rows = []
    for ym, pair_set in monthly_events.items():
        county_counts: dict = defaultdict(int)
        for _event_id, fips in pair_set:
            county_counts[fips] += 1
        for fips, cnt in county_counts.items():
            rows.append({"year_month": ym, "fips": fips, "mention_count": cnt})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["has_mention"] = (df["mention_count"] > 0).astype(int)
    df = df.sort_values(["year_month", "fips"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
