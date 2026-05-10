#!/usr/bin/env python3
"""
dashboard/app.py  —  ICE Enforcement Media Risk
Streamlit dashboard: county-level risk heatmap + model interpretation.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICE Enforcement Media Risk",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).parent.parent  # repo root

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "66": "GU", "72": "PR", "78": "VI",
}

FEAT_LABELS = {
    "pct_hispanic":        "% Hispanic",
    "pct_black":           "% Black",
    "pct_asian":           "% Asian",
    "pct_foreign_born":    "% Foreign-born",
    "median_hh_income":    "Median HH income",
    "pct_poverty":         "% Poverty",
    "pop_density":         "Pop. density (per mi²)",
    "rep_vote_share_2024": "Rep. vote share 2024",
    "crime_rate_per_100k": "Crime rate per 100k",
    "lag_1m":              "Lag 1 month",
    "lag_3m":              "Lag 3 months",
    "lag_6m":              "Lag 6 months",
    "in_border_zone":      "Border zone (100 mi)",
    "urban_ruralsuburban": "Suburban (vs rural)",
    "urban_ruralurban":    "Urban (vs rural)",
}

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading county boundaries …")
def load_geojson() -> dict:
    url = ("https://raw.githubusercontent.com/plotly/datasets/"
           "master/geojson-counties-fips.json")
    with urlopen(url) as r:
        return json.load(r)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(
        BASE / "model" / "model_artifacts" / "predictions_all.csv",
        dtype={"fips": str},
    )
    df["fips"] = df["fips"].str.zfill(5)
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["month_label"] = df["year_month"].dt.strftime("%b %Y")
    return df


@st.cache_data(show_spinner=False)
def load_static() -> pd.DataFrame:
    df = pd.read_csv(
        BASE / "data" / "processed" / "static_features.csv",
        dtype={"fips": str},
    )
    df["fips"] = df["fips"].str.zfill(5)
    return df


@st.cache_data(show_spinner=False)
def load_fixed_effects() -> pd.DataFrame:
    return pd.read_csv(BASE / "model" / "model_artifacts" / "fixed_effects.csv")


@st.cache_data(show_spinner=False)
def load_calibration() -> pd.DataFrame:
    return pd.read_csv(BASE / "model" / "model_artifacts" / "calibration_data.csv")


@st.cache_data(show_spinner=False)
def load_threshold() -> dict:
    return json.loads(
        (BASE / "model" / "model_artifacts" / "threshold.json").read_text()
    )


@st.cache_data(show_spinner=False)
def extract_county_names(_geojson: dict) -> pd.DataFrame:
    rows = [
        {
            "fips":        feat["id"],
            "county_name": feat["properties"].get("NAME", ""),
            "lsad":        feat["properties"].get("LSAD", ""),
            "state_fips":  feat["properties"].get("STATE", ""),
        }
        for feat in _geojson["features"]
    ]
    df = pd.DataFrame(rows)
    df["state_abbr"] = df["state_fips"].map(STATE_FIPS)
    df["display_name"] = (
        df["county_name"] + " " + df["lsad"] + ", " + df["state_abbr"].fillna("")
    ).str.strip()
    return df


# ── Bootstrap ─────────────────────────────────────────────────────────────────
geojson  = load_geojson()
preds    = load_predictions()
static   = load_static()
fe       = load_fixed_effects()
cal      = load_calibration()
thresh   = load_threshold()
names_df = extract_county_names(geojson)

preds = preds.merge(
    names_df[["fips", "display_name", "county_name", "state_abbr"]],
    on="fips", how="left",
)
preds = preds.merge(static, on="fips", how="left")

month_options = sorted(preds["year_month"].unique())
month_labels  = {m: pd.Timestamp(m).strftime("%b %Y") for m in month_options}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗺️ ICE Enforcement Media Risk Model")
st.warning(
    "**Media attention model.** This tool predicts where US counties are likely to appear "
    "in immigration enforcement news coverage — *not* the actual intensity of ICE enforcement. "
    "Coverage is systematically lower in Republican-leaning and rural areas due to media "
    "geography, not because enforcement is less frequent there.",
    icon="⚠️",
)

tab_map, tab_model, tab_about = st.tabs(["🗺️ Risk Map", "📊 Model Details", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:

    # Controls row
    c1, c2, c3 = st.columns([4, 1.4, 1.4])
    with c1:
        selected_ts = st.select_slider(
            "Month",
            options=month_options,
            value=month_options[-1],
            format_func=lambda x: month_labels[x],
        )
    with c2:
        hide_low = st.checkbox(
            "Hide low-confidence",
            value=False,
            help="Counties with < 3 positive months in training data",
        )
    with c3:
        show_flagged = st.checkbox(
            "Flagged only",
            value=False,
            help="Show only counties predicted as high-risk this month",
        )

    # Filter to selected month
    month_df = preds[preds["year_month"] == selected_ts].copy()
    if hide_low:
        month_df = month_df[month_df["low_confidence"] == 0]
    if show_flagged:
        month_df = month_df[month_df["pred_class"] == 1]

    n_flagged = int((preds[preds["year_month"] == selected_ts]["pred_class"] == 1).sum())
    st.caption(
        f"**{month_labels[selected_ts]}** — "
        f"{n_flagged:,} counties flagged ({n_flagged/3222:.1%} of all counties)"
    )

    # Choropleth
    fig_map = px.choropleth(
        month_df,
        geojson=geojson,
        locations="fips",
        color="pred_prob",
        color_continuous_scale="YlOrRd",
        range_color=(0.0, 0.85),
        scope="usa",
        labels={"pred_prob": "Risk Score"},
        hover_name="display_name",
        hover_data={
            "pred_prob":      ":.1%",
            "pred_class":     True,
            "low_confidence": True,
            "fips":           False,
        },
    )
    fig_map.update_layout(
        margin=dict(r=0, t=0, l=0, b=0),
        coloraxis_colorbar=dict(
            title="Risk<br>Score",
            tickformat=".0%",
            len=0.6,
            thickness=14,
        ),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        height=530,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── County detail ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("County Detail")

    county_opts = (
        preds[preds["year_month"] == selected_ts]
        .sort_values("pred_prob", ascending=False)
        .dropna(subset=["display_name"])
        [["fips", "display_name", "pred_prob"]]
        .drop_duplicates("fips")
    )
    county_opts["label"] = (
        county_opts["display_name"]
        + "  —  "
        + (county_opts["pred_prob"] * 100).round(1).astype(str)
        + "%"
    )
    fips_list = county_opts["fips"].tolist()

    sel_fips = st.selectbox(
        "Select a county (sorted by risk score ↓)",
        options=fips_list,
        format_func=lambda f: county_opts.set_index("fips").loc[f, "label"]
        if f in county_opts["fips"].values else f,
    )

    if sel_fips:
        row     = month_df[month_df["fips"] == sel_fips]
        if row.empty:
            row = preds[(preds["fips"] == sel_fips) & (preds["year_month"] == selected_ts)]
        if not row.empty:
            row = row.iloc[0]
            history = preds[preds["fips"] == sel_fips].sort_values("year_month")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score",       f"{row['pred_prob']:.1%}")
            m2.metric("Flagged?",         "Yes ⚠️" if row["pred_class"] else "No ✓")
            m3.metric("Low confidence",   "Yes" if row["low_confidence"] else "No")
            m4.metric("Actual coverage",  "Yes" if row["outcome"] else "No")

            col_trend, col_feats = st.columns([3, 2])

            with col_trend:
                fig_t = go.Figure()
                fig_t.add_bar(
                    x=history["month_label"],
                    y=history["pred_prob"],
                    name="Predicted risk",
                    marker_color="#f97316",
                    opacity=0.75,
                )
                fig_t.add_scatter(
                    x=history["month_label"],
                    y=history["outcome"].astype(float),
                    name="Actual coverage",
                    mode="markers",
                    marker=dict(color="#dc2626", size=9, symbol="circle"),
                )
                fig_t.add_hline(
                    y=thresh["threshold"],
                    line_dash="dash",
                    line_color="#6b7280",
                    annotation_text=f"Threshold {thresh['threshold']:.2f}",
                    annotation_position="top right",
                )
                fig_t.update_layout(
                    title=row.get("display_name", sel_fips),
                    xaxis_title=None,
                    yaxis_title="Probability",
                    yaxis_tickformat=".0%",
                    height=310,
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(t=45, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_t, use_container_width=True)

            with col_feats:
                st.markdown("**Key features**")
                feat_display = {
                    "% Hispanic":       f"{row.get('pct_hispanic', float('nan')):.1f}%",
                    "% Foreign-born":   f"{row.get('pct_foreign_born', float('nan')):.1f}%",
                    "% Poverty":        f"{row.get('pct_poverty', float('nan')):.1f}%",
                    "Rep. vote 2024":   f"{float(row.get('rep_vote_share_2024', 0)) * 100:.1f}%",
                    "Urban/rural":      str(row.get("urban_rural", "—")),
                    "Border zone":      "Yes" if row.get("in_border_zone") else "No",
                    "Pop. density":     f"{row.get('pop_density', float('nan')):.0f} / mi²",
                }
                st.dataframe(
                    pd.DataFrame(feat_display.items(), columns=["Feature", "Value"]),
                    hide_index=True,
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:

    st.subheader("Validation Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-PR",    f"{thresh['auc_pr']:.3f}",
              help="Area under Precision-Recall curve. Baseline (random) ≈ 0.15")
    m2.metric("F2 Score",  f"{thresh['f_beta_at_threshold']:.3f}",
              help="F-beta (β=2): weights recall 2× over precision")
    m3.metric("Recall",    f"{thresh['recall_at_threshold']:.1%}",
              help="Share of true positives captured at operating threshold")
    m4.metric("Precision", f"{thresh['precision_at_threshold']:.1%}",
              help="Share of flagged counties that are true positives")
    m5.metric("Threshold", f"{thresh['threshold']:.3f}",
              help=f"Operating threshold (F2-optimal). Prevalence = {thresh['empirical_prevalence']:.1%}")

    st.divider()
    col_coef, col_cal = st.columns([3, 2])

    # ── Coefficient plot ───────────────────────────────────────────────────────
    with col_coef:
        st.subheader("Fixed-Effect Coefficients")
        st.caption(
            "Logit-scale coefficients on standardised features. "
            "Filled = p < 0.05 | Open = not significant."
        )
        fe_plot = fe[fe["variable"] != "(Intercept)"].copy()
        fe_plot["label"]  = fe_plot["variable"].map(FEAT_LABELS).fillna(fe_plot["variable"])
        fe_plot["sig"]    = fe_plot["p_value"] < 0.05
        fe_plot["color"]  = fe_plot["estimate"].apply(
            lambda x: "#16a34a" if x > 0 else "#dc2626"
        )
        fe_plot = fe_plot.sort_values("estimate")

        fig_coef = go.Figure()
        fig_coef.add_scatter(
            x=fe_plot["estimate"],
            y=fe_plot["label"],
            mode="markers",
            marker=dict(
                color=fe_plot["color"].tolist(),
                size=fe_plot["sig"].map({True: 11, False: 7}).tolist(),
                symbol=fe_plot["sig"].map(
                    {True: "circle", False: "circle-open"}
                ).tolist(),
                line=dict(
                    color=fe_plot["color"].tolist(),
                    width=2,
                ),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(fe_plot["ci_upper"] - fe_plot["estimate"]).tolist(),
                arrayminus=(fe_plot["estimate"] - fe_plot["ci_lower"]).tolist(),
                color="rgba(0,0,0,0.25)",
                thickness=1.5,
            ),
            text=fe_plot["p_value"].apply(
                lambda p: f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
            ),
            hovertemplate="%{y}<br>β = %{x:.3f}<br>%{text}<extra></extra>",
        )
        fig_coef.add_vline(x=0, line_color="#9ca3af", line_dash="dot", line_width=1)
        fig_coef.update_layout(
            xaxis_title="Coefficient (log-odds, standardised features)",
            yaxis_title=None,
            height=440,
            margin=dict(l=10, r=20, t=10, b=40),
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#f3f4f6", zeroline=False),
            yaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # ── Calibration plot ───────────────────────────────────────────────────────
    with col_cal:
        st.subheader("Calibration")
        st.caption("Does the model's predicted probability match observed frequency?")

        fig_cal = go.Figure()
        fig_cal.add_scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#9ca3af", dash="dash", width=1),
            name="Perfect calibration",
            hoverinfo="skip",
        )
        fig_cal.add_scatter(
            x=cal["mean_pred"],
            y=cal["mean_obs"],
            mode="markers+lines",
            marker=dict(
                color="#2563eb",
                size=cal["n"].apply(lambda n: 4 + 14 * (n / cal["n"].max())).tolist(),
                opacity=0.8,
            ),
            line=dict(color="#2563eb", width=1.5),
            name="Model",
            text=cal["n"].apply(lambda n: f"n = {n:,}"),
            hovertemplate=(
                "Predicted: %{x:.1%}<br>"
                "Observed: %{y:.1%}<br>"
                "%{text}<extra></extra>"
            ),
        )
        fig_cal.update_layout(
            xaxis=dict(
                title="Mean predicted probability",
                tickformat=".0%",
                range=[0, 1],
                gridcolor="#f3f4f6",
            ),
            yaxis=dict(
                title="Observed frequency",
                tickformat=".0%",
                range=[0, 1],
                gridcolor="#f3f4f6",
            ),
            height=380,
            margin=dict(t=10, b=40, l=10, r=10),
            legend=dict(orientation="h", y=-0.22),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption(
            "Marker size ∝ number of validation counties in that bin. "
            "The model overestimates in the 0.2–0.8 range — "
            "a common pattern with logit on imbalanced data."
        )

    # ── Model spec summary ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Specification")
    st.code(
        "glmer(outcome ~ pct_hispanic + pct_black + pct_asian + pct_foreign_born\n"
        "              + median_hh_income + pct_poverty + pop_density\n"
        "              + rep_vote_share_2024 + crime_rate_per_100k\n"
        "              + lag_1m + lag_3m + lag_6m\n"
        "              + in_border_zone + urban_rural\n"
        "              + offset(log1p(ice_arrest_rate_state))\n"
        "              + (1 | fips),\n"
        "       family = binomial(link = 'logit'),   # logit fallback from cloglog\n"
        "       data   = train_scaled)               # features standardised on train",
        language="r",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## What this model does

This model predicts whether a US county will appear in **immigration enforcement
media coverage** in the following month, using GDELT Global Knowledge Graph (GKG)
news data as the dependent variable.

It is a **media attention model**, not an enforcement intensity model. The distinction matters:

- A county with active ICE operations but no local press corps may score **low**.
- A county with strong advocacy organizations and a Spanish-language newspaper
  may score **high** even in quiet months.

---

## Methodology

1. **Dependent variable** — county × month binary indicator derived from GDELT GKG v1
   (Jan 2025 – Apr 2026). Articles are filtered to those mentioning `IMMIGRATION` and at
   least one enforcement-specific theme (`DISCRIMINATION_IMMIGRATION_ANTIIMMIGRANT`,
   `TAX_FNCACT_BORDER_PATROL_AGENT`, `WB_2491_BORDER_SECURITY`, etc.).
   Deduplicated by CAMEO event ID — 10 outlets covering the same ICE operation
   count as **one event**, not ten.

2. **Features** — demographic (ACS), political (2024 presidential returns),
   geographic (border zone, urban/rural), and temporal momentum (rolling
   1/3/6-month mention counts). All scaled to unit variance on the training set.

3. **ICE offset** — state-level ICE arrest rate (arrests per 100k, from the
   Deportation Data Project FY23–26) enters as `offset(log1p(rate))`.
   This anchors predictions to observed state enforcement intensity with a
   fixed coefficient of 1, rather than estimating it freely.

4. **Random intercept** per county (3,222 counties) absorbs persistent unexplained
   heterogeneity in local media ecosystems.

5. **Temporal split** — train: 2025-01 → 2025-09, validate: 2025-10 → 2026-03.
   The model predicts coverage at **t+1** given features at **t**.

6. **Link function** — logit (cloglog was the specification but was numerically
   unstable with the offset on rare-event data; predictions are virtually
   identical at 15% prevalence).

---

## Key findings

| Feature | Direction | Why |
|---|---|---|
| **Rep. vote share 2024** | ↓ strongly (β = −0.85) | Media geography: fewer news bureaus in red areas |
| **Urban** | ↑ strongly (β = +0.79) | Urban areas generate more coverage per operation |
| **Border zone** | ↑ (β = +0.70) | ~2× higher baseline log-odds |
| **% Foreign-born** | ↑ (β = +0.13) | Established communities → more reported incidents |
| **% Hispanic** | ↑ (β = +0.11) | Same mechanism |
| **% Poverty** | ↓ (β = −0.42) | Wealthier immigrant enclaves are more visible to media |
| **Lag 1m / 3m** | ↑ | News momentum — prior coverage predicts next month |
| **Crime rate, income** | ≈ 0 | No meaningful signal |

---

## Known limitations

- **Not a law-enforcement dataset.** GDELT captures media coverage, not ground-truth operations.
- **Republican-area blind spot.** The model underestimates risk in conservative, rural areas
  where national media coverage is sparse.
- **Low-confidence counties** (< 3 positive months in training) have unreliable predictions —
  toggleable on the map.
- **Static features are 2023 vintage.** Demographic shifts are not captured.
- **Media proxy bias.** High-profile advocacy events and protests generate coverage even
  in months with low actual enforcement.

---

## Data sources

| Source | Variable | Vintage |
|---|---|---|
| [GDELT GKG v1](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/) | Dependent variable | Jan 2025 – Apr 2026 |
| [Deportation Data Project](https://deportationdata.org) | ICE arrest rate offset | FY2023–2026 |
| [Census ACS](https://www.census.gov/programs-surveys/acs/) | Demographics | 2023 5-yr |
| [USDA RUCC](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/) | Urban/rural | 2023 |
| [MIT Election Lab](https://electionlab.mit.edu/data) | 2024 presidential returns | 2024 |
| [FBI UCR/NIBRS](https://cde.ucr.cjis.gov/) | Crime rate | 2022 |
| [Census TIGER](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | County shapefiles | 2023 |

---

*Built with Python · R · Streamlit · Plotly*
""")
