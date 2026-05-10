#!/usr/bin/env python3
"""
scripts/generate_map.py

Generates two files for GitHub Pages:
  docs/index.html   — interactive Plotly choropleth with month slider
  docs/preview.png  — static snapshot of the latest month (for README embed)

This script renders the Stage 2 model's in-sample predictions (training +
validation period, Jan 2025 – Mar 2026).  Forward predictions beyond the
panel are added separately by forward_predict.py, which also injects the
county risk lookup panel into docs/index.html.

The animated map uses efficient frame updates: county boundaries are stored
once; only the color values (pred_prob) are swapped per frame.
This keeps the HTML under ~10 MB instead of ~50 MB.

Pipeline order:
    python scripts/generate_map.py       # produces docs/index.html
    python scripts/forward_predict.py    # injects lookup panel + predictions.json

Requires:
    pip install plotly pandas
    pip install kaleido==0.2.1   # optional, only needed for preview.png
                                 # must be 0.2.x — kaleido 1.x is incompatible with plotly 5
"""

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.graph_objects as go

BASE = Path(__file__).parent.parent
DOCS = BASE / "docs"
DOCS.mkdir(exist_ok=True)

STATE_FIPS = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading predictions …")
preds = pd.read_csv(
    BASE / "model" / "model_artifacts" / "predictions_all.csv",
    dtype={"fips": str},
)
preds["fips"]        = preds["fips"].str.zfill(5)
preds["year_month"]  = pd.to_datetime(preds["year_month"])
preds["month_label"] = preds["year_month"].dt.strftime("%b %Y")
preds["pred_prob"]   = preds["pred_prob"].round(3)

# ── Load county GeoJSON ───────────────────────────────────────────────────────
print("Loading county boundaries …")
url = ("https://raw.githubusercontent.com/plotly/datasets/"
       "master/geojson-counties-fips.json")
with urlopen(url) as r:
    counties = json.load(r)

name_map = {
    f["id"]: (
        f["properties"].get("NAME", "")
        + " "
        + f["properties"].get("LSAD", "")
        + ", "
        + STATE_FIPS.get(f["properties"].get("STATE", ""), "")
    ).strip()
    for f in counties["features"]
}

# ── Build month-ordered pivot: rows = counties (sorted), cols = months ────────
month_order = sorted(preds["year_month"].unique())
month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in month_order]

# Ensure every month has all FIPS (fill missing with 0)
all_fips = sorted(preds["fips"].unique())
pivot = (
    preds.pivot_table(
        index="fips", columns="year_month", values="pred_prob", aggfunc="first"
    )
    .reindex(index=all_fips, columns=month_order)
    .fillna(0)
)

county_names  = [name_map.get(f, f) for f in all_fips]
flagged_lookup = (
    preds.pivot_table(
        index="fips", columns="year_month", values="pred_class", aggfunc="first"
    )
    .reindex(index=all_fips, columns=month_order)
    .fillna(0)
    .astype(int)
)

# ── Build figure — geometry stored once, only z updates per frame ─────────────
print("Building map …")

first_z    = pivot[month_order[0]].tolist()
first_flag = flagged_lookup[month_order[0]].tolist()

hover_text_base = [
    f"<b>{name}</b><br>FIPS: {fips}"
    for name, fips in zip(county_names, all_fips)
]

def make_hover(z_vals, flag_vals):
    return [
        f"{base}<br>Risk Score: {z:.1%}<br>Flagged: {'Yes ⚠️' if fl else 'No'}"
        for base, z, fl in zip(hover_text_base, z_vals, flag_vals)
    ]

choropleth = go.Choropleth(
    geojson=counties,
    locations=all_fips,
    z=first_z,
    text=make_hover(first_z, first_flag),
    hoverinfo="text",
    colorscale="YlOrRd",
    zmin=0,
    zmax=0.85,
    colorbar=dict(
        title="Risk<br>Score",
        tickformat=".0%",
        len=0.6,
        thickness=14,
    ),
    marker_line_width=0.2,
    marker_line_color="white",
)

# One frame per month — only z and text update, geometry stays fixed
frames = [
    go.Frame(
        data=[go.Choropleth(
            z=pivot[m].tolist(),
            text=make_hover(pivot[m].tolist(), flagged_lookup[m].tolist()),
        )],
        name=label,
    )
    for m, label in zip(month_order, month_labels)
]

# Slider steps
sliders = [dict(
    active=0,
    steps=[
        dict(
            label=label,
            method="animate",
            args=[[label], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
        )
        for label in month_labels
    ],
    x=0.05, xanchor="left",
    y=0, yanchor="top",
    len=0.9,
    currentvalue=dict(
        prefix="Month: ",
        font=dict(size=14),
        xanchor="center",
    ),
    pad=dict(t=10, b=10),
)]

# Play / Pause buttons
updatemenus = [dict(
    type="buttons",
    showactive=False,
    y=0,
    x=0.0,
    xanchor="right",
    yanchor="top",
    pad=dict(t=10, r=10),
    buttons=[
        dict(
            label="▶ Play",
            method="animate",
            args=[None, dict(
                frame=dict(duration=700, redraw=True),
                fromcurrent=True,
                transition=dict(duration=200),
            )],
        ),
        dict(
            label="⏸ Pause",
            method="animate",
            args=[[None], dict(
                frame=dict(duration=0, redraw=False),
                mode="immediate",
                transition=dict(duration=0),
            )],
        ),
    ],
)]

fig = go.Figure(
    data=[choropleth],
    frames=frames,
    layout=go.Layout(
        title=dict(
            text="ICE Enforcement Media Risk by County",
            font=dict(size=20),
            x=0.5,
            xanchor="center",
            y=0.97,
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#cce5ff",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(r=10, t=60, l=10, b=80),
        height=580,
        sliders=sliders,
        updatemenus=updatemenus,
        annotations=[dict(
            text=(
                "⚠️ <b>Media attention model</b> — predicts news coverage probability, "
                "not ICE enforcement intensity. "
                "Two-stage model: MA3 arrest-rate forecast (Stage 1) + glmer (Stage 2). "
                "Coverage is systematically lower in rural and Republican-leaning areas."
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.13,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=11, color="#6b7280"),
        )],
    ),
)

# ── Export HTML ───────────────────────────────────────────────────────────────
html_path = DOCS / "index.html"
fig.write_html(
    html_path,
    include_plotlyjs="cdn",   # load Plotly from CDN — keeps file small
    full_html=True,
    config={"responsive": True, "displayModeBar": True},
)
size_kb = html_path.stat().st_size / 1024
print(f"  Interactive map → {html_path}  ({size_kb:.0f} KB)")

# ── Export static PNG for README (requires kaleido) ───────────────────────────
print("Generating static preview …")
try:
    import plotly.express as px

    last_ts    = month_order[-1]
    last_label = pd.Timestamp(last_ts).strftime("%b %Y")
    last_z     = pivot[last_ts].tolist()
    last_flag  = flagged_lookup[last_ts].tolist()

    fig_static = go.Figure(go.Choropleth(
        geojson=counties,
        locations=all_fips,
        z=last_z,
        colorscale="YlOrRd",
        zmin=0,
        zmax=0.85,
        colorbar=dict(title="Risk Score", tickformat=".0%"),
        marker_line_width=0.2,
        marker_line_color="white",
    ))
    fig_static.update_layout(
        title=dict(
            text=f"ICE Enforcement Media Risk — {last_label}",
            x=0.5, xanchor="center",
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#cce5ff",
        ),
        margin=dict(r=10, t=50, l=10, b=10),
        height=460,
    )
    png_path = DOCS / "preview.png"
    fig_static.write_image(str(png_path), width=1200, height=520, scale=2)
    print(f"  Static preview   → {png_path}")

except Exception as e:
    print(f"  PNG skipped (install kaleido for static image): {e}")
    print("      pip install kaleido")

print("\nDone.")
print("Next steps:")
print("  1. Commit docs/ to your repo and push")
print("  2. GitHub repo → Settings → Pages → Source: Deploy from branch → main / docs/")
print("  3. Map will be live at: https://<you>.github.io/<repo>/")
