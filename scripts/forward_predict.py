#!/usr/bin/env python3
"""
scripts/forward_predict.py

Generates forward county-level risk predictions beyond the training panel.

Steps:
  1. Define Low / Medium / High thresholds from the validation distribution
  2. Predict month-by-month (up to MAX_FORWARD ahead) with 95% prediction intervals
  3. Stop when >50% of counties have intervals that cross a risk boundary
  4. Write docs/predictions.json
  5. Inject a State → County → Month lookup panel into docs/index.html

Usage:
    python scripts/generate_map.py        # must run first
    python scripts/forward_predict.py

Uncertainty method:
    SE(η) = sqrt( Σ_j (x_j · SE(β_j))² + σ_u² )
    where σ_u = sd of county random intercepts (BLUP distribution).
    This is a diagonal approximation (ignores coefficient covariances).
    Intervals are on the log-odds scale, then transformed via logistic().

Lag proxy:
    The model uses mention_count in lag features; we reconstruct this from
    the 'outcome' column (has_mention at t+1) in predictions_all.csv as a
    0/1 proxy.  For most counties (0-1 events/month) this is exact; for
    high-activity counties it slightly underestimates the lag values.
"""

import json
import textwrap
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.special import expit   # logistic / sigmoid

BASE = Path(__file__).parent.parent
DOCS = BASE / "docs"
DOCS.mkdir(exist_ok=True)

MAX_FORWARD   = 6     # maximum months to attempt
STOP_CROSSING = 0.50  # halt when this fraction of counties have CI crossing a boundary

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

# ── Load artifacts ─────────────────────────────────────────────────────────────
print("Loading model artifacts …")

preds_all = pd.read_csv(
    BASE / "model" / "model_artifacts" / "predictions_all.csv",
    dtype={"fips": str},
)
preds_all["fips"]        = preds_all["fips"].str.zfill(5)
preds_all["year_month"]  = pd.to_datetime(preds_all["year_month"])

fe    = pd.read_csv(BASE / "model" / "model_artifacts" / "fixed_effects.csv")
re    = pd.read_csv(BASE / "model" / "model_artifacts" / "random_effects.csv",
                    dtype={"fips": str})
re["fips"] = re["fips"].str.zfill(5)

scale  = pd.read_csv(BASE / "model" / "model_artifacts" / "scale_params.csv")
static = pd.read_csv(BASE / "data" / "processed" / "static_features.csv",
                     dtype={"fips": str})
static["fips"] = static["fips"].str.zfill(5)

# ── Step 1: Risk thresholds from validation distribution ──────────────────────
print("Setting risk thresholds from validation distribution …")

# Fixed thresholds chosen to be intuitive and well-spaced:
#   Low    < 10%  — below the model's operating threshold (~15.7%), quiet county
#   Medium 10-30% — elevated, monitor
#   High   ≥ 30%  — strong signal
LOW_MAX  = 0.10
HIGH_MIN = 0.30

print(f"  Low    : pred_prob < {LOW_MAX:.0%}")
print(f"  Medium : {LOW_MAX:.0%} – {HIGH_MIN:.0%}")
print(f"  High   : pred_prob ≥ {HIGH_MIN:.0%}")

def risk_level(p: float) -> str:
    if p < LOW_MAX:
        return "low"
    elif p < HIGH_MIN:
        return "medium"
    else:
        return "high"

def level_num(r: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}[r]

# ── Model coefficient and scale lookups ───────────────────────────────────────
coef_d = fe.set_index("variable")["estimate"].to_dict()
se_d   = fe.set_index("variable")["std_error"].to_dict()

scale_d = scale.set_index("feature").to_dict("index")   # feature → {mean, sd}

NUMERIC = [
    "pct_hispanic", "pct_black", "pct_asian", "pct_foreign_born",
    "median_hh_income", "pct_poverty", "pop_density",
    "rep_vote_share_2024", "crime_rate_per_100k",
    "lag_1m", "lag_3m", "lag_6m",
]

def scale_val(feat: str, raw: float) -> float:
    if feat not in scale_d:
        return raw
    mu, sd = scale_d[feat]["mean"], scale_d[feat]["sd"]
    v = (raw - mu) / sd
    return 0.0 if (np.isnan(v) or np.isinf(v)) else float(v)

# Random effect standard deviation (adds prediction uncertainty)
sigma_u   = float(re["random_intercept"].std())
re_dict   = re.set_index("fips")["random_intercept"].to_dict()

static_d  = static.drop_duplicates("fips").set_index("fips").to_dict("index")

# ── Build mention_count history from predictions_all ──────────────────────────
# outcome at (fips, year_month=T) = has_mention at T+1
# → has_mention at month M ≈ outcome at year_month = M - 1 month
print("Building historical mention counts …")

all_months = sorted(preds_all["year_month"].unique())
all_fips   = sorted(static["fips"].unique())

# pivot[fips][ym] = outcome (has_mention at ym+1)
pivot = (
    preds_all
    .pivot_table(index="fips", columns="year_month", values="outcome", aggfunc="first")
    .reindex(index=all_fips, columns=all_months)
    .fillna(0)
)

def actual_mention(fips: str, month) -> float:
    """
    has_mention at `month` from historical data.
    month = actual calendar month we want the count for.
    In the pivot, outcome at row ym = has_mention at ym+1,
    so has_mention at M is stored in column M - 1 month.
    """
    key = pd.Timestamp(month) - pd.DateOffset(months=1)
    if fips in pivot.index and key in pivot.columns:
        return float(pivot.loc[fips, key])
    return 0.0

# ── Forward prediction ─────────────────────────────────────────────────────────
# future_counts[(fips, month)] = predicted prob used as lag proxy for that month
future_counts: dict = {}

last_panel  = max(all_months)                              # 2026-03
start_feat  = last_panel + pd.DateOffset(months=1)        # 2026-04 (feature month)
#  outcome being predicted = start_feat + 1 month = 2026-05

def get_count(fips: str, month) -> float:
    """Historical or forward-predicted mention count for (fips, month)."""
    ts = pd.Timestamp(month)
    key = (fips, ts)
    if key in future_counts:
        return future_counts[key]
    return actual_mention(fips, ts)

def compute_lags(fips: str, feat_month) -> tuple:
    """lag_1m, lag_3m, lag_6m at feature_month T.
    lag_1m = count at T-1
    lag_3m = sum of counts at T-3, T-2, T-1
    lag_6m = sum of counts at T-6 .. T-1
    """
    T = pd.Timestamp(feat_month)
    months_back = [T - pd.DateOffset(months=k) for k in range(1, 7)]
    counts = [get_count(fips, m) for m in months_back]
    return counts[0], sum(counts[:3]), sum(counts)

def predict_one(fips: str, feat_month) -> tuple:
    """
    Returns (prob, lower_95, upper_95) for county `fips`,
    predicting whether enforcement will occur at feat_month + 1 month.
    """
    row = static_d.get(fips)
    if row is None:
        return None

    lag1, lag3, lag6 = compute_lags(fips, feat_month)

    # Linear predictor
    eta      = coef_d.get("(Intercept)", 0.0)
    var_eta  = 0.0

    raw = {
        "pct_hispanic":        row.get("pct_hispanic", 0) or 0,
        "pct_black":           row.get("pct_black", 0) or 0,
        "pct_asian":           row.get("pct_asian", 0) or 0,
        "pct_foreign_born":    row.get("pct_foreign_born", 0) or 0,
        "median_hh_income":    row.get("median_hh_income", 0) or 0,
        "pct_poverty":         row.get("pct_poverty", 0) or 0,
        "pop_density":         row.get("pop_density", 0) or 0,
        "rep_vote_share_2024": row.get("rep_vote_share_2024", 0) or 0,
        "crime_rate_per_100k": row.get("crime_rate_per_100k", 0) or 0,
        "lag_1m": lag1,
        "lag_3m": lag3,
        "lag_6m": lag6,
    }

    for feat in NUMERIC:
        xs = scale_val(feat, raw[feat])
        b  = coef_d.get(feat, 0.0)
        s  = se_d.get(feat, 0.0)
        eta     += b * xs
        var_eta += (xs * s) ** 2

    # Border zone (binary, unscaled)
    bz  = int(row.get("in_border_zone", 0) or 0)
    b   = coef_d.get("in_border_zone", 0.0)
    s   = se_d.get("in_border_zone", 0.0)
    eta     += b * bz
    var_eta += (bz * s) ** 2

    # Urban/rural (categorical dummy)
    ur = str(row.get("urban_rural", "rural") or "rural").lower()
    if ur == "suburban":
        key = "urban_ruralsuburban"
    elif ur == "urban":
        key = "urban_ruralurban"
    else:
        key = None
    if key:
        b  = coef_d.get(key, 0.0)
        s  = se_d.get(key, 0.0)
        eta     += b
        var_eta += s ** 2

    # County random intercept
    eta += re_dict.get(fips, 0.0)

    # NOTE: do NOT add sigma_u^2 here.
    # All 3,222 counties were in the training data, so we have estimated BLUPs
    # for each — they are treated as known offsets, not random new draws.
    # Adding sigma_u^2 would inflate intervals as if predicting for an unseen county.

    # Offset: no future ICE data → set to 0 (log1p(0) = 0, term drops out)

    se_eta = float(np.sqrt(var_eta))
    prob   = float(expit(eta))
    lower  = float(expit(eta - 1.96 * se_eta))
    upper  = float(expit(eta + 1.96 * se_eta))

    return round(prob, 4), round(lower, 4), round(upper, 4)

# ── Iterate forward ────────────────────────────────────────────────────────────
print("\nGenerating forward predictions …")

forward_steps = []

for step in range(1, MAX_FORWARD + 1):
    feat_month    = start_feat + pd.DateOffset(months=(step - 1))
    outcome_month = feat_month + pd.DateOffset(months=1)
    label         = outcome_month.strftime("%b %Y")

    print(f"  Step {step}: predicting {label} …", end="", flush=True)

    rows       = []
    n_crossing = 0

    for fips in all_fips:
        result = predict_one(fips, feat_month)
        if result is None:
            continue
        prob, lower, upper = result

        # Store for lag propagation
        future_counts[(fips, feat_month)] = prob

        rl = risk_level(prob)
        rl_lo = risk_level(lower)
        rl_hi = risk_level(upper)
        crosses = (level_num(rl_lo) != level_num(rl_hi))
        if crosses:
            n_crossing += 1

        rows.append({
            "fips":    fips,
            "prob":    prob,
            "lower":   lower,
            "upper":   upper,
            "risk":    rl,
            "crosses": crosses,
        })

    pct = n_crossing / len(rows) if rows else 0.0
    print(f"  {pct:.1%} of counties CI crosses boundary")

    forward_steps.append({
        "label":       label,
        "feat_month":  str(feat_month.date()),
        "rows":        rows,
        "pct_crossing": round(pct, 4),
    })

    if pct > STOP_CROSSING:
        print(f"  → Stop criterion reached (>{STOP_CROSSING:.0%}). "
              f"Keeping {step} prediction month(s).")
        break

# ── Load county names ──────────────────────────────────────────────────────────
print("\nLoading county names …")
url = ("https://raw.githubusercontent.com/plotly/datasets/"
       "master/geojson-counties-fips.json")
with urlopen(url) as r:
    geo = json.load(r)

name_map  = {f["id"]: f["properties"].get("NAME","") + " " +
                       f["properties"].get("LSAD","")
             for f in geo["features"]}
sfips_map = {f["id"]: f["properties"].get("STATE","")
             for f in geo["features"]}

# ── Build predictions.json ─────────────────────────────────────────────────────
print("Building predictions.json …")

avail_months = [s["label"] for s in forward_steps]

counties_out = {}
for fips in all_fips:
    state_abbr   = STATE_FIPS.get(fips[:2], "")
    county_name  = name_map.get(fips, fips).strip()
    preds_list   = []
    for step in forward_steps:
        row = next((r for r in step["rows"] if r["fips"] == fips), None)
        if row:
            preds_list.append({
                "month":  step["label"],
                "prob":   row["prob"],
                "lower":  row["lower"],
                "upper":  row["upper"],
                "risk":   row["risk"],
            })
    counties_out[fips] = {
        "name":  county_name,
        "state": state_abbr,
        "preds": preds_list,
    }

payload = {
    "thresholds": {"low_max": round(LOW_MAX, 4), "high_min": round(HIGH_MIN, 4)},
    "months":     avail_months,
    "counties":   counties_out,
}

json_path = DOCS / "predictions.json"
with open(json_path, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

print(f"  → {json_path}  ({json_path.stat().st_size / 1024:.0f} KB)")

# ── Inject lookup panel into docs/index.html ──────────────────────────────────
print("Injecting lookup panel into docs/index.html …")

html_path = DOCS / "index.html"
if not html_path.exists():
    print("  docs/index.html not found — run generate_map.py first.")
else:
    _ss = ("padding:.4rem .6rem;border:1px solid #d1d5db;border-radius:6px;"
           "font-size:.9rem;min-width:180px;background:#fff;")
    lookup_html = textwrap.dedent(f"""
    <!-- ═══════════════════════════════════════════════
         COUNTY RISK LOOKUP — generated by forward_predict.py
         ═══════════════════════════════════════════════ -->
    <div id="lookup-wrap" style="
        max-width:860px; margin:2.5rem auto 3rem;
        padding:0 1rem; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">

      <h2 style="font-size:1.4rem;margin-bottom:.75rem;">🔍 County Risk Lookup</h2>

      <!-- Disclaimer -->
      <div style="background:#fff8e1;border-left:4px solid #f59e0b;
                  padding:.8rem 1rem;margin-bottom:1.2rem;border-radius:0 6px 6px 0;
                  font-size:.85rem;line-height:1.5;color:#555;">
        <strong>⚠️ Disclaimer:</strong> This is a <strong>coarse statistical model</strong>
        based on news coverage data (GDELT), not official enforcement records.
        Predictions reflect <em>media coverage probability</em>, not confirmed ICE activity.
        Do <strong>not</strong> use this for real enforcement risk assessment.
        Intervals widen rapidly with forecast horizon; treat any prediction beyond
        the first month as highly approximate.
      </div>

      <!-- Threshold legend -->
      <div id="thresh-legend" style="font-size:.82rem;color:#666;margin-bottom:1rem;">
        Loading thresholds …
      </div>

      <!-- Dropdowns -->
      <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end;
                  margin-bottom:1.2rem;">
        <div style="display:flex;flex-direction:column;gap:.25rem;">
          <label style="font-size:.85rem;font-weight:600;">State</label>
          <select id="sel-state" style="{_ss}" disabled>
            <option>Loading …</option>
          </select>
        </div>
        <div style="display:flex;flex-direction:column;gap:.25rem;">
          <label style="font-size:.85rem;font-weight:600;">County</label>
          <select id="sel-county" style="{_ss}" disabled>
            <option>— select state first —</option>
          </select>
        </div>
        <div style="display:flex;flex-direction:column;gap:.25rem;">
          <label style="font-size:.85rem;font-weight:600;">Month</label>
          <select id="sel-month" style="{_ss}" disabled>
            <option>— select county first —</option>
          </select>
        </div>
      </div>

      <!-- Result card -->
      <div id="result-card" style="display:none;
           background:#f8fafc;border:1px solid #e2e8f0;
           border-radius:8px;padding:1.2rem 1.5rem;">
      </div>

    </div>

    <script>
    (function(){{
      var DATA = null;

      function selStyle() {{
        return "padding:.4rem .6rem;border:1px solid #d1d5db;border-radius:6px;" +
               "font-size:.9rem;min-width:180px;background:#fff;cursor:pointer;";
      }}

      function riskBadge(r) {{
        var cfg = {{
          low:    {{bg:"#dcfce7",color:"#166534",label:"🟢 Low"}},
          medium: {{bg:"#fef9c3",color:"#854d0e",label:"🟡 Medium"}},
          high:   {{bg:"#fee2e2",color:"#991b1b",label:"🔴 High"}},
        }};
        var c = cfg[r] || cfg.medium;
        return '<span style="background:'+c.bg+';color:'+c.color+
               ';padding:.2rem .6rem;border-radius:4px;font-weight:600;font-size:.9rem;">'+
               c.label+'</span>';
      }}

      function pct(v) {{ return (v*100).toFixed(1)+'%'; }}

      function showResult() {{
        var fips  = document.getElementById('sel-county').value;
        var month = document.getElementById('sel-month').value;
        if (!fips || !month || !DATA) return;

        var county = DATA.counties[fips];
        if (!county) return;
        var pred = county.preds.find(function(p){{return p.month===month;}});
        if (!pred) return;

        var halfWidth = ((pred.upper - pred.lower)/2*100).toFixed(1);
        var crossesNote = (
          pred.risk !== (pred.lower < DATA.thresholds.low_max ? 'low' :
                         pred.lower < DATA.thresholds.high_min ? 'medium' : 'high') ||
          pred.risk !== (pred.upper < DATA.thresholds.low_max ? 'low' :
                         pred.upper < DATA.thresholds.high_min ? 'medium' : 'high')
        ) ? '<div style="margin-top:.6rem;font-size:.82rem;color:#92400e;'+
            'background:#fef3c7;padding:.4rem .7rem;border-radius:4px;">'+
            '⚠️ The 95% interval spans more than one risk level — treat with extra caution.</div>'
          : '';

        document.getElementById('result-card').style.display = 'block';
        document.getElementById('result-card').innerHTML =
          '<div style="font-size:1.05rem;font-weight:700;margin-bottom:.8rem;">'+
          county.name+', '+county.state+'  —  '+month+'</div>'+
          '<table style="border-collapse:collapse;font-size:.92rem;width:100%;">'+
          '<tr><td style="padding:.3rem 1rem .3rem 0;color:#555;">Predicted probability</td>'+
          '<td style="font-weight:700;font-size:1.1rem;">'+pct(pred.prob)+'</td></tr>'+
          '<tr><td style="padding:.3rem 1rem .3rem 0;color:#555;">95% prediction interval</td>'+
          '<td>'+pct(pred.lower)+' – '+pct(pred.upper)+
          '&nbsp;<span style="color:#888;font-size:.82rem;">(±'+halfWidth+'%)</span></td></tr>'+
          '<tr><td style="padding:.3rem 1rem .3rem 0;color:#555;">Risk level</td>'+
          '<td>'+riskBadge(pred.risk)+'</td></tr>'+
          '</table>'+crossesNote;
      }}

      function populateMonths(fips) {{
        var sel = document.getElementById('sel-month');
        sel.innerHTML = '';
        sel.disabled = true;
        if (!fips || !DATA) return;
        var preds = DATA.counties[fips].preds;
        preds.forEach(function(p) {{
          var o = document.createElement('option');
          o.value = p.month; o.text = p.month;
          sel.appendChild(o);
        }});
        sel.disabled = false;
        showResult();
      }}

      function populateCounties(state) {{
        var sel = document.getElementById('sel-county');
        sel.innerHTML = '<option value="">— select —</option>';
        sel.disabled = true;
        document.getElementById('sel-month').innerHTML =
          '<option>— select county first —</option>';
        document.getElementById('result-card').style.display = 'none';
        if (!state || !DATA) return;

        var entries = Object.entries(DATA.counties)
          .filter(function(e){{return e[1].state===state;}})
          .sort(function(a,b){{return a[1].name.localeCompare(b[1].name);}});
        entries.forEach(function(e) {{
          var o = document.createElement('option');
          o.value = e[0]; o.text = e[1].name;
          sel.appendChild(o);
        }});
        sel.disabled = false;
      }}

      fetch('./predictions.json')
        .then(function(r){{return r.json();}})
        .then(function(data) {{
          DATA = data;

          // Threshold legend
          document.getElementById('thresh-legend').innerHTML =
            'Thresholds from validation distribution &nbsp;|&nbsp; '+
            '<span style="color:#166534;">🟢 Low</span>: &lt;'+
            pct(data.thresholds.low_max)+'&nbsp; '+
            '<span style="color:#854d0e;">🟡 Medium</span>: '+
            pct(data.thresholds.low_max)+'–'+pct(data.thresholds.high_min)+'&nbsp; '+
            '<span style="color:#991b1b;">🔴 High</span>: ≥'+
            pct(data.thresholds.high_min);

          // Populate states
          var states = [...new Set(
            Object.values(data.counties).map(function(c){{return c.state;}})
          )].filter(Boolean).sort();
          var sel = document.getElementById('sel-state');
          sel.innerHTML = '<option value="">— select —</option>';
          states.forEach(function(s) {{
            var o = document.createElement('option');
            o.value = s; o.text = s;
            sel.appendChild(o);
          }});
          sel.disabled = false;
        }})
        .catch(function(e) {{
          document.getElementById('lookup-wrap').innerHTML +=
            '<p style="color:red;">Failed to load predictions.json: '+e+'</p>';
        }});

      document.addEventListener('change', function(e) {{
        if (e.target.id === 'sel-state')  populateCounties(e.target.value);
        if (e.target.id === 'sel-county') populateMonths(e.target.value);
        if (e.target.id === 'sel-month')  showResult();
      }});
    }})();
    </script>
    """)

    html = html_path.read_text(encoding="utf-8")
    if "lookup-wrap" in html:
        # Already injected — replace between markers
        start = html.index("<!-- ═══════════════")
        end   = html.index("</script>", start) + len("</script>")
        html  = html[:start] + lookup_html.strip() + html[end:]
    else:
        html = html.replace("</body>", lookup_html.strip() + "\n</body>")

    html_path.write_text(html, encoding="utf-8")
    print(f"  → docs/index.html updated  ({html_path.stat().st_size/1024:.0f} KB)")


print("\nDone.")
print(f"Predicted months: {[s['label'] for s in forward_steps]}")
