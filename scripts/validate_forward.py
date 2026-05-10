#!/usr/bin/env python3
"""
scripts/validate_forward.py

Walk-forward backtesting validation of the forward_predict.py pipeline.

Strategy
--------
Pick a CUTOFF feature month (default 2025-09, the last training month).
Mask all subsequent months in the panel and predict them step-by-step using
only data available at or before the cutoff — identical logic to
forward_predict.py so we're testing the real production pipeline.

Three offset modes compared
---------------------------
  zero        — ICE arrest rate offset = 0  (original behaviour)
  predicted   — offset from arrest_rate_val_forecasts.csv (MA3 model)
  oracle      — offset from actual panel data  (best-case upper bound)

Two lag-propagation modes per step within each offset mode
----------------------------------------------------------
  oracle_lags     — lag features use actual historical outcomes
  propagated_lags — lag features use predicted prob as proxy  (production)

Metrics per step and overall:
  AUC-ROC, AUC-PR, F2 (β=2) at operating threshold, Precision, Recall,
  mean predicted vs mean actual (calibration check)

Usage
-----
    /opt/anaconda3/bin/python3 scripts/validate_forward.py
    /opt/anaconda3/bin/python3 scripts/validate_forward.py --cutoff 2025-12 --steps 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

BASE = Path(__file__).parent.parent

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", default="2025-09",
                    help="Last feature month BEFORE masking starts (YYYY-MM). "
                         "Default: 2025-09 (end of training period)")
parser.add_argument("--steps", type=int, default=None,
                    help="Number of forward steps to validate. "
                         "Default: all months with ground truth after cutoff.")
args = parser.parse_args()

CUTOFF = pd.Timestamp(args.cutoff)
print(f"Cutoff feature month : {CUTOFF.strftime('%b %Y')}")
print(f"First outcome masked : {(CUTOFF + pd.DateOffset(months=1)).strftime('%b %Y')}")

# ── Load model artifacts ───────────────────────────────────────────────────────
print("\nLoading model artifacts …")

preds_all = pd.read_csv(
    BASE / "model" / "model_artifacts" / "predictions_all.csv",
    dtype={"fips": str},
)
preds_all["fips"]       = preds_all["fips"].str.zfill(5)
preds_all["year_month"] = pd.to_datetime(preds_all["year_month"])

fe    = pd.read_csv(BASE / "model" / "model_artifacts" / "fixed_effects.csv")
re    = pd.read_csv(BASE / "model" / "model_artifacts" / "random_effects.csv",
                    dtype={"fips": str})
re["fips"] = re["fips"].str.zfill(5)

scale  = pd.read_csv(BASE / "model" / "model_artifacts" / "scale_params.csv")
static = pd.read_csv(BASE / "data" / "processed" / "static_features.csv",
                     dtype={"fips": str})
static["fips"] = static["fips"].str.zfill(5)

thresh_path = BASE / "model" / "model_artifacts" / "threshold.json"
with open(thresh_path) as f:
    THRESHOLD = json.load(f)["threshold"]
print(f"Operating threshold  : {THRESHOLD:.4f}")

# ── Load arrest rate forecasts (for 'predicted' offset mode) ───────────────────
val_fc_path = BASE / "model" / "model_artifacts" / "arrest_rate_val_forecasts.csv"
if val_fc_path.exists():
    val_fc = pd.read_csv(val_fc_path, dtype={"state_fips": str})
    val_fc["feature_month"] = pd.to_datetime(val_fc["feature_month"])
    # (state_fips, feature_month) → predicted rate
    predicted_rate_d = {
        (row["state_fips"], row["feature_month"]): row["rate_selected"]
        for _, row in val_fc.iterrows()
    }
    print(f"Loaded val-period predicted rates for {val_fc['state_fips'].nunique()} states")
else:
    predicted_rate_d = {}
    print("No arrest_rate_val_forecasts.csv found — 'predicted' mode will equal 'zero'")

# ── Load actual panel rates (for 'oracle' offset mode) ────────────────────────
panel = pd.read_csv(BASE / "data" / "processed" / "panel.csv", dtype={"fips": str})
panel["fips"]       = panel["fips"].str.zfill(5)
panel["year_month"] = pd.to_datetime(panel["year_month"])
# (state_fips, feature_month) → actual rate
actual_rate_d = {
    (row["fips"][:2], row["year_month"]): row["ice_arrest_rate_state"]
    for _, row in panel.iterrows()
}

# ── Model lookups ──────────────────────────────────────────────────────────────
coef_d  = fe.set_index("variable")["estimate"].to_dict()
se_d    = fe.set_index("variable")["std_error"].to_dict()
scale_d = scale.set_index("feature").to_dict("index")
re_dict = re.set_index("fips")["random_intercept"].to_dict()
static_d = static.drop_duplicates("fips").set_index("fips").to_dict("index")

NUMERIC = [
    "pct_hispanic", "pct_black", "pct_asian", "pct_foreign_born",
    "median_hh_income", "pct_poverty", "pop_density",
    "rep_vote_share_2024", "crime_rate_per_100k",
    "lag_1m", "lag_3m", "lag_6m",
]

def scale_val(feat, raw):
    if feat not in scale_d:
        return raw
    mu, sd = scale_d[feat]["mean"], scale_d[feat]["sd"]
    v = (raw - mu) / sd
    return 0.0 if (np.isnan(v) or np.isinf(v)) else float(v)

# ── Build historical mention matrix ───────────────────────────────────────────
all_months = sorted(preds_all["year_month"].unique())
all_fips   = sorted(static["fips"].unique())

hist_pivot = (
    preds_all
    .pivot_table(index="fips", columns="year_month", values="outcome", aggfunc="first")
    .reindex(index=all_fips, columns=all_months)
    .fillna(0)
)

def actual_mention(fips, month):
    key = pd.Timestamp(month) - pd.DateOffset(months=1)
    if fips in hist_pivot.index and key in hist_pivot.columns:
        return float(hist_pivot.loc[fips, key])
    return 0.0

def actual_outcome(fips, feat_month):
    fm = pd.Timestamp(feat_month)
    if fips in hist_pivot.index and fm in hist_pivot.columns:
        return float(hist_pivot.loc[fips, fm])
    return np.nan

# ── Predict function ───────────────────────────────────────────────────────────
def predict_one(fips, feat_month, lag_fn, ice_rate: float = 0.0):
    """
    lag_fn(fips, month) → mention count proxy.
    ice_rate            → state-level ICE arrest rate for the offset term.
    Returns (prob, lower_95, upper_95) or None.
    """
    row = static_d.get(fips)
    if row is None:
        return None

    T           = pd.Timestamp(feat_month)
    months_back = [T - pd.DateOffset(months=k) for k in range(1, 7)]
    counts      = [lag_fn(fips, m) for m in months_back]
    lag1, lag3, lag6 = counts[0], sum(counts[:3]), sum(counts)

    eta     = coef_d.get("(Intercept)", 0.0)
    var_eta = 0.0

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
        "lag_1m": lag1, "lag_3m": lag3, "lag_6m": lag6,
    }
    for feat in NUMERIC:
        xs = scale_val(feat, raw[feat])
        b  = coef_d.get(feat, 0.0)
        s  = se_d.get(feat, 0.0)
        eta     += b * xs
        var_eta += (xs * s) ** 2

    bz = int(row.get("in_border_zone", 0) or 0)
    b  = coef_d.get("in_border_zone", 0.0)
    s  = se_d.get("in_border_zone", 0.0)
    eta     += b * bz
    var_eta += (bz * s) ** 2

    ur  = str(row.get("urban_rural", "rural") or "rural").lower()
    key = ("urban_ruralsuburban" if ur == "suburban"
           else "urban_ruralurban" if ur == "urban"
           else None)
    if key:
        b  = coef_d.get(key, 0.0)
        s  = se_d.get(key, 0.0)
        eta     += b
        var_eta += s ** 2

    eta += re_dict.get(fips, 0.0)

    # ICE arrest rate offset: log1p(rate)
    eta += float(np.log1p(max(0.0, ice_rate)))

    se_eta = float(np.sqrt(var_eta))
    prob   = float(expit(eta))
    lower  = float(expit(eta - 1.96 * se_eta))
    upper  = float(expit(eta + 1.96 * se_eta))
    return round(prob, 4), round(lower, 4), round(upper, 4)

# ── Metric helpers ─────────────────────────────────────────────────────────────
def auc_roc(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true);  n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0;  prev_tpr = prev_fpr = 0.0;  auc = 0.0;  prev_s = None
    for s, l in pairs:
        if s != prev_s:
            tpr = tp / n_pos;  fpr = fp / n_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_tpr, prev_fpr, prev_s = tpr, fpr, s
        tp += l;  fp += (1 - l)
    auc += (1.0 - prev_fpr) * (1.0 + prev_tpr) / 2
    return round(auc, 4)

def auc_pr(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0;  prev_rec = 0.0;  prev_pre = 1.0;  auc = 0.0
    for _, l in pairs:
        tp += l;  fp += (1 - l)
        rec = tp / n_pos;  pre = tp / (tp + fp)
        auc += (rec - prev_rec) * (pre + prev_pre) / 2
        prev_rec, prev_pre = rec, pre
    return round(auc, 4)

def fbeta(y_true, y_pred_bin, beta=2):
    tp = sum(a and b for a, b in zip(y_true, y_pred_bin))
    fp = sum((not a) and b for a, b in zip(y_true, y_pred_bin))
    fn = sum(a and (not b) for a, b in zip(y_true, y_pred_bin))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = (1 + beta**2) * prec + beta**2 * rec
    return (round(((1+beta**2)*prec*rec/denom) if denom else 0.0, 4),
            round(prec, 4), round(rec, 4))

def calibration_table(y_true, y_score, n_bins=10):
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs);  bs = n // n_bins
    rows = []
    for i in range(n_bins):
        chunk = pairs[i*bs:(i+1)*bs] if i < n_bins-1 else pairs[i*bs:]
        rows.append({
            "decile":    i + 1,
            "mean_pred": round(np.mean([s for s, _ in chunk]), 4),
            "mean_obs":  round(np.mean([l for _, l in chunk]), 4),
        })
    return rows

# ── Determine masked steps ─────────────────────────────────────────────────────
valid_feat_months = sorted(m for m in all_months if pd.Timestamp(m) > CUTOFF)
if args.steps:
    valid_feat_months = valid_feat_months[:args.steps]

print(f"\nMasked steps ({len(valid_feat_months)}):")
for m in valid_feat_months:
    m_ts  = pd.Timestamp(m)
    out_m = m_ts + pd.DateOffset(months=1)
    print(f"  feature {m_ts.strftime('%b %Y')} → outcome {out_m.strftime('%b %Y')}")

# ── Run validation — three offset modes × two lag modes ───────────────────────
OFFSET_MODES = ["zero", "predicted", "oracle"]
LAG_MODES    = ["oracle_lags", "propagated_lags"]

# State FIPS lookup per county
county_state = {f: f[:2] for f in all_fips}

# Storage for lag propagation per (offset_mode, lag_mode)
future_counts  = {(o, l): {} for o in OFFSET_MODES for l in LAG_MODES}
future_oracle  = {}    # actual outcomes for oracle_lags across all offset modes

all_results = []

print("\n" + "─"*88)
print(f"{'Step':<6}{'Outcome':<12}{'Offset':<12}{'Lags':<17}"
      f"{'AUC-ROC':<9}{'AUC-PR':<8}{'F2':<7}{'Prec':<7}{'Rec':<7}{'MPred':>7}{'MObs':>7}")
print("─"*88)

for step_i, feat_month_raw in enumerate(valid_feat_months, start=1):
    feat_month   = pd.Timestamp(feat_month_raw)
    out_month_ts = feat_month + pd.DateOffset(months=1)
    out_label    = out_month_ts.strftime("%b %Y")
    feat_label   = feat_month.strftime("%b %Y")

    truth = {fips: int(actual_outcome(fips, feat_month))
             for fips in all_fips
             if not np.isnan(actual_outcome(fips, feat_month))}
    if not truth:
        continue

    for offset_mode in OFFSET_MODES:
        for lag_mode in LAG_MODES:

            def lag_fn(fips, month,
                       _om=offset_mode, _lm=lag_mode):
                ts  = pd.Timestamp(month)
                key = (fips, ts)
                fc  = future_counts[(_om, _lm)]
                if key in fc:
                    return fc[key]
                return actual_mention(fips, ts)

            probs_list = []
            truth_list = []

            for fips, gt in truth.items():
                sf = county_state[fips]

                # Determine offset rate
                if offset_mode == "zero":
                    ice_rate = 0.0
                elif offset_mode == "predicted":
                    ice_rate = predicted_rate_d.get((sf, feat_month), 0.0)
                else:  # oracle
                    ice_rate = actual_rate_d.get((sf, feat_month), 0.0)

                result = predict_one(fips, feat_month, lag_fn, ice_rate)
                if result is None:
                    continue
                prob, _, _ = result

                # Update future_counts for lag propagation
                if lag_mode == "oracle_lags":
                    future_counts[(offset_mode, lag_mode)][(fips, feat_month)] = float(gt)
                else:
                    future_counts[(offset_mode, lag_mode)][(fips, feat_month)] = prob

                probs_list.append(prob)
                truth_list.append(gt)

            if not probs_list:
                continue

            y_pred_bin = [int(p >= THRESHOLD) for p in probs_list]
            roc  = auc_roc(truth_list, probs_list)
            pr   = auc_pr(truth_list, probs_list)
            f2, prec, rec = fbeta(truth_list, y_pred_bin)
            mpred = round(np.mean(probs_list), 4)
            mobs  = round(np.mean(truth_list), 4)

            all_results.append({
                "step": step_i, "feat": feat_label, "outcome": out_label,
                "offset": offset_mode, "lags": lag_mode,
                "n": len(truth_list), "n_pos": sum(truth_list),
                "auc_roc": roc, "auc_pr": pr,
                "f2": f2, "precision": prec, "recall": rec,
                "mean_pred": mpred, "mean_obs": mobs,
                "y_true": truth_list, "y_score": probs_list,
            })

            print(f"  {step_i:<5}{out_label:<12}{offset_mode:<12}{lag_mode:<17}"
                  f"{roc:<9.4f}{pr:<8.4f}{f2:<7.4f}{prec:<7.4f}{rec:<7.4f}"
                  f"{mpred:>7.4f}{mobs:>7.4f}")

print("─"*88)

# ── Overall summary ────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("OVERALL SUMMARY  (pooled across all steps)")
print("═"*70)

# Focus on propagated_lags (production mode)
print("\nProduction mode (propagated lags):")
print(f"  {'Offset mode':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'F2':>7} "
      f"{'Prec':>7} {'Rec':>7} {'MPred':>8} {'MObs':>8} {'Bias(pp)':>9}")
print("  " + "─"*75)

for offset_mode in OFFSET_MODES:
    rows = [r for r in all_results
            if r["offset"] == offset_mode and r["lags"] == "propagated_lags"]
    if not rows:
        continue
    y_t = [y for r in rows for y in r["y_true"]]
    y_s = [s for r in rows for s in r["y_score"]]
    y_p = [int(s >= THRESHOLD) for s in y_s]

    roc  = auc_roc(y_t, y_s)
    pr   = auc_pr(y_t, y_s)
    f2, prec, rec = fbeta(y_t, y_p)
    mpred = round(np.mean(y_s), 4)
    mobs  = round(np.mean(y_t), 4)
    bias  = (mpred - mobs) * 100

    print(f"  {offset_mode:<14} {roc:>8.4f} {pr:>8.4f} {f2:>7.4f} "
          f"{prec:>7.4f} {rec:>7.4f} {mpred:>8.4f} {mobs:>8.4f} {bias:>+9.1f}pp")

# Calibration comparison
print(f"\n{'─'*70}")
print("Calibration by decile  (production / propagated lags)")
print(f"  {'Decile':>7}  {'Obs':>7}  "
      f"{'zero':>7}  {'predicted':>10}  {'oracle':>7}")
print("  " + "─"*50)

cal_data = {}
for offset_mode in OFFSET_MODES:
    rows = [r for r in all_results
            if r["offset"] == offset_mode and r["lags"] == "propagated_lags"]
    if not rows:
        continue
    y_t = [y for r in rows for y in r["y_true"]]
    y_s = [s for r in rows for s in r["y_score"]]
    cal_data[offset_mode] = calibration_table(y_t, y_s)

if cal_data:
    n_bins = len(next(iter(cal_data.values())))
    obs_col = cal_data[OFFSET_MODES[-1]]  # oracle has same obs values
    for i in range(n_bins):
        obs_val = obs_col[i]["mean_obs"]
        vals    = {m: cal_data[m][i]["mean_pred"] for m in cal_data}
        print(f"  {i+1:>7}  {obs_val:>7.3f}  "
              f"{vals.get('zero',0):>7.3f}  "
              f"{vals.get('predicted',0):>10.3f}  "
              f"{vals.get('oracle',0):>7.3f}")

# ── Per-step improvement table ────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("Per-step F2 comparison (propagated lags):")
print(f"  {'Step':<6}{'Outcome':<12}{'F2 zero':>9}{'F2 pred':>9}{'F2 oracle':>10}  {'Δ(pred-zero)':>13}")
for step_i in sorted({r["step"] for r in all_results}):
    def get_f2(offset):
        r = next((x for x in all_results
                  if x["step"] == step_i and x["offset"] == offset
                  and x["lags"] == "propagated_lags"), None)
        return r["f2"] if r else float("nan")
    out = next((r["outcome"] for r in all_results if r["step"] == step_i), "?")
    z = get_f2("zero");  p = get_f2("predicted");  o = get_f2("oracle")
    delta = p - z if not np.isnan(p) and not np.isnan(z) else float("nan")
    print(f"  {step_i:<6}{out:<12}{z:>9.4f}{p:>9.4f}{o:>10.4f}  {delta:>+13.4f}")

print("\nDone.")
