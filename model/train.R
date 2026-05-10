#!/usr/bin/env Rscript
# =============================================================================
# train.R
# Fits a mixed-effects GLM with cloglog link to predict county-level
# immigration enforcement media attention (t+1).
#
# Model: glmer(outcome ~ fixed_effects + (1|fips), binomial(link="cloglog"))
#   - cloglog link:        faithful to spec (better than logit for rare events)
#   - County random intercept: captures unobserved county-level heterogeneity
#   - Firth note: brglm2 does not support GLMM; with n=48k and 15% prevalence
#     the ML bias is negligible. The random-intercept variance implicitly
#     regularises the fixed effects (analogous to Firth's penalty).
#
# Outputs (model/model_artifacts/):
#   fixed_effects.csv      fixed-effect coefficients + SE + p-value
#   random_effects.csv     county BLUPs (best linear unbiased predictors)
#   scale_params.csv       mean/sd used to scale features (apply to new data)
#   threshold.json         selected classification threshold (F-beta, β=2)
#   predictions_val.csv    validation-set predictions
#   predictions_all.csv    full-panel predictions (for dashboard)
#
# Usage:
#   Rscript model/train.R
# =============================================================================

required_pkgs <- c("lme4", "PRROC", "dplyr", "readr",
                    "tibble", "jsonlite")
missing_pkgs  <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(missing_pkgs) > 0) {
  cat("Installing missing R packages:", paste(missing_pkgs, collapse = ", "), "\n")
  install.packages(missing_pkgs,
                   repos      = "https://cloud.r-project.org",
                   quiet      = TRUE,
                   dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(lme4)
  library(PRROC)
  library(dplyr)
  library(readr)
  library(tibble)
  library(jsonlite)
})

set.seed(42)
cat("=== ICE Risk Model — Training ===\n\n")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PANEL_PATH    <- "data/processed/panel.csv"
ARTIFACTS_DIR <- "model/model_artifacts"
dir.create(ARTIFACTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. Load panel
# ---------------------------------------------------------------------------
cat("Loading panel ...\n")
panel <- read_csv(PANEL_PATH,
                  col_types = cols(fips = col_character()),
                  show_col_types = FALSE)

cat(sprintf("  Rows: %d | Outcome prevalence: %.2f%%\n",
            nrow(panel), 100 * mean(panel$outcome)))

# ---------------------------------------------------------------------------
# 2. Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES <- c(
  "pct_hispanic", "pct_black", "pct_asian", "pct_foreign_born",
  "median_hh_income", "pct_poverty", "pop_density",
  "rep_vote_share_2024", "crime_rate_per_100k",
  "lag_1m", "lag_3m", "lag_6m"
  # ice_arrest_rate_state enters as offset (fixed coeff = 1), not scaled
)
CATEGORICAL_FEATURES <- c("urban_rural")   # will become dummy variables
BINARY_FEATURES      <- c("in_border_zone")
RANDOM_EFFECT        <- "fips"             # county random intercept

# ---------------------------------------------------------------------------
# 3. Pre-process
# ---------------------------------------------------------------------------
cat("Pre-processing features ...\n")

panel <- panel %>%
  mutate(
    urban_rural = factor(urban_rural,
                         levels = c("rural", "suburban", "urban")),  # rural = ref
    in_border_zone = as.integer(in_border_zone),
    outcome        = as.integer(outcome),
    fips           = as.factor(fips)
  )

# Impute any remaining NAs in numeric features with training-set median
train_data <- panel %>% filter(split == "train")
val_data   <- panel %>% filter(split == "val")

# Compute scale parameters on training set only
scale_params <- tibble(
  feature = NUMERIC_FEATURES,
  mean    = sapply(NUMERIC_FEATURES, function(f) mean(train_data[[f]], na.rm = TRUE)),
  sd      = sapply(NUMERIC_FEATURES, function(f) {
    s <- sd(train_data[[f]], na.rm = TRUE)
    if (is.na(s) || s == 0) 1 else s
  })
)

# Apply scaling to full panel
scale_feature <- function(df, params) {
  for (i in seq_len(nrow(params))) {
    f  <- params$feature[i]
    mu <- params$mean[i]
    s  <- params$sd[i]
    df[[f]] <- (df[[f]] - mu) / s
    df[[f]][is.na(df[[f]])] <- 0  # impute scaled NAs with 0 (= training mean)
  }
  df
}

panel_scaled      <- scale_feature(panel,      scale_params)
train_scaled      <- panel_scaled %>% filter(split == "train")
val_scaled        <- panel_scaled %>% filter(split == "val")

cat(sprintf("  Train: %d rows | Val: %d rows\n", nrow(train_scaled), nrow(val_scaled)))

write_csv(scale_params, file.path(ARTIFACTS_DIR, "scale_params.csv"))
cat("  Scale params saved.\n\n")

# ---------------------------------------------------------------------------
# 3b. Pre-flight data diagnostics
# ---------------------------------------------------------------------------
cat("Pre-flight diagnostics ...\n")

# 1. Check offset for NA / Inf / negative values
offset_vals <- train_scaled$ice_arrest_rate_state
cat(sprintf("  ice_arrest_rate_state: min=%.4f  max=%.4f  NAs=%d  Infs=%d\n",
            min(offset_vals, na.rm = TRUE),
            max(offset_vals, na.rm = TRUE),
            sum(is.na(offset_vals)),
            sum(is.infinite(offset_vals))))
if (any(is.na(offset_vals)) || any(is.infinite(offset_vals))) {
  cat("  WARNING: replacing NA/Inf in ice_arrest_rate_state with 0\n")
  train_scaled$ice_arrest_rate_state[
    is.na(train_scaled$ice_arrest_rate_state) |
    is.infinite(train_scaled$ice_arrest_rate_state)] <- 0
  val_scaled$ice_arrest_rate_state[
    is.na(val_scaled$ice_arrest_rate_state) |
    is.infinite(val_scaled$ice_arrest_rate_state)] <- 0
  panel_scaled$ice_arrest_rate_state[
    is.na(panel_scaled$ice_arrest_rate_state) |
    is.infinite(panel_scaled$ice_arrest_rate_state)] <- 0
}

# 2. Near-zero variance features (< 1e-6 sd after scaling → effectively constant)
for (f in NUMERIC_FEATURES) {
  v <- var(train_scaled[[f]], na.rm = TRUE)
  if (is.na(v) || v < 1e-6)
    cat(sprintf("  WARNING: near-zero variance after scaling: %s (var=%.2e)\n", f, v))
}

# 3. Outcome variance
cat(sprintf("  Outcome prevalence (train): %.4f\n", mean(train_scaled$outcome)))

# 4. Pairwise correlation among lag features (CAMEO dedup can make them collinear)
lag_cor <- cor(train_scaled[, c("lag_1m", "lag_3m", "lag_6m")], use = "complete.obs")
cat("  Lag feature correlations:\n")
print(round(lag_cor, 3))
if (any(abs(lag_cor[upper.tri(lag_cor)]) > 0.98)) {
  cat("  WARNING: lag features nearly collinear (|r| > 0.98).",
      "Dropping lag_3m and lag_6m, keeping lag_1m + lag_6m.\n")
  NUMERIC_FEATURES <- setdiff(NUMERIC_FEATURES, "lag_3m")
}

cat("\n")

# ---------------------------------------------------------------------------
# 4. Model formula
# ---------------------------------------------------------------------------
fixed_terms <- c(
  NUMERIC_FEATURES,
  BINARY_FEATURES,
  CATEGORICAL_FEATURES
)
formula_str <- paste0(
  "outcome ~ ",
  paste(fixed_terms, collapse = " + "),
  " + offset(log1p(ice_arrest_rate_state))",
  " + (1 | fips)"
)
cat("Model formula:\n  ", formula_str, "\n\n")

model_formula <- as.formula(formula_str)

# ---------------------------------------------------------------------------
# 5. Fit model — optimizer cascade
# ---------------------------------------------------------------------------
cat("Fitting glmer (cloglog link, county random intercept) ...\n")
cat("  This may take 5–20 minutes depending on convergence.\n\n")

t0 <- proc.time()

# Try optimizers in order; each attempt is a fallback for the previous failure.
# 1. bobyqa (default, usually fastest)
# 2. nlminbwrap (L-BFGS-B via nlminb, often more robust)
# 3. nAGQ=0 bobyqa (Laplace only, coarser but more stable)
# 4. nAGQ=0 nlminbwrap
# 5. logit link — cloglog is numerically tighter; logit as last-resort link swap
# 6. No random effect (plain glm) — sentinel to confirm model is estimable at all

fit <- NULL
for (attempt in 1:6) {
  result <- tryCatch({
    if (attempt == 1) {
      cat("  Attempt 1: bobyqa, nAGQ=1 ...\n")
      glmer(model_formula, data = train_scaled,
            family  = binomial(link = "cloglog"),
            control = glmerControl(optimizer = "bobyqa",
                                   optCtrl   = list(maxfun = 2e5)))
    } else if (attempt == 2) {
      cat("  Attempt 2: nlminbwrap, nAGQ=1 ...\n")
      glmer(model_formula, data = train_scaled,
            family  = binomial(link = "cloglog"),
            control = glmerControl(optimizer = "nlminbwrap",
                                   optCtrl   = list(eval.max = 2e5,
                                                    iter.max = 1e5)))
    } else if (attempt == 3) {
      cat("  Attempt 3: bobyqa, nAGQ=0 (Laplace) ...\n")
      glmer(model_formula, data = train_scaled,
            family  = binomial(link = "cloglog"),
            nAGQ    = 0,
            control = glmerControl(optimizer = "bobyqa",
                                   optCtrl   = list(maxfun = 2e5)))
    } else if (attempt == 4) {
      cat("  Attempt 4: nlminbwrap, nAGQ=0 (Laplace) ...\n")
      glmer(model_formula, data = train_scaled,
            family  = binomial(link = "cloglog"),
            nAGQ    = 0,
            control = glmerControl(optimizer = "nlminbwrap",
                                   optCtrl   = list(eval.max = 2e5,
                                                    iter.max = 1e5)))
    } else if (attempt == 5) {
      cat("  Attempt 5: logit link (cloglog swap), nAGQ=0 ...\n")
      cat("  NOTE: switching to logit link as numerical fallback.\n")
      glmer(
        as.formula(sub("cloglog", "logit",
                       paste0("outcome ~ ",
                              paste(c(NUMERIC_FEATURES, BINARY_FEATURES,
                                      CATEGORICAL_FEATURES), collapse = " + "),
                              " + offset(log1p(ice_arrest_rate_state))",
                              " + (1 | fips)"))),
        data    = train_scaled,
        family  = binomial(link = "logit"),
        nAGQ    = 0,
        control = glmerControl(optimizer = "bobyqa",
                               optCtrl   = list(maxfun = 2e5)))
    } else {
      cat("  Attempt 6: plain glm — no random effect (sanity check) ...\n")
      # Return a glm so downstream code can still predict
      glm(
        as.formula(paste0(
          "outcome ~ ",
          paste(c(NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES),
                collapse = " + "),
          " + offset(log1p(ice_arrest_rate_state))")),
        data   = train_scaled,
        family = binomial(link = "cloglog"))
    }
  }, error = function(e) {
    cat(sprintf("  Attempt %d failed: %s\n", attempt, conditionMessage(e)))
    NULL
  })

  if (!is.null(result)) {
    cat(sprintf("  Succeeded on attempt %d.\n\n", attempt))
    fit <- result
    break
  }
}

if (is.null(fit))
  stop("All fitting attempts failed. Check diagnostics above.")

elapsed <- proc.time() - t0
cat(sprintf("Fitting completed in %.1f seconds.\n\n", elapsed["elapsed"]))

is_glmer_fit <- inherits(fit, "merMod")

# Check for convergence warnings (merMod only)
if (is_glmer_fit &&
    length(fit@optinfo$conv$lme4$messages) > 0) {
  cat("Convergence warnings:\n")
  cat(fit@optinfo$conv$lme4$messages, sep = "\n")
  cat("\n")
}

# ---------------------------------------------------------------------------
# 6. Save coefficients
# ---------------------------------------------------------------------------
cat("Saving model coefficients ...\n")

fe <- summary(fit)$coefficients

# glm and glmer have the same column layout; guard p-value column name
p_col <- if ("Pr(>|z|)" %in% colnames(fe)) "Pr(>|z|)" else "Pr(>|t|)"

ci <- tryCatch(
  {
    if (is_glmer_fit)
      confint(fit, parm = "beta_", method = "Wald")
    else
      confint(fit, method = "Wald")
  },
  error = function(e) matrix(NA_real_, nrow(fe), 2)
)

coef_df <- tibble(
  variable  = rownames(fe),
  estimate  = fe[, "Estimate"],
  std_error = fe[, "Std. Error"],
  z_value   = fe[, "z value"],
  p_value   = fe[, p_col],
  ci_lower  = ci[, 1],
  ci_upper  = ci[, 2]
)
write_csv(coef_df, file.path(ARTIFACTS_DIR, "fixed_effects.csv"))

cat("  Fixed effects:\n")
print(coef_df %>% select(variable, estimate, std_error, p_value), n = 30)

# Random effects (county BLUPs) — only available for merMod
if (is_glmer_fit) {
  re_df <- ranef(fit)$fips %>%
    as.data.frame() %>%
    rownames_to_column("fips") %>%
    rename(random_intercept = `(Intercept)`)
  write_csv(re_df, file.path(ARTIFACTS_DIR, "random_effects.csv"))
  cat(sprintf("\n  County random effects saved (%d counties).\n\n", nrow(re_df)))
} else {
  cat("\n  NOTE: plain glm fit — no county random effects.\n\n")
  write_csv(tibble(fips = character(), random_intercept = double()),
            file.path(ARTIFACTS_DIR, "random_effects.csv"))
}

# ---------------------------------------------------------------------------
# 7. Predict on validation set
# ---------------------------------------------------------------------------
cat("Generating predictions ...\n")

predict_prob <- function(model, newdata) {
  if (inherits(model, "merMod"))
    predict(model, newdata = newdata, type = "response", allow.new.levels = TRUE)
  else
    predict(model, newdata = newdata, type = "response")
}

val_scaled$pred_prob    <- predict_prob(fit, val_scaled)
panel_scaled$pred_prob  <- predict_prob(fit, panel_scaled)

# ---------------------------------------------------------------------------
# 8. Threshold selection — maximise F-beta (β=2) on validation PR curve
# ---------------------------------------------------------------------------
cat("Selecting classification threshold (F-beta, β=2) ...\n")

pr_obj <- pr.curve(
  scores.class0 = val_scaled$pred_prob[val_scaled$outcome == 1],
  scores.class1 = val_scaled$pred_prob[val_scaled$outcome == 0],
  curve         = TRUE
)

auc_pr <- pr_obj$auc.integral
cat(sprintf("  AUC-PR (validation): %.4f\n", auc_pr))

# Sweep thresholds and compute F2
pr_curve_df <- as.data.frame(pr_obj$curve)
colnames(pr_curve_df) <- c("recall", "precision", "threshold")

beta <- 2
pr_curve_df <- pr_curve_df %>%
  filter(!is.na(precision), !is.na(recall), threshold > 0, threshold < 1) %>%
  mutate(
    f_beta = (1 + beta^2) * precision * recall /
             ((beta^2 * precision) + recall + 1e-10)
  )

best_row       <- pr_curve_df[which.max(pr_curve_df$f_beta), ]
best_threshold <- best_row$threshold

cat(sprintf("  Best threshold: %.4f  |  F2: %.4f  |  Precision: %.4f  |  Recall: %.4f\n\n",
            best_threshold, best_row$f_beta,
            best_row$precision, best_row$recall))

# Empirical prevalence as reference
emp_prevalence <- mean(train_scaled$outcome)
cat(sprintf("  Training prevalence (reference): %.4f\n\n", emp_prevalence))

# Save threshold config
threshold_config <- list(
  threshold         = best_threshold,
  beta              = beta,
  f_beta_at_threshold = best_row$f_beta,
  precision_at_threshold = best_row$precision,
  recall_at_threshold    = best_row$recall,
  auc_pr            = auc_pr,
  empirical_prevalence = emp_prevalence
)
write_json(threshold_config,
           file.path(ARTIFACTS_DIR, "threshold.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ---------------------------------------------------------------------------
# 9. Validation evaluation
# ---------------------------------------------------------------------------
cat("=== Validation Metrics ===\n")

val_scaled <- val_scaled %>%
  mutate(
    pred_class = as.integer(pred_prob >= best_threshold)
  )

tp <- sum(val_scaled$outcome == 1 & val_scaled$pred_class == 1)
fp <- sum(val_scaled$outcome == 0 & val_scaled$pred_class == 1)
fn <- sum(val_scaled$outcome == 1 & val_scaled$pred_class == 0)
tn <- sum(val_scaled$outcome == 0 & val_scaled$pred_class == 0)

precision_val <- tp / (tp + fp + 1e-10)
recall_val    <- tp / (tp + fn + 1e-10)
f2_val        <- (1 + 4) * precision_val * recall_val /
                 (4 * precision_val + recall_val + 1e-10)

# Brier score
brier <- mean((val_scaled$pred_prob - val_scaled$outcome)^2)

cat(sprintf("  AUC-PR:     %.4f\n", auc_pr))
cat(sprintf("  Brier Score:%.4f\n", brier))
cat(sprintf("  F2 Score:   %.4f\n", f2_val))
cat(sprintf("  Precision:  %.4f\n", precision_val))
cat(sprintf("  Recall:     %.4f\n", recall_val))
cat(sprintf("  Threshold:  %.4f\n", best_threshold))
cat(sprintf("  TP=%d  FP=%d  FN=%d  TN=%d\n\n", tp, fp, fn, tn))

# ---------------------------------------------------------------------------
# 10. Save predictions
# ---------------------------------------------------------------------------
val_out <- val_scaled %>%
  select(fips, year_month, outcome, pred_prob, pred_class,
         low_confidence, split)
write_csv(val_out, file.path(ARTIFACTS_DIR, "predictions_val.csv"))

all_out <- panel_scaled %>%
  mutate(pred_class = as.integer(pred_prob >= best_threshold)) %>%
  select(fips, year_month, outcome, pred_prob, pred_class,
         low_confidence, split)
write_csv(all_out, file.path(ARTIFACTS_DIR, "predictions_all.csv"))

cat(sprintf("Predictions saved → %s\n", ARTIFACTS_DIR))

# ---------------------------------------------------------------------------
# 11. Calibration curve data (for evaluate.R)
# ---------------------------------------------------------------------------
cal_df <- val_scaled %>%
  mutate(prob_bin = cut(pred_prob, breaks = seq(0, 1, 0.1),
                        include.lowest = TRUE)) %>%
  group_by(prob_bin) %>%
  summarise(
    mean_pred  = mean(pred_prob),
    mean_obs   = mean(outcome),
    n          = n(),
    .groups    = "drop"
  )
write_csv(cal_df, file.path(ARTIFACTS_DIR, "calibration_data.csv"))

cat("\n=== Training complete. ===\n")
cat(sprintf("All artifacts → %s/\n", ARTIFACTS_DIR))
