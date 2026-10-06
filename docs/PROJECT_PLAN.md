# Project plan and learning checkpoints

## 1. Business understanding
We want to identify customer profiles associated with churn so a retention team can prioritize investigation. Observational unit: one customer. Target: supplied Churn Value. The future horizon and feature collection dates are not documented. This is an educational retrospective experiment until those are established.
Checkpoint: explain why predicting historical labels differs from predicting next month's churn.

## 2. Data understanding
Audit schema, IDs, class balance, missing values, ranges and source provenance. The dataset is a California telco snapshot. Do not present it as Kenyan customer data. The PDF mentor dashboard uses a separate bank dataset (10,000 rows, France/Germany/Spain, credit scores and balances); those fields cannot be recreated from this workbook.
Checkpoint: distinguish a within-segment churn rate from the fraction of all churners in a segment. The sample notebook mixes these interpretations.

## 3. Preparation
Allowlist predictors. Exclude outcome-derived fields, redundant target, IDs, geographic identifiers and unverified CLTV. Gender and Senior Citizen are reserved for aggregate subgroup evaluation, not model inputs. Convert blank Total Charges to missing and fit imputers on training folds. Retain zero-tenure customers.
Checkpoint: explain why dropping all incomplete customers can bias the cohort.

## 4. Modelling
Stratified 60/20/20 train/validation/test split with fixed seed and recorded IDs. Compare dummy, logistic regression, random forest and gradient boosting using training-only five-fold average precision. Choose the model by that score. Tune threshold only on validation F1; no financial cost assumptions are supplied. Keep the fitted training model frozen for final evaluation.
Checkpoint: interpret precision, recall, PR-AUC/average precision, ROC-AUC and Brier score. Explain why accuracy alone is insufficient.

## 5. Evaluation
Evaluate once on the test set after freezing choices. Inspect false positives/negatives and subgroup results. Importance is computed on validation data. No claim of causal retention benefit, production calibration or temporal generalization.
Checkpoint: describe how you would validate on a later real cohort and run an ethical retention experiment.

## 6. Delivery and monitoring plan
Streamlit displays historical cohort insights and evaluation, and supports batch scoring. Future production needs authenticated access, feature contracts, verified prediction horizon, dated cohorts, drift monitoring, delayed outcome evaluation and controlled intervention trials. This prototype does not implement those services.
Checkpoint: explain why a dashboard plus a model is not yet a live retention system.

## Later enhancements
After reviewing the first baseline: limited training-only hyperparameter search, calibration using separate folds, capacity-based threshold selection, training-only customer segmentation and a Power BI export. Do not add RNNs, time-series sales forecasts, purchase frequency or engagement scores without the relevant data.
