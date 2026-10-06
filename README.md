# Customer Churn Prediction — Telco Retention Lab

An educational, reproducible telco churn project by Jairus Omondi. Covers business understanding, data audit, exploration, leakage-aware preparation, model comparison, threshold selection, holdout evaluation, explanations and a Streamlit dashboard.

**Live demo:** https://jairus-telco-retention.kola-j-omondi.chatgpt.site

The demo runs the trained gradient boosting model entirely in the browser. It includes customer cohort analysis, a profile scoring form, model comparison, test metrics and methodology. The exported model matched Python on 100 sampled profiles (maximum absolute probability difference below 1e-15).

**Notebook:** [Complete executed analysis](notebooks/01_customer_churn.ipynb), with EDA, preprocessing, model comparison, threshold trade-offs, evaluation, calibration diagnostics and conclusions.

## Start on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
# Place mentor workbook in data/raw/Telco_customer_churn.xlsx
python -m churn.pipeline
pytest -q
streamlit run app.py
```

For notebooks, run `jupyter lab` from the repository root and open `notebooks/01_customer_churn.ipynb`. Linux/macOS use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Repository contents

- `src/churn/pipeline.py`: data checks, preprocessing, train/validation/test experiment.
- `notebooks/01_customer_churn.ipynb`: explanations, exercises and generated results.
- `app.py`: historical customer dashboard and unlabelled batch scoring.
- `reports/`: aggregate audit, model comparison, metrics, charts and feature importance.
- `docs/PROJECT_PLAN.md`: full lifecycle, scope and learning checkpoints.
- `docs/MODEL_CARD.md`: results, limits and intended use.
- `tests/`: predictor leakage, preprocessing and metric checks.

## Method

Target: `Churn Value`. Split: stratified 60/20/20, seed 42. Models are selected using five-fold training average precision. Threshold maximizes validation F1. The test set is used only after model and threshold are frozen. Preprocessing is fitted inside each cross-validation fold. Outcome-derived fields are excluded by an explicit feature allowlist. The final artifact retains the training-fitted model; it is not refitted on validation/test.

## Scope and source limitations

The mentor workbook has 7,043 customer snapshots; 1,869 have churn labels. No dated sales transactions or longitudinal customer history are supplied. Monthly charges are a billing snapshot, not a sales growth series. The mentor PDF shows a different banking dataset. The provided sample notebook uses a different CSV schema and should not be run unchanged on this workbook.

The future label horizon and decision-time availability of predictors are unverified. This project demonstrates retrospective classification, not a validated future churn intervention system. Sensitive attributes are excluded as predictors but proxy effects can remain. Subgroup metrics are descriptive and do not prove fairness.

Raw data, mentor materials, customer-level exports and serialized model artifacts are excluded from public Git. Dataset provenance and redistribution terms must be verified before public release. Only load joblib artifacts created by this project locally.

## Live demo source

`demo/` is a portable static website. Run `python -m http.server 8000 --directory demo` and open http://localhost:8000. Rebuild model exports after training with `python scripts/export_demo.py`.

The live demo is hosted with Sites. GitHub changes do not automatically redeploy it. No Render service or paid inference API is used.

## Verification

- Three pytest checks passed.
- All 13 notebook code cells executed; outputs and charts are embedded.
- Browser scoring matched Python on 100 profiles.
- JavaScript syntax checks passed.

The separate Dynamic AI Chatbot project follows after reviewing churn.

Streamlit default-view and empty-filter runtime checks passed.
