# Model card — educational baseline

Selected model: gradient_boosting. Decision threshold: 0.31, chosen on validation F1. Training-only five-fold average precision determined the model; the small score differences do not establish superiority beyond this sample.

## Frozen test results

| Metric | Value |
|---|---:|
| Average precision | 0.6698 |
| ROC-AUC | 0.8522 |
| Precision | 0.5469 |
| Recall | 0.7487 |
| F1 | 0.6321 |
| Accuracy | 0.7686 |
| Brier score | 0.1342 |

Confusion matrix: 803 true negatives, 232 false positives, 94 false negatives and 280 true positives. Among 374 churn-labelled test customers, 280 were identified; 232 non-churners were also flagged. This shows the cost of improving recall. A 0.31 threshold is an academic choice, not an economically optimal retention policy.

## Data and use

7,043 records, supplied telco workbook. 60% training, 20% validation, 20% test with stratification. Raw charges are converted to numeric; missing values are imputed within training folds. Excluded: outcome fields, unverified CLTV, identifiers/geography and sensitive demographics. Demographics are used only for descriptive held-out subgroup checks.

Intended use: internship education and retrospective classification demonstrations. Not suitable for autonomous customer decisions. Future label horizon, timestamps, data provenance and redistribution rights remain unverified. Snapshot sampling does not test future drift. Importance is association, not causation. Brier score is reported, but full calibration assessment and uncertainty intervals remain future work.

## Verification

The training command completed on the supplied workbook. Three pytest tests passed. All 13 notebook code cells executed in order with embedded outputs; this environment used the included in-process runner because network kernel sockets are restricted. The browser model matched Python predictions on 100 sampled records to less than 1e-15 absolute probability difference. JavaScript syntax checks passed. The demo is publicly deployed at https://jairus-telco-retention.kola-j-omondi.chatgpt.site.

Streamlit default-view and empty-filter runtime checks passed.
