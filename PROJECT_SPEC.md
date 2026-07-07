# Technical Design

## Goal

Build a modular explainable loan-approval workflow for tabular credit data.
The system should train standard classifiers, explain individual decisions,
audit group-level fairness, and present the full workflow through an
interactive dashboard.

The project is scoped as a transparent decision-support demo rather than a
production lending engine.

## Core Requirements

1. Load public credit-risk datasets through reproducible scripts.
2. Train multiple baseline and ensemble models on a consistent split.
3. Keep preprocessing reusable between training, testing, and live dashboard inputs.
4. Explain predictions with both global and local methods.
5. Generate counterfactual alternatives for rejected or approved applicants.
6. Audit fairness across protected attributes available in the datasets.
7. Convert technical explanation output into plain-English text.
8. Compute a Trust Score for human-review routing.
9. Expose the workflow through a Streamlit dashboard.
10. Cover critical behavior with unit tests.

## Data

| Dataset | Role |
|---|---|
| UCI German Credit | Small, fast dataset for local iteration and demo use. |
| Home Credit Default Risk | Larger, messier dataset for scalability checks. |

Datasets are not committed to the repository. `data/download_data.py` downloads
German Credit directly and prints Kaggle setup guidance for Home Credit.

## Pipeline

```text
Raw applicant data
    -> loader
    -> train/test split
    -> preprocessing
    -> model training
    -> evaluation
    -> explainability
    -> fairness audit
    -> decision-support outputs
    -> Streamlit dashboard
```

## Modules

### `xai_loan.data`

- `loader.py` loads German Credit and Home Credit into normalized DataFrames.
- `preprocessor.py` handles categorical encoding, numeric scaling, imputation,
  and SMOTE training resampling.
- `feature_labels.py` maps raw dataset fields and categorical codes into
  human-readable labels for the dashboard and narrative generator.

### `xai_loan.models`

- `train.py` trains Logistic Regression, Random Forest, XGBoost, and tuned
  XGBoost variants.
- `evaluate.py` computes accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix values, and ROC curve data.
- `registry.py` saves and loads fitted artifacts through `joblib`.
- `pipeline.py` wraps a preprocessor and model for raw-row prediction.

### `xai_loan.explainers`

- `shap_explainer.py` provides global importance and local waterfall-ready
  explanations.
- `lime_explainer.py` provides independent local explanation weights.
- `counterfactual.py` wraps DiCE to generate decision-changing alternatives.
- `narrative.py` turns local attributions and counterfactuals into deterministic
  plain-English summaries.

### `xai_loan.fairness`

- `audit.py` computes demographic parity difference, equalized odds difference,
  false-positive-rate parity, and simple reweighing sample weights.

### `xai_loan.trust`

- `score.py` combines three signals into a 0-100 Trust Score:
  - model confidence,
  - SHAP/LIME agreement,
  - counterfactual feasibility.

The score maps to a routing verdict so low-trust predictions can be reviewed by
a human.

### `app`

The Streamlit dashboard has five tabs:

1. Predict
2. Explain
3. Counterfactuals
4. Fairness
5. About

The app only loads saved artifacts. It does not train models during dashboard
runtime.

## Reproducibility

```bash
pip install -r requirements.txt
pip install -e .
python data/download_data.py
python scripts/train_all.py --skip-home-credit
streamlit run app/streamlit_app.py
```

For the full Home Credit workflow, configure Kaggle credentials and run:

```bash
python data/download_data.py --home-credit
python scripts/train_all.py
```

## Quality Bar

- Critical paths covered by `pytest`.
- Generated datasets, model artifacts, and reports are excluded from git.
- Public functions and classes include type hints and docstrings.
- Narrative output is deterministic.
- The dashboard defaults to the fast German Credit workflow and can switch to
  Home Credit when artifacts are available.

## Known Limitations

- The Home Credit workflow uses the main application table, not the full
  multi-table competition feature set.
- The default 0.5 threshold is not optimized for imbalanced credit-risk costs.
- Fairness metrics are group-level checks and do not replace a full compliance
  review.
- Counterfactual feasibility is heuristic and should be reviewed for domain
  realism before any production use.
