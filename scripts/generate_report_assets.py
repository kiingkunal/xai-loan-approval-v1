"""Generates report figures and tables from already-trained models.

Run `scripts/train_all.py` first — this script only loads what's saved
in `models/`, it never trains anything itself. The train/test split is
regenerated deterministically (same loader, same `RANDOM_STATE`,
`TEST_SIZE`), so it doesn't need to be persisted anywhere to stay
consistent with what the models were actually evaluated on.

Usage:
    python scripts/generate_report_assets.py
    python scripts/generate_report_assets.py --skip-home-credit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from xai_loan.data.feature_labels import NARRATIVE_FEATURE_ALLOWLIST
from xai_loan.data.loader import DatasetMetadata, load_german_credit, load_home_credit
from xai_loan.data.preprocessor import split_features_target
from xai_loan.explainers.counterfactual import CounterfactualGenerator
from xai_loan.explainers.lime_explainer import LIMEExplainer
from xai_loan.explainers.narrative import generate_narrative
from xai_loan.explainers.shap_explainer import SHAPExplainer
from xai_loan.fairness.audit import FairnessAuditor, sensitive_feature_series
from xai_loan.models.evaluate import evaluate
from xai_loan.models.pipeline import PipelineModel
from xai_loan.models.registry import load_model
from xai_loan.trust.score import compute_trust_score
from xai_loan.utils.config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE

_MODEL_NAMES = ("logreg", "rf", "xgb", "xgb_tuned")


def _plot_roc_curves(models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, output_path: Path) -> None:
    """Plot ROC curves for every model on one held-out test set."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test)
        ax.plot(
            metrics["roc_curve"]["fpr"],
            metrics["roc_curve"]["tpr"],
            label=f"{name} (AUC={metrics['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_fairness_audit(
    model: object,
    X_test_raw: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    y_test: pd.Series,
    metadata: DatasetMetadata,
    output_path: Path,
) -> None:
    """Audit the best model against every protected attribute and save a CSV."""
    auditor = FairnessAuditor()
    rows = []
    for label, sensitive_series in sensitive_feature_series(metadata["protected_cols"], X_test_raw):
        audit_result = auditor.audit(model, X_test_transformed, y_test, sensitive_series)
        rows.append(
            {
                "protected_attribute": label,
                "demographic_parity_difference": audit_result["demographic_parity_difference"],
                "equalized_odds_difference": audit_result["equalized_odds_difference"],
                "false_positive_rate_parity_difference": audit_result[
                    "false_positive_rate_parity_difference"
                ],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_example_narrative(
    name: str,
    model: object,
    preprocessor: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    shap_explainer: SHAPExplainer,
    background_transformed: pd.DataFrame,
    metadata: DatasetMetadata,
    output_path: Path,
) -> None:
    """Generate one example applicant's narrative + trust score and save it as text."""
    pipeline_model = PipelineModel(
        preprocessor, model, metadata["categorical_cols"] + metadata["numeric_cols"]
    )

    # DiCE rejects query instances with missing values (Home Credit has
    # plenty of NaNs in the raw data; a real loan officer's form
    # wouldn't). Pick the first complete-case row instead of always
    # X_test.iloc[[0]], falling back to row 0 if every row has a gap.
    feature_cols = metadata["categorical_cols"] + metadata["numeric_cols"]
    complete_case_mask = X_test[feature_cols].notna().all(axis=1)
    raw_instance = X_test.loc[complete_case_mask].iloc[[0]] if complete_case_mask.any() else X_test.iloc[[0]]
    preprocessed_instance = preprocessor.transform(raw_instance)
    prediction = int(model.predict(preprocessed_instance)[0])
    shap_exp = shap_explainer.local_explanation(preprocessed_instance)

    lime_explainer = LIMEExplainer().fit(model, background_transformed)
    lime_exp = lime_explainer.local_explanation(preprocessed_instance)

    cf_train_df = X_train.copy()
    cf_train_df[metadata["target_col"]] = y_train
    counterfactual_generator = CounterfactualGenerator().fit(
        model,
        preprocessor,
        cf_train_df,
        metadata["categorical_cols"],
        metadata["numeric_cols"],
        metadata["target_col"],
    )
    # Restrict to the same human-meaningful columns the narrative is
    # allowed to cite -- see feature_labels.NARRATIVE_FEATURE_ALLOWLIST.
    features_to_vary = sorted(NARRATIVE_FEATURE_ALLOWLIST.get(name, set()) & set(feature_cols))
    counterfactuals = counterfactual_generator.generate(
        raw_instance, n=2, desired_class=1 - prediction, features_to_vary=features_to_vary
    )

    narrative = generate_narrative(
        raw_instance,
        prediction,
        shap_exp,
        counterfactuals,
        mode="template",
        categorical_cols=metadata["categorical_cols"],
        dataset=name,
    )
    trust_result = compute_trust_score(raw_instance, pipeline_model, shap_exp, lime_exp, counterfactuals)

    report_text = (
        f"Example applicant narrative ({name}, tuned XGBoost):\n\n{narrative}\n\n"
        f"Trust score: {trust_result['score']}/100 -- verdict: {trust_result['verdict']}\n"
        f"Reason: {trust_result['reason']}\n"
    )
    output_path.write_text(report_text)


def generate_assets_for_dataset(
    name: str,
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    reports_dir: Path = REPORTS_DIR,
) -> None:
    """Generate ROC curves, a SHAP summary plot, a fairness audit table,
    and one example narrative+trust score for an already-trained dataset.

    Args:
        name: Dataset identifier matching the prefix `train_all.py` saved
            models under, e.g. ``"german"``.
        df: Raw loaded DataFrame (output of a `loader` function).
        metadata: Column-role metadata for `df`.
        reports_dir: Directory to save generated assets into.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = load_model(f"{name}_preprocessor")
    models = {model_name: load_model(f"{name}_{model_name}") for model_name in _MODEL_NAMES}
    best_model = models["xgb_tuned"]

    X, y = split_features_target(df, metadata["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_test_transformed = preprocessor.transform(X_test)

    _plot_roc_curves(models, X_test_transformed, y_test, reports_dir / f"{name}_roc_curves.png")

    background_raw = X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE)
    background_transformed = preprocessor.transform(background_raw)
    shap_explainer = SHAPExplainer().fit(best_model, background_transformed)
    shap_explainer.plot_summary()
    plt.savefig(reports_dir / f"{name}_shap_summary.png", bbox_inches="tight")
    plt.close("all")

    _write_fairness_audit(
        best_model, X_test, X_test_transformed, y_test, metadata, reports_dir / f"{name}_fairness_audit.csv"
    )

    _write_example_narrative(
        name,
        best_model,
        preprocessor,
        X_train,
        y_train,
        X_test,
        shap_explainer,
        background_transformed,
        metadata,
        reports_dir / f"{name}_example_narrative.txt",
    )


def main() -> None:
    """Parse CLI args and generate report assets for every configured dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home-credit-sample-size",
        type=int,
        default=50_000,
        help="Must match the sample size used in train_all.py for a consistent split.",
    )
    parser.add_argument("--skip-home-credit", action="store_true")
    args = parser.parse_args()

    print("Generating report assets for German Credit...")
    german_df, german_metadata = load_german_credit()
    generate_assets_for_dataset("german", german_df, german_metadata)

    if args.skip_home_credit:
        return

    print(f"Generating report assets for Home Credit (sample_size={args.home_credit_sample_size})...")
    try:
        home_credit_df, home_credit_metadata = load_home_credit(
            sample_size=args.home_credit_sample_size
        )
    except FileNotFoundError as error:
        print(f"Skipping Home Credit: {error}")
        return

    generate_assets_for_dataset("home_credit", home_credit_df, home_credit_metadata)


if __name__ == "__main__":
    main()
