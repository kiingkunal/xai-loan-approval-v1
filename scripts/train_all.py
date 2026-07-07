"""One-shot script: trains and saves every model for every dataset.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --home-credit-sample-size 100000
    python scripts/train_all.py --skip-home-credit

For each dataset, this fits one `LoanDataPreprocessor`, trains
logreg/rf/xgb plus a grid-searched ``xgb_tuned``, evaluates all four on
a held-out test set, and saves the preprocessor + every model + the
dataset's column metadata to `models/` so the Streamlit dashboard and
report-asset script can load them without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xai_loan.data.loader import DatasetMetadata, load_german_credit, load_home_credit
from xai_loan.data.preprocessor import LoanDataPreprocessor, split_features_target
from xai_loan.models.evaluate import compare_models
from xai_loan.models.registry import save_model
from xai_loan.models.train import train_models, tune_xgboost
from xai_loan.utils.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE


def train_and_evaluate_dataset(
    name: str,
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Fit a preprocessor and four models for one dataset, then save everything.

    Args:
        name: Short dataset identifier used as a filename prefix, e.g.
            ``"german"`` or ``"home_credit"``.
        df: Raw loaded DataFrame (output of a `loader` function).
        metadata: Column-role metadata for `df` (output of the same
            loader call).
        models_dir: Directory to save the preprocessor/models/metadata into.
        reports_dir: Directory to save the comparison table into.

    Returns:
        The model comparison DataFrame (accuracy/precision/recall/f1/roc_auc
        per model on the held-out test set).
    """
    X, y = split_features_target(df, metadata["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = LoanDataPreprocessor(metadata["categorical_cols"], metadata["numeric_cols"])
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    X_train_resampled, y_train_resampled = preprocessor.apply_smote(X_train_transformed, y_train)

    models = train_models(X_train_resampled, y_train_resampled)
    models["xgb_tuned"] = tune_xgboost(X_train_resampled, y_train_resampled)

    comparison = compare_models(models, X_test_transformed, y_test)

    save_model(preprocessor, f"{name}_preprocessor", models_dir=models_dir)
    save_model(dict(metadata), f"{name}_metadata", models_dir=models_dir)
    for model_name, model in models.items():
        save_model(model, f"{name}_{model_name}", models_dir=models_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / f"{name}_model_comparison.csv")

    return comparison


def main() -> None:
    """Parse CLI args and train every configured dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home-credit-sample-size",
        type=int,
        default=50_000,
        help="Rows to sample from Home Credit (full dataset is ~307K rows).",
    )
    parser.add_argument(
        "--skip-home-credit",
        action="store_true",
        help="Train only on German Credit.",
    )
    args = parser.parse_args()

    print("Training on German Credit...")
    german_df, german_metadata = load_german_credit()
    german_results = train_and_evaluate_dataset("german", german_df, german_metadata)
    print(german_results.round(4))

    if args.skip_home_credit:
        return

    print(f"\nTraining on Home Credit (sample_size={args.home_credit_sample_size})...")
    try:
        home_credit_df, home_credit_metadata = load_home_credit(
            sample_size=args.home_credit_sample_size
        )
    except FileNotFoundError as error:
        print(f"Skipping Home Credit: {error}")
        return

    home_credit_results = train_and_evaluate_dataset(
        "home_credit", home_credit_df, home_credit_metadata
    )
    print(home_credit_results.round(4))


if __name__ == "__main__":
    main()
