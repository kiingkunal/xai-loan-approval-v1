"""Tests for model training, evaluation, and the model registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from xai_loan.models.evaluate import compare_models, evaluate
from xai_loan.models.registry import load_model, save_model
from xai_loan.models.train import tune_xgboost


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42
    )
    return X_train, y_train, X_test, y_test


def test_train_models_returns_fitted_estimators(classification_data) -> None:
    from xai_loan.models.train import train_models

    X_train, y_train, _, _ = classification_data
    models = train_models(X_train, y_train, models=("logreg", "rf"))

    assert set(models.keys()) == {"logreg", "rf"}
    for model in models.values():
        # fitted models can predict without raising
        model.predict(X_train.iloc[:1])


def test_train_models_unknown_identifier_raises(classification_data) -> None:
    from xai_loan.models.train import train_models

    X_train, y_train, _, _ = classification_data
    with pytest.raises(ValueError):
        train_models(X_train, y_train, models=("not_a_model",))


def test_tune_xgboost_returns_best_estimator(classification_data) -> None:
    X_train, y_train, _, _ = classification_data
    small_grid = {"max_depth": [3], "n_estimators": [50], "learning_rate": [0.1]}
    best_model = tune_xgboost(X_train, y_train, param_grid=small_grid, cv=2)
    best_model.predict(X_train.iloc[:1])


def test_evaluate_returns_expected_keys_and_bounds(classification_data) -> None:
    X_train, y_train, X_test, y_test = classification_data
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2

    assert "roc_curve" in metrics
    assert set(metrics["roc_curve"].keys()) == {"fpr", "tpr", "thresholds"}


def test_compare_models_returns_dataframe_with_one_row_per_model(classification_data) -> None:
    X_train, y_train, X_test, y_test = classification_data
    models = {
        "logreg": LogisticRegression(max_iter=1000).fit(X_train, y_train),
    }
    comparison = compare_models(models, X_test, y_test)

    assert list(comparison.index) == ["logreg"]
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(comparison.columns)


def test_registry_round_trip(tmp_path: Path, classification_data) -> None:
    X_train, y_train, X_test, _ = classification_data
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    path = save_model(model, "test_model", models_dir=tmp_path)
    assert path.exists()

    loaded = load_model("test_model", models_dir=tmp_path)
    original_preds = model.predict(X_test)
    loaded_preds = loaded.predict(X_test)
    assert (original_preds == loaded_preds).all()


def test_load_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model("does_not_exist", models_dir=tmp_path)
