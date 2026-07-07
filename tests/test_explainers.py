"""Tests for SHAP/LIME explainer output shapes and cross-agreement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xai_loan.explainers.lime_explainer import LIMEExplainer
from xai_loan.explainers.shap_explainer import SHAPExplainer

FEATURE_NAMES = ["x0_dominant", "x1_noise", "x2_noise", "x3_noise"]


@pytest.fixture
def dominant_feature_data() -> tuple[pd.DataFrame, pd.Series]:
    """A dataset where one feature has 5x the weight of the others.

    Used to sanity-check that SHAP and LIME agree on what matters most,
    since LIME exists in this framework purely to cross-validate SHAP.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "x0_dominant": rng.normal(size=n),
            "x1_noise": rng.normal(scale=0.1, size=n),
            "x2_noise": rng.normal(scale=0.1, size=n),
            "x3_noise": rng.normal(scale=0.1, size=n),
        }
    )
    logits = 5 * X["x0_dominant"] + rng.normal(scale=0.1, size=n)
    y = pd.Series((logits > 0).astype(int), name="target")
    return X, y


@pytest.fixture
def fitted_logreg(dominant_feature_data) -> tuple[LogisticRegression, pd.DataFrame]:
    X, y = dominant_feature_data
    model = LogisticRegression().fit(X, y)
    return model, X


@pytest.fixture
def fitted_rf(dominant_feature_data) -> tuple[RandomForestClassifier, pd.DataFrame]:
    X, y = dominant_feature_data
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    return model, X


def test_shap_methods_before_fit_raise() -> None:
    explainer = SHAPExplainer()
    with pytest.raises(RuntimeError):
        explainer.global_importance()


def test_shap_global_importance_shape_tree_model(fitted_rf) -> None:
    model, X = fitted_rf
    explainer = SHAPExplainer().fit(model, X.iloc[:50])
    importance = explainer.global_importance()

    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert len(importance) == len(FEATURE_NAMES)
    assert set(importance["feature"]) == set(FEATURE_NAMES)
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_shap_global_importance_shape_non_tree_model(fitted_logreg) -> None:
    model, X = fitted_logreg
    explainer = SHAPExplainer(kernel_nsamples=100).fit(model, X.iloc[:30])
    importance = explainer.global_importance()

    assert len(importance) == len(FEATURE_NAMES)
    assert importance.iloc[0]["feature"] == "x0_dominant"


def test_shap_local_explanation_shape(fitted_rf) -> None:
    model, X = fitted_rf
    explainer = SHAPExplainer().fit(model, X.iloc[:50])
    result = explainer.local_explanation(X.iloc[[0]])

    assert set(result.keys()) == {"feature_names", "shap_values", "base_value"}
    assert len(result["shap_values"]) == len(FEATURE_NAMES)
    assert isinstance(result["base_value"], float)


def test_shap_plots_run_without_raising(fitted_rf) -> None:
    model, X = fitted_rf
    explainer = SHAPExplainer().fit(model, X.iloc[:50])

    explainer.plot_summary()
    plt.close("all")
    explainer.plot_waterfall(X.iloc[[0]])
    plt.close("all")


def test_lime_local_explanation_before_fit_raises() -> None:
    explainer = LIMEExplainer()
    with pytest.raises(RuntimeError):
        explainer.local_explanation(pd.DataFrame({"a": [1]}))


def test_lime_local_explanation_shape(fitted_rf) -> None:
    model, X = fitted_rf
    explainer = LIMEExplainer().fit(model, X.iloc[:50])
    result = explainer.local_explanation(X.iloc[[0]], num_features=4)

    assert set(result.keys()) == {"feature_names", "weights", "intercept"}
    assert len(result["feature_names"]) == len(result["weights"]) == 4
    assert set(result["feature_names"]).issubset(set(FEATURE_NAMES))


def test_shap_and_lime_agree_on_dominant_feature(fitted_logreg) -> None:
    """Cross-validation sanity check: SHAP and LIME should both flag the
    feature with 5x the weight as the top contributor."""
    model, X = fitted_logreg
    background = X.iloc[:50]
    instance = X.iloc[[60]]

    shap_explainer = SHAPExplainer(kernel_nsamples=200).fit(model, background)
    shap_result = shap_explainer.local_explanation(instance)
    shap_top_feature = shap_result["feature_names"][
        int(np.argmax(np.abs(shap_result["shap_values"])))
    ]

    lime_explainer = LIMEExplainer().fit(model, background)
    lime_result = lime_explainer.local_explanation(instance, num_features=4)
    lime_top_feature = lime_result["feature_names"][
        int(np.argmax(np.abs(lime_result["weights"])))
    ]

    assert shap_top_feature == "x0_dominant"
    assert lime_top_feature == "x0_dominant"
