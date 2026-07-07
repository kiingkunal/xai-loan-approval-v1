"""Tests for the DiCE counterfactual wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from xai_loan.data.preprocessor import LoanDataPreprocessor
from xai_loan.explainers.counterfactual import CounterfactualGenerator

CATEGORICAL_COLS = ["city"]
NUMERIC_COLS = ["income", "loan_amount"]
TARGET_COL = "target"


@pytest.fixture
def fitted_pipeline() -> tuple[LogisticRegression, LoanDataPreprocessor, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame(
        {
            "city": rng.choice(["delhi", "mumbai"], n),
            "income": rng.normal(50_000, 10_000, n),
            "loan_amount": rng.normal(20_000, 5_000, n),
        }
    )
    # clearly separable: high income relative to loan amount -> approved (0)
    score = df["income"] - 1.5 * df["loan_amount"]
    df[TARGET_COL] = (score <= score.median()).astype(int)

    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    X_transformed = preprocessor.fit_transform(df[CATEGORICAL_COLS + NUMERIC_COLS])
    model = LogisticRegression(max_iter=1000).fit(X_transformed, df[TARGET_COL])
    return model, preprocessor, df


def test_generate_before_fit_raises() -> None:
    generator = CounterfactualGenerator()
    with pytest.raises(RuntimeError):
        generator.generate(pd.DataFrame({"city": ["delhi"]}))


def test_generate_returns_counterfactuals_with_correct_columns(fitted_pipeline) -> None:
    model, preprocessor, df = fitted_pipeline
    generator = CounterfactualGenerator().fit(
        model, preprocessor, df, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
    )

    instance = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[[0]]
    current_pred = model.predict(preprocessor.transform(instance))[0]
    desired_class = 1 - int(current_pred)

    counterfactuals = generator.generate(instance, n=2, desired_class=desired_class)

    assert len(counterfactuals) <= 2
    assert len(counterfactuals) > 0
    for cf in counterfactuals:
        assert set(cf.keys()) == set(CATEGORICAL_COLS + NUMERIC_COLS)
        assert TARGET_COL not in cf


def test_generated_counterfactuals_flip_the_prediction(fitted_pipeline) -> None:
    model, preprocessor, df = fitted_pipeline
    generator = CounterfactualGenerator().fit(
        model, preprocessor, df, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
    )

    instance = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[[0]]
    current_pred = model.predict(preprocessor.transform(instance))[0]
    desired_class = 1 - int(current_pred)

    counterfactuals = generator.generate(instance, n=2, desired_class=desired_class)
    cf_df = pd.DataFrame(counterfactuals)[CATEGORICAL_COLS + NUMERIC_COLS]
    cf_preds = model.predict(preprocessor.transform(cf_df))

    assert (cf_preds == desired_class).all()


def test_features_to_vary_restricts_changed_columns(fitted_pipeline) -> None:
    # restricting the search to "income" only must never touch city or
    # loan_amount -- this is what keeps counterfactuals actionable for
    # features a model uses but a person can't realistically change.
    model, preprocessor, df = fitted_pipeline
    generator = CounterfactualGenerator().fit(
        model, preprocessor, df, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
    )

    instance = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[[0]]
    current_pred = model.predict(preprocessor.transform(instance))[0]
    desired_class = 1 - int(current_pred)

    counterfactuals = generator.generate(
        instance, n=2, desired_class=desired_class, features_to_vary=["income"]
    )

    assert len(counterfactuals) > 0
    for cf in counterfactuals:
        assert cf["city"] == instance.iloc[0]["city"]
        assert cf["loan_amount"] == pytest.approx(instance.iloc[0]["loan_amount"])


def test_generate_returns_empty_list_when_no_counterfactual_exists(fitted_pipeline) -> None:
    # Force "city" to have exactly zero influence by zeroing its one-hot
    # coefficients directly, then pick the most confidently-classified
    # row -- guarantees no city value can ever flip the prediction. DiCE
    # raises UserConfigValidationException when it finds zero
    # counterfactuals (distinct from finding fewer than `n`), which
    # generate() must catch and translate to [] instead of propagating.
    model, preprocessor, df = fitted_pipeline
    X_transformed = preprocessor.transform(df[CATEGORICAL_COLS + NUMERIC_COLS])
    city_col_indices = [i for i, name in enumerate(X_transformed.columns) if name.startswith("categorical__city")]
    model.coef_[0, city_col_indices] = 0.0

    generator = CounterfactualGenerator().fit(
        model, preprocessor, df, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
    )

    probabilities = model.predict_proba(X_transformed)[:, 1]
    most_confident_idx = int(np.argmax(np.abs(probabilities - 0.5)))
    instance = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[[most_confident_idx]]
    current_pred = model.predict(preprocessor.transform(instance))[0]
    desired_class = 1 - int(current_pred)

    counterfactuals = generator.generate(
        instance, n=2, desired_class=desired_class, features_to_vary=["city"]
    )
    assert counterfactuals == []
