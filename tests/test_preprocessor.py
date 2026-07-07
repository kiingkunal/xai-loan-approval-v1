"""Tests for LoanDataPreprocessor invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_loan.data.preprocessor import LoanDataPreprocessor, split_features_target

CATEGORICAL_COLS = ["color", "city"]
NUMERIC_COLS = ["income", "age"]


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green", None], size=n),
            "city": rng.choice(["delhi", "mumbai", None], size=n),
            "income": rng.normal(50_000, 10_000, size=n),
            "age": rng.integers(20, 60, size=n).astype(float),
            "target": rng.integers(0, 2, size=n),
        }
    )
    df.loc[0, "income"] = np.nan
    df.loc[1, "age"] = np.nan
    return df


def test_split_features_target_removes_target_from_features() -> None:
    df = _make_df()
    X, y = split_features_target(df, target_col="target")
    assert "target" not in X.columns
    assert y.name == "target"
    assert len(X) == len(y) == len(df)


def test_fit_returns_self_for_chaining() -> None:
    df = _make_df()
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    result = preprocessor.fit(df)
    assert result is preprocessor


def test_transform_before_fit_raises() -> None:
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    with pytest.raises(RuntimeError):
        preprocessor.transform(_make_df())


def test_fit_transform_has_no_missing_values() -> None:
    df = _make_df()
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    transformed = preprocessor.fit_transform(df)
    assert not transformed.isna().any().any()


def test_fit_transform_preserves_row_count_and_index() -> None:
    df = _make_df()
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    transformed = preprocessor.fit_transform(df)
    assert len(transformed) == len(df)
    assert list(transformed.index) == list(df.index)


def test_transform_handles_unseen_category_without_raising() -> None:
    train_df = _make_df()
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    preprocessor.fit(train_df)

    live_applicant = pd.DataFrame(
        {
            "color": ["purple"],  # never seen during fit
            "city": ["delhi"],
            "income": [60_000.0],
            "age": [35.0],
        }
    )
    transformed = preprocessor.transform(live_applicant)
    assert len(transformed) == 1
    assert not transformed.isna().any().any()


def test_numeric_columns_are_scaled() -> None:
    df = _make_df(n=200)
    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    transformed = preprocessor.fit_transform(df)
    numeric_out_cols = [c for c in transformed.columns if c.startswith("numeric__")]
    assert len(numeric_out_cols) == len(NUMERIC_COLS)
    for col in numeric_out_cols:
        assert abs(transformed[col].mean()) < 1e-6
        assert abs(transformed[col].std(ddof=0) - 1.0) < 1e-6


def test_apply_smote_balances_classes() -> None:
    rng = np.random.default_rng(1)
    n_majority, n_minority = 90, 10
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n_majority + n_minority),
            "f2": rng.normal(size=n_majority + n_minority),
        }
    )
    y = pd.Series([0] * n_majority + [1] * n_minority, name="target")

    preprocessor = LoanDataPreprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    X_resampled, y_resampled = preprocessor.apply_smote(X, y)

    counts = y_resampled.value_counts()
    assert counts[0] == counts[1]
    assert len(X_resampled) == len(y_resampled)
