"""Preprocessing pipeline shared by training and live inference.

`LoanDataPreprocessor` learns its encoders/scalers once on training data
(`fit`) and applies that exact same transform to test data or a single
live applicant (`transform`). Fitting on training data only — never on
test or live data — is what prevents leakage: a live applicant must never
influence what "average income" or "known categories" mean.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xai_loan.utils.config import RANDOM_STATE


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a feature matrix and target series.

    Args:
        df: DataFrame containing both features and the target column.
        target_col: Name of the target column.

    Returns:
        A tuple of ``(X, y)`` where ``X`` is ``df`` without ``target_col``
        and ``y`` is the target column as a Series.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


class LoanDataPreprocessor:
    """Fits encoders/scalers on training data and applies them consistently.

    Attributes:
        categorical_cols: Categorical feature columns to one-hot encode.
        numeric_cols: Numeric feature columns to impute and scale.
        random_state: Seed used by SMOTE resampling.
    """

    def __init__(
        self,
        categorical_cols: list[str],
        numeric_cols: list[str],
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Initialize the preprocessor with the feature columns it will see.

        Args:
            categorical_cols: Names of categorical feature columns.
            numeric_cols: Names of numeric feature columns.
            random_state: Seed for SMOTE resampling.
        """
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.random_state = random_state
        self._column_transformer: ColumnTransformer | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        """Learn imputers/encoders/scalers from training features only.

        Args:
            df: Training feature DataFrame (no target column required;
                if present, it is simply ignored).

        Returns:
            self, for chaining (e.g. ``preprocessor.fit(X_train)``).
        """
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, self.categorical_cols),
                ("numeric", numeric_pipeline, self.numeric_cols),
            ]
        )
        self._column_transformer.fit(df[self.categorical_cols + self.numeric_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted transform to new data.

        Works identically for a held-out test set or a single live
        applicant row, since both just need the categories/scaling
        learned during `fit`. Unknown categorical values are ignored
        (encoded as all-zero) rather than raising, since a live
        applicant can supply a category never seen in training.

        Args:
            df: Feature DataFrame containing at least the columns this
                preprocessor was fitted on. Extra columns (e.g. an ID
                column or the target) are ignored.

        Returns:
            Transformed feature DataFrame with one-hot + scaled columns,
            named via the fitted transformer's output feature names, and
            the original row index preserved.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self._column_transformer is None:
            raise RuntimeError("LoanDataPreprocessor.transform() called before fit().")

        transformed = self._column_transformer.transform(df[self.categorical_cols + self.numeric_cols])
        feature_names = self._column_transformer.get_feature_names_out()
        return pd.DataFrame(transformed, columns=feature_names, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on `df` and immediately transform it.

        Args:
            df: Training feature DataFrame.

        Returns:
            Transformed feature DataFrame, identical to calling `fit`
            followed by `transform` on the same data.
        """
        return self.fit(df).transform(df)

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Resample the (already-transformed) training set with SMOTE.

        Only ever call this on training data — applying SMOTE to test or
        live data would synthesize fake evaluation/applicant rows.

        Args:
            X: Transformed numeric feature matrix (output of `transform`).
            y: Training target labels aligned with `X`.

        Returns:
            A tuple of ``(X_resampled, y_resampled)`` with the minority
            class oversampled to match the majority class.
        """
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(
            np.asarray(y_resampled), name=y.name
        )
