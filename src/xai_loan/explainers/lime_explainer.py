"""LIME-based local explanations, used to cross-check SHAP.

Returns feature contributions keyed by feature name (via LIME's
`as_map()`, not `as_list()`'s human-readable strings like
"credit_amount <= 1000.00") so they line up index-for-index with SHAP's
output — required for the trust score's SHAP/LIME rank-agreement check.
"""

from __future__ import annotations

from typing import Self

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from xai_loan.utils.config import RANDOM_STATE

_POSITIVE_CLASS_LABEL = 1


class LIMEExplainer:
    """Wraps `lime.lime_tabular.LimeTabularExplainer` for tabular data.

    Attributes:
        feature_names: Column names of the background data, in order.
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        """Initialize the explainer.

        Args:
            random_state: Seed for LIME's internal perturbation sampling.
        """
        self.random_state = random_state
        self.feature_names: list[str] = []
        self._model: object | None = None
        self._explainer: LimeTabularExplainer | None = None

    def fit(self, model: object, X_background: pd.DataFrame) -> Self:
        """Initialize the LIME explainer on a representative background sample.

        Args:
            model: A fitted classifier exposing `predict_proba`.
            X_background: Reference data LIME uses to learn per-feature
                value distributions for perturbation sampling.

        Returns:
            self, for chaining.
        """
        self.feature_names = list(X_background.columns)
        self._model = model
        self._explainer = LimeTabularExplainer(
            X_background.to_numpy(),
            feature_names=self.feature_names,
            class_names=["approve", "reject"],
            mode="classification",
            random_state=self.random_state,
        )
        return self

    def local_explanation(self, instance: pd.DataFrame, num_features: int = 10) -> dict[str, object]:
        """Explain a single prediction in terms of per-feature contributions.

        Args:
            instance: A single-row DataFrame with the same columns the
                explainer was fitted on.
            num_features: Maximum number of top features LIME reports.

        Returns:
            A dict with ``feature_names`` (list[str]) and ``weights``
            (list[float], same order as ``feature_names``, sorted by
            descending absolute contribution) and ``intercept`` (float).
        """
        if self._explainer is None or self._model is None:
            raise RuntimeError("LIMEExplainer.local_explanation() called before fit().")

        explanation = self._explainer.explain_instance(
            instance.iloc[0].to_numpy(),
            self._model.predict_proba,
            num_features=num_features,
            labels=(_POSITIVE_CLASS_LABEL,),
        )
        feature_map = explanation.as_map()[_POSITIVE_CLASS_LABEL]
        feature_names = [self.feature_names[idx] for idx, _ in feature_map]
        weights = [weight for _, weight in feature_map]
        intercept = float(explanation.intercept[_POSITIVE_CLASS_LABEL])

        return {
            "feature_names": feature_names,
            "weights": weights,
            "intercept": intercept,
        }
