"""SHAP-based global and local explanations.

SHAP's raw output shape depends on both the explainer type and the model:
`TreeExplainer` on `RandomForestClassifier` returns a 3D array
``(n_samples, n_features, n_classes)``, but on `XGBClassifier` it returns
a 2D array ``(n_samples, n_features)`` that is already oriented toward
the positive class (XGBoost's native binary objective has a single
output). `KernelExplainer` (used for non-tree models) returns 3D like
the RF case. `_positive_class_values` normalizes all of these into one
consistent ``(n_samples, n_features)`` array for the positive class
(label 1 = loan rejected), so the rest of this class doesn't need to
know which explainer produced the values.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

_TREE_MODEL_TYPES = (RandomForestClassifier, XGBClassifier, DecisionTreeClassifier)


def _positive_class_values(raw_shap_values: object) -> np.ndarray:
    """Normalize raw SHAP output to a 2D positive-class array.

    Args:
        raw_shap_values: Whatever `Explainer.shap_values` returned —
            a list of per-class arrays (legacy API), a 3D array
            ``(n_samples, n_features, n_classes)``, or a 2D array
            already oriented toward a single output.

    Returns:
        A ``(n_samples, n_features)`` array of SHAP values for the
        positive class.
    """
    if isinstance(raw_shap_values, list):
        return np.asarray(raw_shap_values[1])
    values = np.asarray(raw_shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _positive_class_base_value(expected_value: object) -> float:
    """Normalize a SHAP explainer's `expected_value` to a single float.

    Args:
        expected_value: Either a scalar (single-output models like
            XGBoost) or a per-class array (e.g. ``[base0, base1]``).

    Returns:
        The base value for the positive class.
    """
    values = np.atleast_1d(np.asarray(expected_value, dtype=float))
    return float(values[1]) if values.size > 1 else float(values[0])


class SHAPExplainer:
    """Wraps SHAP's TreeExplainer/KernelExplainer behind one consistent API.

    Attributes:
        feature_names: Column names of the background data, in order.
    """

    def __init__(self, kernel_nsamples: int = 100) -> None:
        """Initialize the explainer.

        Args:
            kernel_nsamples: Number of perturbation samples used by
                `KernelExplainer` for non-tree models (ignored for tree
                models, which compute exact values). Higher is more
                accurate but slower.
        """
        self.kernel_nsamples = kernel_nsamples
        self.feature_names: list[str] = []
        self._explainer: shap.TreeExplainer | shap.KernelExplainer | None = None
        self._is_tree_model = False
        self._X_background: pd.DataFrame | None = None

    def fit(self, model: object, X_background: pd.DataFrame) -> Self:
        """Initialize a TreeExplainer or KernelExplainer for `model`.

        Args:
            model: A fitted classifier exposing `predict_proba`.
                Tree-based models (`RandomForestClassifier`,
                `XGBClassifier`, `DecisionTreeClassifier`) get the exact,
                fast `TreeExplainer`. Everything else (e.g.
                `LogisticRegression`) gets the model-agnostic but slower
                `KernelExplainer`.
            X_background: Reference data the explainer measures
                deviations against — typically a sample of the training
                set.

        Returns:
            self, for chaining.
        """
        self.feature_names = list(X_background.columns)
        self._X_background = X_background
        self._is_tree_model = isinstance(model, _TREE_MODEL_TYPES)
        if self._is_tree_model:
            self._explainer = shap.TreeExplainer(model, X_background)
        else:
            self._explainer = shap.KernelExplainer(model.predict_proba, X_background)
        return self

    def _raw_shap_values(self, X: pd.DataFrame) -> object:
        if self._explainer is None:
            raise RuntimeError("SHAPExplainer method called before fit().")
        if self._is_tree_model:
            # check_additivity=False sidesteps a known floating-point
            # quirk where XGBoost's margin output and the sum of SHAP
            # values disagree by a tiny epsilon on some samples.
            return self._explainer.shap_values(X, check_additivity=False)
        return self._explainer.shap_values(X, nsamples=self.kernel_nsamples)

    def global_importance(self) -> pd.DataFrame:
        """Rank features by mean absolute SHAP value over the background set.

        Returns:
            A DataFrame with columns ``feature`` and ``mean_abs_shap``,
            sorted descending by importance.
        """
        if self._X_background is None:
            raise RuntimeError("SHAPExplainer.global_importance() called before fit().")
        values = _positive_class_values(self._raw_shap_values(self._X_background))
        mean_abs = np.abs(values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(self, instance: pd.DataFrame) -> dict[str, object]:
        """Explain a single prediction in terms of per-feature SHAP values.

        Args:
            instance: A single-row DataFrame with the same columns the
                explainer was fitted on.

        Returns:
            A dict with ``feature_names`` (list[str]), ``shap_values``
            (list[float], same order as ``feature_names``), and
            ``base_value`` (float, the model's average output before
            seeing any features).
        """
        values = _positive_class_values(self._raw_shap_values(instance))[0]
        base_value = _positive_class_base_value(self._explainer.expected_value)
        return {
            "feature_names": list(self.feature_names),
            "shap_values": values.tolist(),
            "base_value": base_value,
        }

    def plot_summary(self) -> None:
        """Render a SHAP summary (beeswarm) plot of the background set.

        Draws to the current matplotlib figure; the caller decides
        whether to `plt.show()` or save it.
        """
        if self._X_background is None:
            raise RuntimeError("SHAPExplainer.plot_summary() called before fit().")
        values = _positive_class_values(self._raw_shap_values(self._X_background))
        shap.summary_plot(values, self._X_background, feature_names=self.feature_names, show=False)

    def plot_waterfall(self, instance: pd.DataFrame) -> None:
        """Render a SHAP waterfall plot for one prediction.

        Args:
            instance: A single-row DataFrame with the same columns the
                explainer was fitted on.
        """
        explanation_dict = self.local_explanation(instance)
        explanation = shap.Explanation(
            values=np.asarray(explanation_dict["shap_values"]),
            base_values=explanation_dict["base_value"],
            data=instance.iloc[0].to_numpy(),
            feature_names=explanation_dict["feature_names"],
        )
        shap.plots.waterfall(explanation, show=False)
