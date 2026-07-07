"""Fairness auditing and bias mitigation using Fairlearn metrics."""

from __future__ import annotations

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
)

_AGE_CUTOFF_YEARS = 25.0


def bucket_age_band(age_years: pd.Series, cutoff: float = _AGE_CUTOFF_YEARS) -> pd.Series:
    """Bucket continuous age into under/over a cutoff, in years.

    Fairlearn's group-fairness metrics expect a small number of discrete
    groups, not a unique value per row — and 25 is the cutoff commonly
    used in German Credit fairness studies specifically because it's
    where this dataset's age-related approval disparity concentrates.

    Args:
        age_years: Continuous age values, in years.
        cutoff: Age boundary between the two bands.

    Returns:
        A Series of ``"under_25"`` / ``"25_or_over"``-style string labels
        (named using the actual `cutoff`), same index as `age_years`.
    """
    cutoff_label = int(cutoff) if cutoff == int(cutoff) else cutoff
    return age_years.apply(lambda age: f"under_{cutoff_label}" if age < cutoff else f"{cutoff_label}_or_over")


def sensitive_feature_series(
    protected_cols: list[str], X: pd.DataFrame, cutoff: float = _AGE_CUTOFF_YEARS
) -> list[tuple[str, pd.Series]]:
    """Build a (label, group-membership-series) pair per protected column.

    Continuous protected columns are bucketed into bands first via
    `bucket_age_band`: raw ``age`` (already in years, German Credit) is
    bucketed directly; Home Credit's ``DAYS_BIRTH`` is *negative* days
    since birth, so it's converted to positive years first. Already-
    categorical protected columns (``sex``, ``CODE_GENDER``) pass through
    unchanged.

    Args:
        protected_cols: Column names from a dataset's metadata
            (`DatasetMetadata["protected_cols"]`).
        X: Raw (un-preprocessed) feature DataFrame containing those columns.
        cutoff: Age boundary passed through to `bucket_age_band`.

    Returns:
        A list of ``(column_name, sensitive_series)`` pairs, one per
        protected column, ready to pass as `FairnessAuditor.audit`'s
        `sensitive_features` argument.
    """
    pairs = []
    for col in protected_cols:
        if col == "age":
            pairs.append((col, bucket_age_band(X[col], cutoff)))
        elif col == "DAYS_BIRTH":
            pairs.append((col, bucket_age_band(-X[col] / 365.25, cutoff)))
        else:
            pairs.append((col, X[col]))
    return pairs


class FairnessAuditor:
    """Audits a trained classifier for disparities across a protected attribute."""

    def audit(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.Series,
    ) -> dict[str, object]:
        """Compute group-fairness metrics for a fitted classifier.

        Args:
            model: A fitted classifier exposing `predict`.
            X: Feature matrix to score (already preprocessed, matching
                what `model` was trained on).
            y: True labels aligned with `X`.
            sensitive_features: Protected attribute values aligned with
                `X` (e.g. ``sex`` or an age band), used to group rows.

        Returns:
            A dict with ``demographic_parity_difference`` and
            ``equalized_odds_difference`` (both in [0, 1], 0 = no
            disparity between groups), ``false_positive_rate_by_group``
            (dict keyed by group value), ``false_positive_rate_parity_difference``
            (max minus min across groups), and ``selection_rate_by_group``.
        """
        y_pred = model.predict(X)

        frame = MetricFrame(
            metrics={"selection_rate": selection_rate, "false_positive_rate": false_positive_rate},
            y_true=y,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        fpr_by_group = frame.by_group["false_positive_rate"]

        return {
            "demographic_parity_difference": demographic_parity_difference(
                y, y_pred, sensitive_features=sensitive_features
            ),
            "equalized_odds_difference": equalized_odds_difference(
                y, y_pred, sensitive_features=sensitive_features
            ),
            "false_positive_rate_by_group": fpr_by_group.to_dict(),
            "false_positive_rate_parity_difference": float(fpr_by_group.max() - fpr_by_group.min()),
            "selection_rate_by_group": frame.by_group["selection_rate"].to_dict(),
        }

    def apply_reweighing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.Series,
    ) -> pd.Series:
        """Compute per-row sample weights that mitigate demographic-parity bias.

        Implements the Kamiran & Calders (2012) reweighing formula:
        each row's weight is ``P(group) * P(label) / P(group, label)``.
        Weighting this way exactly equalizes the *weighted* label
        distribution across groups without altering or duplicating any
        row (unlike SMOTE, which synthesizes new rows) — pass the result
        as `sample_weight` to a classifier's `fit` to mitigate bias
        learned from the training data.

        Args:
            X: Training feature matrix — only used to validate it's the
                same length as `y` and `sensitive_features`; the weight
                formula itself depends only on labels and group
                membership.
            y: Training labels.
            sensitive_features: Protected attribute values aligned with
                `y`.

        Returns:
            A Series of per-row weights, same index as `y`. Rows from a
            (group, label) combination that's underrepresented relative
            to the group's and label's overall marginal frequencies get
            a weight above 1; overrepresented combinations get a weight
            below 1.

        Raises:
            ValueError: If `X`, `y`, and `sensitive_features` aren't all
                the same length.
        """
        if not (len(X) == len(y) == len(sensitive_features)):
            raise ValueError("X, y, and sensitive_features must have the same length.")

        n = len(y)
        weights = pd.Series(1.0, index=y.index)
        for group_value in sensitive_features.unique():
            for label_value in y.unique():
                mask = (sensitive_features == group_value).to_numpy() & (y == label_value).to_numpy()
                joint_prob = mask.sum() / n
                if joint_prob == 0:
                    continue
                group_prob = (sensitive_features == group_value).sum() / n
                label_prob = (y == label_value).sum() / n
                weights[mask] = (group_prob * label_prob) / joint_prob
        return weights
