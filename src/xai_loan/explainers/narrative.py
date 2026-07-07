"""Plain-English explanation generator.

The system-level idea: SHAP values and DiCE counterfactuals are accurate
but not something a loan applicant (or most loan officers) can read
directly — "categorical__checking_status_A11 contributed +0.34 to the
log-odds" means nothing to a non-technical reader. This module turns
those numbers into a deterministic sentence a human can act on.

The generator is deliberately template-based: it uses already-computed SHAP
values and counterfactuals, performs no external calls, and produces the same
sentence for the same inputs every time. That makes the explanation auditable,
testable, and safe to display beside the model output.

When `categorical_cols`/`dataset` are supplied, feature names are run
through `xai_loan.data.feature_labels` for proper humanization (German
Credit's ``A11``-style codes resolved to their documented meaning) *and*
restricted to `feature_labels.NARRATIVE_FEATURE_ALLOWLIST` — a model
like Home Credit's genuinely uses ~119 features, many of them technical
(building statistics, document-submission flags) with no real-world
actionable meaning to an applicant. The narrative only ever cites the
same human-meaningful fields the Predict form asks for; the full
technical SHAP/LIME breakdown remains available, unrestricted, in the
dashboard's Explain tab for analysts. Without `categorical_cols`/
`dataset` (the original two-argument call), this degrades to a generic
underscore-stripped humanizer with no filtering — kept for backward
compatibility with callers that don't have dataset context.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from xai_loan.data import feature_labels

_MAX_TOP_FACTORS = 3
_MAX_COUNTERFACTUAL_PHRASES = 2


def _humanize_feature_name(transformed_name: str) -> str:
    """Generic fallback humanizer, used when no dataset context is given.

    Args:
        transformed_name: A column name from the preprocessed feature
            matrix (i.e. `LoanDataPreprocessor`'s output columns).

    Returns:
        An underscore-stripped phrase, e.g. ``"checking status A11"`` —
        readable but not translated, since there's no dataset context
        to look up a code's documented meaning.
    """
    name = re.sub(r"^(categorical|numeric)__", "", transformed_name)
    return name.replace("_", " ")


def _humanize(transformed_name: str, categorical_cols: list[str] | None, dataset: str | None) -> str:
    """Dispatch to the dataset-aware humanizer if context is available."""
    if dataset is not None and categorical_cols is not None:
        return feature_labels.humanize_feature(transformed_name, categorical_cols, dataset)
    return _humanize_feature_name(transformed_name)


def _top_contributing_factors(
    shap_explanation: dict[str, object],
    prediction: int,
    categorical_cols: list[str] | None = None,
    dataset: str | None = None,
) -> list[tuple[str, float]]:
    """Find the features that most support the decision actually made.

    SHAP values here are oriented toward the positive class (label 1 =
    reject): a positive value pushes toward rejection, negative pushes
    toward approval. To describe "why this decision", we want factors
    pushing in the *same direction as the decision made* — so for a
    rejection we want the most positive values, but for an approval we
    want the most negative ones. Multiplying by a direction sign lets
    one code path handle both cases instead of duplicating logic.

    When `dataset` is given, candidates are first restricted to
    `feature_labels.NARRATIVE_FEATURE_ALLOWLIST[dataset]` — see module
    docstring for why.

    Args:
        shap_explanation: Output of `SHAPExplainer.local_explanation`.
        prediction: The predicted class (0 = approve, 1 = reject).
        categorical_cols: The dataset's raw categorical column names,
            needed to resolve transformed feature names back to their
            raw column for the allowlist check.
        dataset: ``"german"`` or ``"home_credit"``. If None, no
            allowlist filtering is applied.

    Returns:
        Up to `_MAX_TOP_FACTORS` ``(feature_name, percent_share)`` pairs,
        sorted by descending contribution, where percentages sum to
        (at most) 100 across *all* allowed contributing factors, not
        just the ones returned.
    """
    feature_names = shap_explanation["feature_names"]
    shap_values = np.asarray(shap_explanation["shap_values"], dtype=float)

    allowed_indices = list(range(len(feature_names)))
    if dataset is not None and categorical_cols is not None:
        allowlist = feature_labels.NARRATIVE_FEATURE_ALLOWLIST.get(dataset)
        if allowlist is not None:
            allowed_indices = [
                i
                for i, name in enumerate(feature_names)
                if feature_labels.split_transformed_feature_name(name, categorical_cols)[0] in allowlist
            ]

    if not allowed_indices:
        return []

    direction = 1.0 if prediction == 1 else -1.0
    aligned_values = shap_values[allowed_indices] * direction
    total_supporting = aligned_values[aligned_values > 0].sum()

    if total_supporting <= 1e-12:
        # Edge case: no allowed feature actually supports the predicted
        # class (can happen near the decision boundary). Fall back to
        # ranking by raw magnitude so the narrative still has something
        # to say.
        order = np.argsort(-np.abs(aligned_values))[:_MAX_TOP_FACTORS]
        return [(feature_names[allowed_indices[i]], 0.0) for i in order]

    order = np.argsort(-aligned_values)
    top_local_indices = [i for i in order if aligned_values[i] > 0][:_MAX_TOP_FACTORS]
    return [
        (feature_names[allowed_indices[i]], float(aligned_values[i] / total_supporting * 100))
        for i in top_local_indices
    ]


def _describe_counterfactual_change(
    instance: pd.DataFrame, counterfactual: dict[str, object], dataset: str | None = None
) -> tuple[str, str] | None:
    """Describe the single most impactful change in one counterfactual.

    When `dataset` is given, only changes to columns in
    `feature_labels.NARRATIVE_FEATURE_ALLOWLIST[dataset]` are considered
    — see module docstring. In practice this rarely matters because
    `CounterfactualGenerator.generate`'s `features_to_vary` should
    already be restricted to the same allowlist at the source; this is a
    defensive second filter, not the primary mechanism.

    Args:
        instance: The original single-row applicant DataFrame (raw,
            un-preprocessed features).
        counterfactual: One counterfactual dict from
            `CounterfactualGenerator.generate` (raw feature values).
        dataset: ``"german"`` or ``"home_credit"``. If None, no
            allowlist filtering or code-to-label translation is applied.

    Returns:
        A ``(raw_col, phrase)`` tuple — e.g. ``("credit_amount", "increase
        loan amount by approximately 2400 (18%)")`` — or None if no
        allowed feature actually changed. `raw_col` lets the caller dedupe
        by *which feature* changed, not just exact phrase text, since
        DiCE's `random` method can return several counterfactuals that
        all land on the same lever with slightly different magnitudes.
    """
    original = instance.iloc[0]
    allowlist = feature_labels.NARRATIVE_FEATURE_ALLOWLIST.get(dataset) if dataset is not None else None

    changes: list[tuple[str, float, object, object]] = []
    for raw_col, new_value in counterfactual.items():
        if allowlist is not None and raw_col not in allowlist:
            continue
        old_value = original[raw_col]
        if pd.isna(old_value) and pd.isna(new_value):
            continue
        if old_value == new_value:
            continue
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)) and old_value != 0:
            relative_change = abs(new_value - old_value) / abs(old_value)
        else:
            relative_change = 1.0  # categorical change, or numeric from zero
        changes.append((raw_col, relative_change, old_value, new_value))

    if not changes:
        return None

    raw_col, relative_change, old_value, new_value = max(changes, key=lambda c: c[1])
    readable_feature = feature_labels.humanize_column(raw_col, dataset) if dataset else raw_col.replace("_", " ")

    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        delta = new_value - old_value
        direction = "increase" if delta > 0 else "reduce"
        phrase = f"{direction} {readable_feature} by approximately {abs(delta):.0f} ({relative_change * 100:.0f}%)"
        return raw_col, phrase

    if dataset is not None:
        old_label = feature_labels.humanize_value(raw_col, old_value, dataset)
        new_label = feature_labels.humanize_value(raw_col, new_value, dataset)
    else:
        old_label, new_label = old_value, new_value
    return raw_col, f"change {readable_feature} from '{old_label}' to '{new_label}'"


def _generate_template_narrative(
    instance: pd.DataFrame,
    prediction: int,
    shap_explanation: dict[str, object],
    counterfactuals: list[dict[str, object]],
    categorical_cols: list[str] | None = None,
    dataset: str | None = None,
) -> str:
    """Build the deterministic, template-based narrative sentence."""
    verb = "rejected" if prediction == 1 else "approved"
    noun = "rejection" if prediction == 1 else "approval"

    top_factors = _top_contributing_factors(shap_explanation, prediction, categorical_cols, dataset)
    factor_phrases = [
        f"{_humanize(name, categorical_cols, dataset)} (contributing {pct:.0f}% to the {noun} score)"
        if pct > 0
        else _humanize(name, categorical_cols, dataset)
        for name, pct in top_factors
    ]

    if not factor_phrases:
        factors_sentence = ""
    elif len(factor_phrases) == 1:
        factors_sentence = f" The top factor was {factor_phrases[0]}."
    else:
        factors_sentence = (
            f" The top factor was {factor_phrases[0]}, followed by "
            + " and ".join(factor_phrases[1:]) + "."
        )

    # Scan every provided counterfactual (not just the first
    # `_MAX_COUNTERFACTUAL_PHRASES`), keeping at most one phrase per
    # underlying feature -- DiCE's `random` method can return several
    # counterfactuals that all land on the same lever with slightly
    # different magnitudes, which without this dedup would read as
    # "either increase income by 257% or increase income by 1882%".
    cf_phrases: list[str] = []
    seen_raw_cols: set[str] = set()
    for cf in counterfactuals:
        described = _describe_counterfactual_change(instance, cf, dataset)
        if described is None:
            continue
        raw_col, phrase = described
        if raw_col in seen_raw_cols:
            continue
        seen_raw_cols.add(raw_col)
        cf_phrases.append(phrase)
        if len(cf_phrases) >= _MAX_COUNTERFACTUAL_PHRASES:
            break
    if len(cf_phrases) == 1:
        cf_sentence = f" To get a different decision, the applicant would need to {cf_phrases[0]}."
    elif cf_phrases:
        cf_sentence = (
            " To get a different decision, the applicant would need to either "
            + " or ".join(cf_phrases)
            + "."
        )
    else:
        cf_sentence = ""

    return f"This application was {verb}.{factors_sentence}{cf_sentence}".strip()


def generate_narrative(
    instance: pd.DataFrame,
    prediction: int,
    shap_explanation: dict[str, object],
    counterfactuals: list[dict[str, object]],
    mode: str = "template",
    categorical_cols: list[str] | None = None,
    dataset: str | None = None,
) -> str:
    """Generate a plain-English explanation of a loan decision.

    Args:
        instance: A single-row raw (un-preprocessed) applicant DataFrame.
        prediction: The predicted class (0 = approve, 1 = reject).
        shap_explanation: Output of `SHAPExplainer.local_explanation` for
            this same instance.
        counterfactuals: Output of `CounterfactualGenerator.generate` for
            this same instance (may be empty).
        mode: Must be ``"template"``. The argument is kept for backward
            compatibility with earlier callers.
        categorical_cols: The dataset's raw categorical column names. If
            given along with `dataset`, feature names are translated to
            their documented human meaning and restricted to
            `xai_loan.data.feature_labels.NARRATIVE_FEATURE_ALLOWLIST`.
            If omitted, falls back to a generic underscore-stripped
            humanizer with no filtering.
        dataset: ``"german"`` or ``"home_credit"``.

    Returns:
        A plain-English narrative string.

    Raises:
        ValueError: If `mode` isn't ``"template"``.
    """
    if mode != "template":
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'template'.")

    return _generate_template_narrative(
        instance, prediction, shap_explanation, counterfactuals, categorical_cols, dataset
    )
