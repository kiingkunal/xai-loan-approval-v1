"""Trust score — NOVEL CONTRIBUTION #2.

The system-level problem this solves: SHAP, LIME, and DiCE each produce
an explanation for *every* prediction, regardless of whether that
prediction is actually trustworthy. A loan officer staring at five
different explanation widgets for a borderline case has no single
number telling them "this one needs your judgment, that one doesn't."
The trust score collapses three independent signals — is the model
actually confident, do two unrelated explanation methods agree on what
mattered, and is there a realistic way to change the outcome — into one
0-100 number and a routing verdict, so the dashboard can default to
auto-deciding the easy cases and flagging the hard ones for a human.

Design rationale per component:

- ``model_confidence``: a probability of exactly 0.5 means the model is
  sitting exactly on the decision boundary — that's the *least*
  trustworthy a prediction can be, so it must score 0, not 0.5. Rescaling
  ``[0.5, 1.0] -> [0.0, 1.0]`` makes that explicit instead of leaving a
  raw probability that looks "halfway confident" at the boundary.

- ``shap_lime_agreement``: SHAP and LIME compute feature importance via
  completely different mechanisms (exact game-theoretic attribution vs.
  local linear surrogate fitting). If they agree on which features drove
  the decision, that's evidence the explanation reflects something real
  about the model rather than an artifact of one method. Measured as
  Jaccard overlap of each method's top-5 features — simpler and more
  interpretable in a viva than a rank-correlation coefficient, and
  doesn't assume either ranking is linearly comparable.

- ``counterfactual_feasibility``: a counterfactual that requires
  tripling the applicant's income is technically a valid DiCE output but
  practically useless — it doesn't represent a real path to a different
  decision. Scoring ``1 / (1 + max_relative_change)`` punishes large
  required changes smoothly (no value, however large, ever resets the
  score below 0) and takes the *best* (most feasible) counterfactual
  found, since trust hinges on whether *any* realistic path exists, not
  whether all of them are realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_loan.utils.config import TRUST_SCORE_THRESHOLDS, TRUST_SCORE_WEIGHTS

_TOP_K_FOR_AGREEMENT = 5

_WEAK_COMPONENT_LABELS: dict[str, str] = {
    "model_confidence": "the model's prediction is close to the decision boundary",
    "shap_lime_agreement": "SHAP and LIME disagree on which features matter most",
    "counterfactual_feasibility": "no small, realistic change was found that would flip the decision",
}


def _model_confidence(model: object, instance: pd.DataFrame) -> tuple[float, int]:
    """Compute a [0, 1] confidence score and the predicted class.

    Args:
        model: A classifier exposing `predict_proba` over `instance`'s
            feature representation — either a bare scikit-learn model
            (if `instance` is already preprocessed) or a
            `xai_loan.models.pipeline.PipelineModel` (if `instance` is
            raw, which is required for `counterfactual_feasibility`'s
            diffing against raw counterfactuals to make sense).
        instance: A single-row DataFrame.

    Returns:
        A tuple of ``(confidence, predicted_class)`` where confidence is
        0 at the decision boundary (probability 0.5) and 1 when the
        model assigns full probability to one class.
    """
    probabilities = model.predict_proba(instance)[0]
    predicted_class = int(np.argmax(probabilities))
    max_proba = float(probabilities[predicted_class])
    confidence = max(0.0, 2.0 * (max_proba - 0.5))
    return confidence, predicted_class


def _shap_lime_agreement(
    shap_exp: dict[str, object], lime_exp: dict[str, object], top_k: int = _TOP_K_FOR_AGREEMENT
) -> float:
    """Jaccard similarity of the top-K most influential features per method.

    Args:
        shap_exp: Output of `SHAPExplainer.local_explanation`.
        lime_exp: Output of `LIMEExplainer.local_explanation`.
        top_k: How many top features per method to compare.

    Returns:
        A value in [0, 1]: 1 if both methods name the exact same top-K
        feature set, 0 if they share none.
    """
    shap_values = np.asarray(shap_exp["shap_values"], dtype=float)
    shap_order = np.argsort(-np.abs(shap_values))[:top_k]
    shap_top = {shap_exp["feature_names"][i] for i in shap_order}

    # LIMEExplainer already returns features sorted by descending |weight|.
    lime_top = set(lime_exp["feature_names"][:top_k])

    if not shap_top and not lime_top:
        return 0.0
    return len(shap_top & lime_top) / len(shap_top | lime_top)


def _counterfactual_feasibility(instance: pd.DataFrame, counterfactuals: list[dict[str, object]]) -> float:
    """Score how small/realistic the best available counterfactual is.

    Args:
        instance: The original single-row raw applicant DataFrame.
        counterfactuals: Output of `CounterfactualGenerator.generate`
            (raw feature dicts); may be empty.

    Returns:
        0.0 if no counterfactual was found at all. Otherwise the best
        (highest) feasibility across all provided counterfactuals, where
        each counterfactual's feasibility is
        ``1 / (1 + max_relative_change)`` over its changed features.
    """
    if not counterfactuals:
        return 0.0

    original = instance.iloc[0]
    best_feasibility = 0.0
    for counterfactual in counterfactuals:
        max_relative_change = 0.0
        for feature, new_value in counterfactual.items():
            old_value = original.get(feature)
            if old_value is None or old_value == new_value:
                continue
            if (
                isinstance(old_value, (int, float))
                and isinstance(new_value, (int, float))
                and old_value != 0
            ):
                relative_change = abs(new_value - old_value) / abs(old_value)
            else:
                relative_change = 1.0  # categorical change, or change from zero
            max_relative_change = max(max_relative_change, relative_change)
        feasibility = 1.0 / (1.0 + max_relative_change)
        best_feasibility = max(best_feasibility, feasibility)
    return best_feasibility


def _build_reason(score: float, verdict: str, components: dict[str, float]) -> str:
    """Compose a short human-readable explanation of the trust score."""
    if verdict != "send-to-human-review":
        return (
            f"Trust score {score:.0f}/100 — model is confident, SHAP and LIME agree "
            "on key drivers, and a realistic counterfactual exists."
        )

    weakest_name, _ = min(components.items(), key=lambda item: item[1])
    weak_label = _WEAK_COMPONENT_LABELS[weakest_name]
    if score < TRUST_SCORE_THRESHOLDS["human_review"]:
        return f"Trust score {score:.0f}/100 — strong human review recommended; {weak_label}."
    return f"Trust score {score:.0f}/100 — routine human review recommended; {weak_label}."


def compute_trust_score(
    instance: pd.DataFrame,
    model: object,
    shap_exp: dict[str, object],
    lime_exp: dict[str, object],
    counterfactuals: list[dict[str, object]],
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compute a 0-100 trust score and routing verdict for one prediction.

    Args:
        instance: A single-row applicant DataFrame, in whatever feature
            representation `model.predict_proba` expects. Since
            `counterfactuals` are always in raw feature units, this
            should be the raw instance with `model` being a
            `xai_loan.models.pipeline.PipelineModel` wrapping the actual
            classifier — the same convention `CounterfactualGenerator`
            uses.
        model: A classifier exposing `predict_proba` over `instance`.
        shap_exp: Output of `SHAPExplainer.local_explanation` for this
            same instance.
        lime_exp: Output of `LIMEExplainer.local_explanation` for this
            same instance.
        counterfactuals: Output of `CounterfactualGenerator.generate` for
            this same instance (may be empty).
        weights: Override for the three component weights. Defaults to
            `xai_loan.utils.config.TRUST_SCORE_WEIGHTS` (0.4 / 0.3 / 0.3).
            Expected to sum to 1.0 so the final score lands in [0, 100].

    Returns:
        A dict with ``score`` (float, 0-100), ``verdict``
        (``"auto-approve"``, ``"auto-reject"``, or
        ``"send-to-human-review"``), ``components`` (dict of the three
        [0, 1] component scores), and ``reason`` (str).
    """
    weights = weights if weights is not None else TRUST_SCORE_WEIGHTS

    confidence, predicted_class = _model_confidence(model, instance)
    agreement = _shap_lime_agreement(shap_exp, lime_exp)
    feasibility = _counterfactual_feasibility(instance, counterfactuals)

    components = {
        "model_confidence": confidence,
        "shap_lime_agreement": agreement,
        "counterfactual_feasibility": feasibility,
    }
    score = 100.0 * sum(weights[name] * value for name, value in components.items())

    if score >= TRUST_SCORE_THRESHOLDS["auto_decision"]:
        verdict = "auto-reject" if predicted_class == 1 else "auto-approve"
    else:
        verdict = "send-to-human-review"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "components": components,
        "reason": _build_reason(score, verdict, components),
    }
