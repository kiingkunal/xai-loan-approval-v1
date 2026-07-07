"""Tests for the trust score's bounds and component logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_loan.trust.score import compute_trust_score

TRUST_SCORE_AUTO_DECISION_THRESHOLD = 80.0


class _StubModel:
    """Fake model returning a fixed probability pair, isolating trust
    score math from any real model/preprocessing."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, instance: pd.DataFrame) -> np.ndarray:
        return np.tile(self._probabilities, (len(instance), 1))


INSTANCE = pd.DataFrame({"income": [50_000.0], "loan_amount": [20_000.0], "city": ["delhi"]})

CONFIDENT_SHAP = {"feature_names": ["income", "loan_amount", "city"], "shap_values": [0.5, 0.3, 0.1]}
CONFIDENT_LIME = {"feature_names": ["income", "loan_amount", "city"], "weights": [0.4, 0.2, 0.05]}

# 6 features so the top-5 window genuinely excludes one feature each —
# with only 3 features (as above), top-5 captures the full set regardless
# of order, so Jaccard can't distinguish "agree" from "disagree". LIME's
# feature_names list is pre-sorted by descending |weight| (matching what
# LIMEExplainer.local_explanation actually returns).
SHAP_6 = {
    "feature_names": ["f1", "f2", "f3", "f4", "f5", "f6"],
    "shap_values": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
}
AGREEING_LIME_6 = {"feature_names": ["f1", "f2", "f3", "f4", "f5", "f6"], "weights": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
# top-5 = {f6,f5,f4,f3,f2}, excludes f1 -- SHAP_6's top-5 excludes f6 instead.
DISAGREEING_LIME_6 = {"feature_names": ["f6", "f5", "f4", "f3", "f2", "f1"], "weights": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

EASY_COUNTERFACTUAL = [{"income": 51_000.0, "loan_amount": 20_000.0, "city": "delhi"}]
HARD_COUNTERFACTUAL = [{"income": 500_000.0, "loan_amount": 20_000.0, "city": "delhi"}]


def test_score_and_components_are_bounded() -> None:
    model = _StubModel([0.3, 0.7])
    result = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, EASY_COUNTERFACTUAL)

    assert 0.0 <= result["score"] <= 100.0
    for value in result["components"].values():
        assert 0.0 <= value <= 1.0


def test_verdict_is_one_of_the_three_allowed_values() -> None:
    model = _StubModel([0.3, 0.7])
    result = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, EASY_COUNTERFACTUAL)
    assert result["verdict"] in {"auto-approve", "auto-reject", "send-to-human-review"}


def test_high_confidence_agreement_and_easy_counterfactual_gives_high_score() -> None:
    # max probability 0.99 (very confident), SHAP/LIME agree exactly,
    # counterfactual only needs a tiny 2% income change -> should clear
    # the auto-decision threshold.
    model = _StubModel([0.01, 0.99])
    result = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, EASY_COUNTERFACTUAL)

    assert result["score"] >= 80.0
    assert result["verdict"] == "auto-reject"  # predicted_class=1 at >=80 score


def test_boundary_confidence_gives_zero_confidence_component() -> None:
    # probability exactly 0.5 = sitting on the decision boundary -> the
    # least trustworthy a prediction can be, must score 0, not 0.5.
    model = _StubModel([0.5, 0.5])
    result = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, EASY_COUNTERFACTUAL)
    assert result["components"]["model_confidence"] == pytest.approx(0.0)


def test_disagreeing_explainers_lower_agreement_component() -> None:
    model = _StubModel([0.3, 0.7])
    agreeing = compute_trust_score(INSTANCE, model, SHAP_6, AGREEING_LIME_6, EASY_COUNTERFACTUAL)
    disagreeing = compute_trust_score(INSTANCE, model, SHAP_6, DISAGREEING_LIME_6, EASY_COUNTERFACTUAL)

    assert agreeing["components"]["shap_lime_agreement"] == pytest.approx(1.0)
    assert disagreeing["components"]["shap_lime_agreement"] == pytest.approx(4 / 6)
    assert disagreeing["components"]["shap_lime_agreement"] < agreeing["components"]["shap_lime_agreement"]


def test_unrealistic_counterfactual_lowers_feasibility_component() -> None:
    model = _StubModel([0.3, 0.7])
    easy = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, EASY_COUNTERFACTUAL)
    hard = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, HARD_COUNTERFACTUAL)

    assert hard["components"]["counterfactual_feasibility"] < easy["components"]["counterfactual_feasibility"]


def test_no_counterfactuals_gives_zero_feasibility() -> None:
    model = _StubModel([0.3, 0.7])
    result = compute_trust_score(INSTANCE, model, CONFIDENT_SHAP, CONFIDENT_LIME, [])
    assert result["components"]["counterfactual_feasibility"] == 0.0


def test_low_score_triggers_human_review_with_reason() -> None:
    # near-boundary confidence + disagreement + no counterfactual -> low score
    model = _StubModel([0.51, 0.49])
    result = compute_trust_score(INSTANCE, model, SHAP_6, DISAGREEING_LIME_6, [])

    assert result["score"] < TRUST_SCORE_AUTO_DECISION_THRESHOLD
    assert result["verdict"] == "send-to-human-review"
    assert "human review" in result["reason"].lower()
