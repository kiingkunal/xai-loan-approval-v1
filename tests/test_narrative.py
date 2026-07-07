"""Tests for the plain-English narrative generator.

Mostly checks determinism (same inputs -> byte-identical output) and that the
generator degrades gracefully when no counterfactuals are available.
"""

from __future__ import annotations

import pandas as pd
import pytest

from xai_loan.explainers.narrative import generate_narrative

INSTANCE = pd.DataFrame(
    {"checking_status": ["A11"], "credit_amount": [5000.0], "employment": ["A73"]}
)

SHAP_EXPLANATION = {
    "feature_names": [
        "categorical__checking_status_A11",
        "numeric__credit_amount",
        "categorical__employment_A73",
    ],
    "shap_values": [0.5, 0.3, 0.1],
    "base_value": -0.2,
}

COUNTERFACTUALS = [
    {"checking_status": "A11", "credit_amount": 3000.0, "employment": "A73"},
    {"checking_status": "A11", "credit_amount": 5000.0, "employment": "A75"},
]


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, COUNTERFACTUALS, mode="bogus")


def test_template_mode_is_deterministic() -> None:
    first = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, COUNTERFACTUALS, mode="template")
    second = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, COUNTERFACTUALS, mode="template")
    assert first == second


def test_rejected_narrative_mentions_rejection_and_top_factor() -> None:
    narrative = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, COUNTERFACTUALS, mode="template")
    assert "rejected" in narrative
    assert "checking status A11" in narrative


def test_approved_narrative_mentions_approval() -> None:
    # for prediction=0 (approve), the narrative should describe factors
    # supporting approval -- direction flips relative to the rejection case.
    narrative = generate_narrative(INSTANCE, 0, SHAP_EXPLANATION, COUNTERFACTUALS, mode="template")
    assert "approved" in narrative


def test_narrative_includes_counterfactual_phrase() -> None:
    narrative = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, COUNTERFACTUALS, mode="template")
    assert "credit amount" in narrative
    assert "different decision" in narrative


def test_narrative_with_no_counterfactuals_omits_change_sentence() -> None:
    narrative = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, [], mode="template")
    assert "This application was rejected." in narrative
    assert "different decision" not in narrative


def test_duplicate_counterfactual_phrases_are_not_repeated() -> None:
    # both counterfactuals land on the exact same single-feature change --
    # the narrative must say it once, not "either X or X".
    duplicate_counterfactuals = [
        {"checking_status": "A11", "credit_amount": 3000.0, "employment": "A73"},
        {"checking_status": "A11", "credit_amount": 3000.0, "employment": "A73"},
    ]
    narrative = generate_narrative(INSTANCE, 1, SHAP_EXPLANATION, duplicate_counterfactuals, mode="template")
    assert narrative.count("reduce credit amount by approximately 2000 (40%)") == 1
    assert " or reduce credit amount" not in narrative


def test_counterfactuals_on_same_feature_with_different_magnitudes_only_shown_once() -> None:
    # both counterfactuals change credit_amount, but to different values --
    # dedup must key on *which feature* changed, not exact phrase text,
    # or this reads as "either reduce X by 40% or reduce X by 80%".
    same_feature_different_magnitude = [
        {"checking_status": "A11", "credit_amount": 3000.0, "employment": "A73"},
        {"checking_status": "A11", "credit_amount": 1000.0, "employment": "A73"},
    ]
    narrative = generate_narrative(
        INSTANCE, 1, SHAP_EXPLANATION, same_feature_different_magnitude, mode="template"
    )
    assert narrative.count("credit amount by approximately") == 1
    assert " or reduce credit amount" not in narrative


# --- dataset-aware mode: real code translation + feature allowlisting ---

GERMAN_CATEGORICAL_COLS = ["checking_status", "savings_status", "employment"]

GERMAN_INSTANCE = pd.DataFrame(
    {"checking_status": ["A14"], "savings_status": ["A61"], "credit_amount": [5000.0]}
)
GERMAN_SHAP_EXPLANATION = {
    "feature_names": [
        "categorical__checking_status_A14",
        "categorical__savings_status_A61",
        "numeric__credit_amount",
    ],
    "shap_values": [0.5, 0.3, 0.1],
    "base_value": -0.2,
}
GERMAN_COUNTERFACTUALS = [{"checking_status": "A11", "savings_status": "A61", "credit_amount": 3000.0}]


def test_dataset_aware_narrative_translates_german_codes_to_documented_meaning() -> None:
    narrative = generate_narrative(
        GERMAN_INSTANCE,
        1,
        GERMAN_SHAP_EXPLANATION,
        GERMAN_COUNTERFACTUALS,
        mode="template",
        categorical_cols=GERMAN_CATEGORICAL_COLS,
        dataset="german",
    )
    assert "no checking account" in narrative
    assert "A14" not in narrative
    assert "A11" not in narrative  # counterfactual's checking_status code, also translated


HOME_CREDIT_SHAP_EXPLANATION = {
    "feature_names": ["numeric__NONLIVINGAREA_AVG", "numeric__AMT_INCOME_TOTAL"],
    # the disallowed engineered feature has the LARGER shap value --
    # without allowlist filtering it would be the narrative's top factor.
    "shap_values": [0.9, 0.3],
    "base_value": -0.2,
}
HOME_CREDIT_INSTANCE = pd.DataFrame({"AMT_INCOME_TOTAL": [150_000.0], "NONLIVINGAREA_AVG": [0.01]})


def test_dataset_aware_narrative_excludes_non_curated_home_credit_features() -> None:
    narrative = generate_narrative(
        HOME_CREDIT_INSTANCE,
        1,
        HOME_CREDIT_SHAP_EXPLANATION,
        [],
        mode="template",
        categorical_cols=[],
        dataset="home_credit",
    )
    assert "annual income" in narrative
    assert "NONLIVINGAREA" not in narrative.upper().replace(" ", "")


def test_dataset_aware_counterfactual_ignores_non_curated_feature_changes() -> None:
    # the only thing that changed is a disallowed engineered feature --
    # there's no human-meaningful alternative to describe.
    counterfactuals = [{"AMT_INCOME_TOTAL": 150_000.0, "NONLIVINGAREA_AVG": 0.5}]
    narrative = generate_narrative(
        HOME_CREDIT_INSTANCE,
        1,
        HOME_CREDIT_SHAP_EXPLANATION,
        counterfactuals,
        mode="template",
        categorical_cols=[],
        dataset="home_credit",
    )
    assert "different decision" not in narrative
