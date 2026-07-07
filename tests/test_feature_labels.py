"""Tests for the human-readable feature/value label lookups."""

from __future__ import annotations

from xai_loan.data.feature_labels import (
    NARRATIVE_FEATURE_ALLOWLIST,
    humanize_column,
    humanize_feature,
    humanize_value,
    split_transformed_feature_name,
)

GERMAN_CATEGORICAL_COLS = ["checking_status", "other_payment_plans", "personal_status"]


def test_split_numeric_feature() -> None:
    raw_col, code = split_transformed_feature_name("numeric__credit_amount", GERMAN_CATEGORICAL_COLS)
    assert raw_col == "credit_amount"
    assert code is None


def test_split_categorical_feature_with_simple_name() -> None:
    raw_col, code = split_transformed_feature_name(
        "categorical__checking_status_A14", GERMAN_CATEGORICAL_COLS
    )
    assert raw_col == "checking_status"
    assert code == "A14"


def test_split_categorical_feature_with_underscore_in_raw_column_name() -> None:
    # other_payment_plans itself contains underscores -- naive splitting
    # on the last underscore would wrongly carve off "plans" as the column.
    raw_col, code = split_transformed_feature_name(
        "categorical__other_payment_plans_A141", GERMAN_CATEGORICAL_COLS
    )
    assert raw_col == "other_payment_plans"
    assert code == "A141"


def test_humanize_column_known_and_unknown() -> None:
    assert humanize_column("checking_status", "german") == "checking account balance"
    assert humanize_column("not_a_real_column", "german") == "not a real column"


def test_humanize_value_known_code_and_passthrough() -> None:
    assert humanize_value("checking_status", "A14", "german") == "no checking account"
    assert humanize_value("checking_status", "A99", "german") == "A99"  # undocumented code passes through


def test_humanize_feature_categorical_and_numeric() -> None:
    categorical = humanize_feature("categorical__checking_status_A14", GERMAN_CATEGORICAL_COLS, "german")
    assert categorical == "checking account balance (no checking account)"

    numeric = humanize_feature("numeric__credit_amount", GERMAN_CATEGORICAL_COLS, "german")
    assert numeric == "loan amount"


def test_narrative_allowlist_covers_every_german_credit_feature_column() -> None:
    # German Credit's full 20-attribute feature set is documented and
    # human-meaningful -- nothing should be excluded from the narrative.
    from xai_loan.data.loader import load_german_credit

    _, metadata = load_german_credit()
    feature_cols = set(metadata["categorical_cols"]) | set(metadata["numeric_cols"])
    assert feature_cols.issubset(NARRATIVE_FEATURE_ALLOWLIST["german"])


def test_narrative_allowlist_is_a_small_subset_for_home_credit() -> None:
    # Home Credit's model uses ~119 features; only the curated,
    # human-meaningful subset should be allowed in the narrative.
    assert len(NARRATIVE_FEATURE_ALLOWLIST["home_credit"]) < 20
    assert "NONLIVINGAREA_AVG" not in NARRATIVE_FEATURE_ALLOWLIST["home_credit"]
    assert "AMT_INCOME_TOTAL" in NARRATIVE_FEATURE_ALLOWLIST["home_credit"]
