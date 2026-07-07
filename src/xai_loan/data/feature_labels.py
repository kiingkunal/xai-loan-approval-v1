"""Human-readable labels for raw dataset columns and category codes.

Centralizes the dataset documentation lookups so the Predict form
(`app/field_specs.py`) and the plain-English narrative generator
(`explainers/narrative.py`) describe the same feature the same way,
instead of each inventing its own phrasing. German Credit's category
codes (``A11``, ``A30``, ...) come straight from the UCI Statlog
attribute documentation, transcribed here, not invented.

`NARRATIVE_FEATURE_ALLOWLIST` exists because the two consumers of this
module have different audiences: the Predict form and the narrative
generator both face a loan applicant, so they're restricted to the same
set of human-meaningful, actionable columns. The Explain tab's raw
SHAP/LIME breakdown is for technical review and isn't restricted —
Home Credit's model genuinely does use ~119 features (building
statistics, document-submission flags, etc.), and an analyst should be
able to see that. An applicant just shouldn't be told their rejection
hinges on "NONLIVINGAREA_AVG".
"""

from __future__ import annotations

GERMAN_CREDIT_COLUMN_LABELS: dict[str, str] = {
    "checking_status": "checking account balance",
    "duration": "loan duration",
    "credit_history": "credit history",
    "purpose": "loan purpose",
    "credit_amount": "loan amount",
    "savings_status": "savings account balance",
    "employment": "time in current employment",
    "installment_commitment": "installment rate",
    "personal_status": "marital status",
    "other_parties": "co-applicant/guarantor",
    "residence_since": "years at current residence",
    "property_magnitude": "property owned",
    "age": "age",
    "other_payment_plans": "other installment plans",
    "housing": "housing situation",
    "existing_credits": "number of existing credits",
    "job": "job type",
    "num_dependents": "number of dependents",
    "own_telephone": "registered telephone",
    "foreign_worker": "foreign worker status",
}

# Transcribed from the UCI Statlog (German Credit Data) attribute documentation.
GERMAN_CREDIT_VALUE_LABELS: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 INR",
        "A12": "0 to 200 INR",
        "A13": ">= 200 INR",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing",
    },
    "purpose": {
        "A40": "new car",
        "A41": "used car",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_status": {
        "A61": "< 100 INR",
        "A62": "100 to 500 INR",
        "A63": "500 to 1000 INR",
        "A64": ">= 1000 INR",
        "A65": "unknown / no savings account",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 to 4 years",
        "A74": "4 to 7 years",
        "A75": ">= 7 years",
    },
    "personal_status": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_parties": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property_magnitude": {
        "A121": "real estate",
        "A122": "savings/life insurance",
        "A123": "car or other",
        "A124": "no property",
    },
    "other_payment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled, non-resident",
        "A172": "unskilled, resident",
        "A173": "skilled employee/official",
        "A174": "management / self-employed / highly qualified",
    },
    "own_telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

HOME_CREDIT_COLUMN_LABELS: dict[str, str] = {
    "NAME_FAMILY_STATUS": "marital status",
    "NAME_EDUCATION_TYPE": "education level",
    "CNT_CHILDREN": "number of children",
    "DAYS_BIRTH": "age",
    "NAME_INCOME_TYPE": "income type",
    "OCCUPATION_TYPE": "occupation",
    "DAYS_EMPLOYED": "years employed",
    "NAME_HOUSING_TYPE": "housing situation",
    "FLAG_OWN_REALTY": "real estate ownership",
    "FLAG_OWN_CAR": "car ownership",
    "NAME_CONTRACT_TYPE": "loan type",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount requested",
    "AMT_ANNUITY": "loan annuity",
}

HOME_CREDIT_VALUE_LABELS: dict[str, dict[str, str]] = {
    "FLAG_OWN_REALTY": {"Y": "yes", "N": "no"},
    "FLAG_OWN_CAR": {"Y": "yes", "N": "no"},
}

COLUMN_LABELS_BY_DATASET: dict[str, dict[str, str]] = {
    "german": GERMAN_CREDIT_COLUMN_LABELS,
    "home_credit": HOME_CREDIT_COLUMN_LABELS,
}
VALUE_LABELS_BY_DATASET: dict[str, dict[str, dict[str, str]]] = {
    "german": GERMAN_CREDIT_VALUE_LABELS,
    "home_credit": HOME_CREDIT_VALUE_LABELS,
}

NARRATIVE_FEATURE_ALLOWLIST: dict[str, set[str]] = {
    "german": set(GERMAN_CREDIT_COLUMN_LABELS),
    "home_credit": set(HOME_CREDIT_COLUMN_LABELS),
}


def split_transformed_feature_name(
    transformed_name: str, categorical_cols: list[str]
) -> tuple[str, str | None]:
    """Split a preprocessed column name into its raw column and category code.

    Preprocessed names look like ``"categorical__checking_status_A11"`` or
    ``"numeric__credit_amount"`` (the `ColumnTransformer` branch, then the
    raw column name, then -- for categoricals -- the one-hot category
    value). Raw column names can themselves contain underscores (e.g.
    ``other_payment_plans``), so the raw column can't be recovered by
    blindly splitting on the last underscore; instead, each known
    categorical column name is tried as a literal prefix.

    Args:
        transformed_name: A column name from the preprocessed feature
            matrix (`LoanDataPreprocessor`'s output columns).
        categorical_cols: The dataset's known raw categorical column
            names, used to resolve the prefix match.

    Returns:
        ``(raw_col, code)`` for a categorical feature, or
        ``(raw_col, None)`` for a numeric feature or anything that
        doesn't match a known branch prefix.
    """
    if transformed_name.startswith("numeric__"):
        return transformed_name.removeprefix("numeric__"), None
    if transformed_name.startswith("categorical__"):
        rest = transformed_name.removeprefix("categorical__")
        for raw_col in categorical_cols:
            if rest == raw_col:
                return raw_col, None
            if rest.startswith(raw_col + "_"):
                return raw_col, rest[len(raw_col) + 1 :]
        return rest, None
    return transformed_name, None


def humanize_column(raw_col: str, dataset: str) -> str:
    """Human label for a bare raw column name, with no value attached.

    Args:
        raw_col: A raw (un-preprocessed) column name.
        dataset: ``"german"`` or ``"home_credit"``.

    Returns:
        The documented column label, or an underscore-stripped fallback
        if `raw_col` isn't in the label dictionary.
    """
    return COLUMN_LABELS_BY_DATASET.get(dataset, {}).get(raw_col, raw_col.replace("_", " "))


def humanize_value(raw_col: str, raw_value: object, dataset: str) -> object:
    """Human label for one raw column's value, e.g. a German Credit code.

    Args:
        raw_col: A raw (un-preprocessed) column name.
        raw_value: The value to look up (e.g. ``"A14"``).
        dataset: ``"german"`` or ``"home_credit"``.

    Returns:
        The documented value label, or `raw_value` itself unchanged if
        there's no entry (e.g. for columns with no documented code map,
        or numeric values, which don't need translating).
    """
    return VALUE_LABELS_BY_DATASET.get(dataset, {}).get(raw_col, {}).get(raw_value, raw_value)


def humanize_feature(transformed_name: str, categorical_cols: list[str], dataset: str) -> str:
    """Build a human-readable phrase for one preprocessed feature.

    Args:
        transformed_name: A column name from the preprocessed feature
            matrix.
        categorical_cols: The dataset's known raw categorical column names.
        dataset: ``"german"`` or ``"home_credit"`` -- selects which label
            dictionaries to use. Falls back to underscore-stripped raw
            names for any other value (e.g. unit tests using a synthetic
            dataset).

    Returns:
        A phrase like ``"checking account balance (no checking account)"``
        for a categorical feature, or ``"loan amount"`` for a numeric one.
    """
    raw_col, code = split_transformed_feature_name(transformed_name, categorical_cols)
    label = humanize_column(raw_col, dataset)
    if code is not None:
        return f"{label} ({humanize_value(raw_col, code, dataset)})"
    return label
