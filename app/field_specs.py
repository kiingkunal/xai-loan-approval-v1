"""Human-facing form field definitions for the Predict tab.

The model needs every column in a dataset's `categorical_cols` +
`numeric_cols` to make a prediction, but a real loan officer's form
shouldn't ask for all of them — German Credit's raw values are
inscrutable codes (``A11``, ``A30``, ...) straight from the UCI
encoding, and Home Credit has 119 model features, most of which (e.g.
external bureau scores, document-submission flags) aren't things an
applicant would type in by hand in a real deployment.

This module defines, per dataset, a curated, human-labeled subset of
fields to actually show. Whatever isn't shown gets filled with a
sensible default (training-set median/mode) computed once in
`load_dataset_bundle` — the model still receives a complete row, the
form just doesn't make the user fill in fields that don't belong on a
form.

Category-code labels (`_CHECKING_STATUS`, etc.) are pulled from
`xai_loan.data.feature_labels` rather than redefined here, so the form
and the plain-English narrative generator always describe a code the
same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from xai_loan.data.feature_labels import GERMAN_CREDIT_VALUE_LABELS, HOME_CREDIT_VALUE_LABELS


@dataclass
class CategoricalField:
    """One categorical input on the form.

    Attributes:
        raw_col: The underlying model feature column name.
        label: Human-facing field label.
        section: Form section heading this field is grouped under.
        code_labels: Maps each raw category value to a human-readable
            display label. If None, raw values are shown as-is (used
            for already-readable categories, e.g. Home Credit's
            ``"Cash loans"``).
    """

    raw_col: str
    label: str
    section: str
    code_labels: dict[str, str] | None = None


@dataclass
class NumericField:
    """One numeric input on the form.

    Attributes:
        raw_col: The underlying model feature column name.
        label: Human-facing field label.
        section: Form section heading this field is grouped under.
        default: Default value shown in the input, in UI units.
        min_value: Minimum value the input accepts, in UI units.
        max_value: Maximum value the input accepts, in UI units.
        step: Input step size, in UI units.
        to_raw: Converts the UI value to the raw model-feature unit.
            Defaults to identity; overridden for fields like age, where
            the UI asks for years but the model feature (Home Credit's
            ``DAYS_BIRTH``) is negative days since birth.
    """

    raw_col: str
    label: str
    section: str
    default: float
    min_value: float
    max_value: float
    step: float = 1.0
    to_raw: Callable[[float], float] = field(default=lambda value: value)


_CHECKING_STATUS = GERMAN_CREDIT_VALUE_LABELS["checking_status"]
_CREDIT_HISTORY = GERMAN_CREDIT_VALUE_LABELS["credit_history"]
_PURPOSE = GERMAN_CREDIT_VALUE_LABELS["purpose"]
_SAVINGS_STATUS = GERMAN_CREDIT_VALUE_LABELS["savings_status"]
_EMPLOYMENT = GERMAN_CREDIT_VALUE_LABELS["employment"]
_PERSONAL_STATUS = GERMAN_CREDIT_VALUE_LABELS["personal_status"]
_OTHER_PARTIES = GERMAN_CREDIT_VALUE_LABELS["other_parties"]
_PROPERTY_MAGNITUDE = GERMAN_CREDIT_VALUE_LABELS["property_magnitude"]
_OTHER_PAYMENT_PLANS = GERMAN_CREDIT_VALUE_LABELS["other_payment_plans"]
_HOUSING = GERMAN_CREDIT_VALUE_LABELS["housing"]
_JOB = GERMAN_CREDIT_VALUE_LABELS["job"]
_OWN_TELEPHONE = GERMAN_CREDIT_VALUE_LABELS["own_telephone"]
_FOREIGN_WORKER = GERMAN_CREDIT_VALUE_LABELS["foreign_worker"]

GERMAN_CREDIT_FIELDS: list[CategoricalField | NumericField] = [
    CategoricalField("personal_status", "Marital Status / Sex", "Personal Details", _PERSONAL_STATUS),
    NumericField("age", "Age (years)", "Personal Details", default=35, min_value=18, max_value=90),
    NumericField("num_dependents", "Number of Dependents", "Personal Details", default=1, min_value=1, max_value=2),
    CategoricalField("foreign_worker", "Foreign Worker", "Personal Details", _FOREIGN_WORKER),
    CategoricalField("own_telephone", "Has a Registered Telephone", "Personal Details", _OWN_TELEPHONE),
    CategoricalField("employment", "Time in Current Employment", "Employment & Housing", _EMPLOYMENT),
    CategoricalField("job", "Job Type", "Employment & Housing", _JOB),
    CategoricalField("housing", "Housing Situation", "Employment & Housing", _HOUSING),
    NumericField("residence_since", "Years at Current Residence", "Employment & Housing", default=2, min_value=1, max_value=4),
    CategoricalField("property_magnitude", "Property Owned", "Employment & Housing", _PROPERTY_MAGNITUDE),
    CategoricalField("checking_status", "Checking Account Balance", "Financial History", _CHECKING_STATUS),
    CategoricalField("savings_status", "Savings Account Balance", "Financial History", _SAVINGS_STATUS),
    CategoricalField("credit_history", "Credit History", "Financial History", _CREDIT_HISTORY),
    NumericField("existing_credits", "Number of Existing Credits", "Financial History", default=1, min_value=1, max_value=4),
    CategoricalField("other_payment_plans", "Other Installment Plans", "Financial History", _OTHER_PAYMENT_PLANS),
    CategoricalField("other_parties", "Co-applicant / Guarantor", "Financial History", _OTHER_PARTIES),
    CategoricalField("purpose", "Loan Purpose", "Loan Details", _PURPOSE),
    NumericField("credit_amount", "Loan Amount (INR)", "Loan Details", default=2500, min_value=250, max_value=20_000, step=50),
    NumericField("duration", "Loan Duration (months)", "Loan Details", default=18, min_value=4, max_value=72),
    NumericField("installment_commitment", "Installment Rate (% of income)", "Loan Details", default=3, min_value=1, max_value=4),
]


def _years_to_days_birth(years: float) -> float:
    return -years * 365.25


def _years_to_days_employed(years: float) -> float:
    return -years * 365.25


_FLAG_OWN_REALTY = HOME_CREDIT_VALUE_LABELS["FLAG_OWN_REALTY"]
_FLAG_OWN_CAR = HOME_CREDIT_VALUE_LABELS["FLAG_OWN_CAR"]

HOME_CREDIT_FIELDS: list[CategoricalField | NumericField] = [
    CategoricalField("NAME_FAMILY_STATUS", "Marital Status", "Personal Details"),
    CategoricalField("NAME_EDUCATION_TYPE", "Education Level", "Personal Details"),
    NumericField("CNT_CHILDREN", "Number of Children", "Personal Details", default=0, min_value=0, max_value=10),
    NumericField(
        "DAYS_BIRTH", "Age (years)", "Personal Details",
        default=35, min_value=18, max_value=100, to_raw=_years_to_days_birth,
    ),
    CategoricalField("NAME_INCOME_TYPE", "Income Type", "Employment & Housing"),
    CategoricalField("OCCUPATION_TYPE", "Occupation", "Employment & Housing"),
    NumericField(
        "DAYS_EMPLOYED", "Years Employed", "Employment & Housing",
        default=5, min_value=0, max_value=50, to_raw=_years_to_days_employed,
    ),
    CategoricalField("NAME_HOUSING_TYPE", "Housing Situation", "Employment & Housing"),
    CategoricalField("FLAG_OWN_REALTY", "Owns Real Estate", "Employment & Housing", _FLAG_OWN_REALTY),
    CategoricalField("FLAG_OWN_CAR", "Owns a Car", "Employment & Housing", _FLAG_OWN_CAR),
    CategoricalField("NAME_CONTRACT_TYPE", "Loan Type", "Loan Details"),
    NumericField("AMT_INCOME_TOTAL", "Annual Income (INR)", "Loan Details", default=150_000, min_value=0, max_value=10_000_000, step=1000),
    NumericField("AMT_CREDIT", "Loan Amount Requested (INR)", "Loan Details", default=250_000, min_value=0, max_value=5_000_000, step=1000),
    NumericField("AMT_ANNUITY", "Loan Annuity (INR, Monthly Payment)", "Loan Details", default=25_000, min_value=0, max_value=500_000, step=100),
]

FIELD_SPECS: dict[str, list[CategoricalField | NumericField]] = {
    "german": GERMAN_CREDIT_FIELDS,
    "home_credit": HOME_CREDIT_FIELDS,
}
