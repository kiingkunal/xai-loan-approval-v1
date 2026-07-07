"""Load the German Credit and Home Credit datasets.

Both loaders return a ``(DataFrame, DatasetMetadata)`` pair so downstream
code (preprocessing, training, fairness auditing) never has to hardcode
column names — it reads them from the metadata dict instead.

Design choice shared by both loaders: a column that directly encodes
gender (``sex`` for German Credit, ``CODE_GENDER`` for Home Credit) is
listed in ``protected_cols`` for fairness auditing but deliberately left
out of ``categorical_cols`` / ``numeric_cols``, so the default feature set
doesn't train on gender directly. Age is handled differently: it stays in
the feature set *and* is flagged as protected, since age is both a
legitimate underwriting signal and a common axis of credit-fairness
concern in the literature — excluding it from features would just hide
the question the fairness audit exists to answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from xai_loan.utils.config import GERMAN_CREDIT_DIR, HOME_CREDIT_DIR, RANDOM_STATE


class DatasetMetadata(TypedDict):
    """Column-role metadata returned alongside every loaded dataset."""

    target_col: str
    protected_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]


_GERMAN_CREDIT_COLUMNS: list[str] = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]

_GERMAN_CREDIT_NUMERIC_COLS: list[str] = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

_GERMAN_CREDIT_CATEGORICAL_COLS: list[str] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

_PERSONAL_STATUS_TO_SEX: dict[str, str] = {
    "A91": "male",  # divorced/separated male
    "A92": "female",  # divorced/separated/married female
    "A93": "male",  # single male
    "A94": "male",  # married/widowed male
    "A95": "female",  # single female
}

_GERMAN_CREDIT_DOWNLOAD_HINT = (
    "German Credit data not found at {path}. Run "
    "`python data/download_data.py` to download it."
)

_HOME_CREDIT_DOWNLOAD_HINT = (
    "Home Credit data not found at {path}. Run "
    "`python data/download_data.py --home-credit` after setting up Kaggle "
    "authentication (see data/README.md)."
)


def load_german_credit(data_dir: Path = GERMAN_CREDIT_DIR) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load the UCI German Credit (Statlog) dataset.

    Args:
        data_dir: Directory containing ``german.data``. Defaults to the
            project's standard location.

    Returns:
        A tuple of the loaded DataFrame (1,000 rows) and its column
        metadata. The target column is named ``target`` with 0 = good
        credit risk (approve) and 1 = bad credit risk (reject), remapped
        from the dataset's original 1/2 encoding. A derived ``sex``
        column is added from ``personal_status`` for fairness auditing.

    Raises:
        FileNotFoundError: If ``german.data`` is missing, with
            instructions to run the download script.
    """
    file_path = data_dir / "german.data"
    if not file_path.exists():
        raise FileNotFoundError(_GERMAN_CREDIT_DOWNLOAD_HINT.format(path=file_path))

    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=_GERMAN_CREDIT_COLUMNS)

    df["target"] = df["class"].map({1: 0, 2: 1}).astype(int)
    df = df.drop(columns=["class"])

    df["sex"] = df["personal_status"].map(_PERSONAL_STATUS_TO_SEX)

    metadata: DatasetMetadata = {
        "target_col": "target",
        "protected_cols": ["sex", "age"],
        "categorical_cols": list(_GERMAN_CREDIT_CATEGORICAL_COLS),
        "numeric_cols": list(_GERMAN_CREDIT_NUMERIC_COLS),
    }
    return df, metadata


def load_home_credit(
    sample_size: int | None = None,
    data_dir: Path = HOME_CREDIT_DIR,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load the Home Credit Default Risk main application table.

    Args:
        sample_size: If given, randomly sample this many rows (with the
            project's fixed random seed) instead of loading all ~307K
            rows. Useful for fast iteration on the large dataset.
        data_dir: Directory containing ``application_train.csv``.
            Defaults to the project's standard location.

    Returns:
        A tuple of the loaded DataFrame and its column metadata. The
        target column is renamed from ``TARGET`` to ``target`` (1 =
        default, 0 = repaid) for consistency with ``load_german_credit``.
        The applicant ID column (``SK_ID_CURR``) is dropped since it
        carries no predictive or explanatory signal.

    Raises:
        FileNotFoundError: If ``application_train.csv`` is missing, with
            instructions to run the download script.
    """
    file_path = data_dir / "application_train.csv"
    if not file_path.exists():
        raise FileNotFoundError(_HOME_CREDIT_DOWNLOAD_HINT.format(path=file_path))

    df = pd.read_csv(file_path)
    df = df.drop(columns=["SK_ID_CURR"])
    df = df.rename(columns={"TARGET": "target"})

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "target"]
    categorical_cols = [c for c in feature_cols if df[c].dtype == object]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    protected_cols = [c for c in ("CODE_GENDER", "DAYS_BIRTH") if c in df.columns]
    if "CODE_GENDER" in categorical_cols:
        categorical_cols.remove("CODE_GENDER")

    metadata: DatasetMetadata = {
        "target_col": "target",
        "protected_cols": protected_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    return df, metadata
