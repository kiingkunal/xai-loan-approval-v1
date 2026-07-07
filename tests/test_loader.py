"""Tests for dataset loaders' critical paths.

Uses a tiny synthetic ``german.data`` file written to a temp directory so
these tests never depend on the real dataset being downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xai_loan.data.loader import load_german_credit, load_home_credit

_SYNTHETIC_ROWS = [
    "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1",
    "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2",
    "A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1",
]


def _write_synthetic_german_credit(tmp_path: Path) -> Path:
    data_dir = tmp_path / "german_credit"
    data_dir.mkdir()
    (data_dir / "german.data").write_text("\n".join(_SYNTHETIC_ROWS) + "\n")
    return data_dir


def test_load_german_credit_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_german_credit(data_dir=tmp_path / "does_not_exist")


def test_load_german_credit_maps_target_and_metadata(tmp_path: Path) -> None:
    data_dir = _write_synthetic_german_credit(tmp_path)
    df, metadata = load_german_credit(data_dir=data_dir)

    assert len(df) == len(_SYNTHETIC_ROWS)
    assert metadata["target_col"] == "target"
    assert set(metadata["protected_cols"]) == {"sex", "age"}
    assert "sex" not in metadata["categorical_cols"]
    assert "class" not in df.columns

    # rows above end in " 1" or " 2" -> target 0 (good) or 1 (bad)
    assert df["target"].tolist() == [0, 1, 0]


def test_load_german_credit_derives_sex_from_personal_status(tmp_path: Path) -> None:
    data_dir = _write_synthetic_german_credit(tmp_path)
    df, _ = load_german_credit(data_dir=data_dir)

    # row 0: A93 (single male) -> male; row 1: A92 (female) -> female
    assert df.loc[0, "sex"] == "male"
    assert df.loc[1, "sex"] == "female"


def test_load_home_credit_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_home_credit(data_dir=tmp_path / "does_not_exist")


def test_load_home_credit_renames_target_and_drops_id(tmp_path: Path) -> None:
    data_dir = tmp_path / "home_credit"
    data_dir.mkdir()
    csv_content = (
        "SK_ID_CURR,TARGET,CODE_GENDER,DAYS_BIRTH,AMT_INCOME_TOTAL\n"
        "100001,0,M,-10000,150000\n"
        "100002,1,F,-12000,90000\n"
        "100003,0,F,-9000,200000\n"
    )
    (data_dir / "application_train.csv").write_text(csv_content)

    df, metadata = load_home_credit(data_dir=data_dir)

    assert "SK_ID_CURR" not in df.columns
    assert "target" in df.columns
    assert "TARGET" not in df.columns
    assert set(metadata["protected_cols"]) == {"CODE_GENDER", "DAYS_BIRTH"}
    assert "CODE_GENDER" not in metadata["categorical_cols"]
    assert "DAYS_BIRTH" in metadata["numeric_cols"]


def test_load_home_credit_sample_size_limits_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "home_credit"
    data_dir.mkdir()
    header = "SK_ID_CURR,TARGET,CODE_GENDER,DAYS_BIRTH,AMT_INCOME_TOTAL\n"
    rows = "\n".join(f"{100000 + i},{i % 2},M,-10000,100000" for i in range(50))
    (data_dir / "application_train.csv").write_text(header + rows + "\n")

    df, _ = load_home_credit(sample_size=10, data_dir=data_dir)
    assert len(df) == 10
