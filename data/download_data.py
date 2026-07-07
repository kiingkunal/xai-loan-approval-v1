"""Download the datasets used by the xai-loan-approval framework.

UCI German Credit downloads automatically over plain HTTP (public domain,
no account needed). Home Credit Default Risk lives on Kaggle and requires
an authenticated Kaggle API token, so that path is opt-in via ``--home-credit``
and prints manual setup instructions instead of failing if auth is missing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

DATA_DIR = Path(__file__).resolve().parent
GERMAN_CREDIT_DIR = DATA_DIR / "german_credit"
HOME_CREDIT_DIR = DATA_DIR / "home_credit"

KAGGLE_INSTRUCTIONS = """
Home Credit Default Risk requires Kaggle authentication:

  1. Create a Kaggle account and accept the competition rules at
     kaggle.com/c/home-credit-default-risk
  2. Generate an API token (Kaggle account settings -> "Create New Token"),
     which downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
  3. pip install kaggle
  4. Re-run: python data/download_data.py --home-credit
""".strip()


def download_german_credit() -> None:
    """Download the UCI German Credit dataset over plain HTTP.

    Skips the download if the target file already exists, so the script
    is safe to re-run.
    """
    GERMAN_CREDIT_DIR.mkdir(parents=True, exist_ok=True)
    target = GERMAN_CREDIT_DIR / "german.data"
    if target.exists():
        print(f"German Credit data already present at {target}, skipping.")
        return

    print(f"Downloading German Credit data to {target} ...")
    urllib.request.urlretrieve(GERMAN_CREDIT_URL, target)
    print("Done.")


def download_home_credit() -> None:
    """Download Home Credit Default Risk via the Kaggle CLI, if available.

    Prints setup instructions and returns without error if the ``kaggle``
    package isn't installed or no API token is configured, rather than
    raising — this dataset is optional for prototyping.
    """
    HOME_CREDIT_DIR.mkdir(parents=True, exist_ok=True)
    target = HOME_CREDIT_DIR / "application_train.csv"
    if target.exists():
        print(f"Home Credit data already present at {target}, skipping.")
        return

    if shutil.which("kaggle") is None:
        print(KAGGLE_INSTRUCTIONS)
        return

    kaggle_token = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_token.exists():
        print(KAGGLE_INSTRUCTIONS)
        return

    print("Downloading Home Credit Default Risk via Kaggle CLI ...")
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "home-credit-default-risk",
            "-p",
            str(HOME_CREDIT_DIR),
        ],
        check=False,
    )
    if result.returncode != 0:
        print(KAGGLE_INSTRUCTIONS)
        return

    zip_path = HOME_CREDIT_DIR / "home-credit-default-risk.zip"
    if zip_path.exists():
        shutil.unpack_archive(str(zip_path), str(HOME_CREDIT_DIR))
        zip_path.unlink()
    print("Done.")


def main() -> None:
    """Parse CLI args and run the requested downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home-credit",
        action="store_true",
        help="Also attempt to download Home Credit Default Risk from Kaggle.",
    )
    args = parser.parse_args()

    download_german_credit()
    if args.home_credit:
        download_home_credit()


if __name__ == "__main__":
    sys.exit(main())
