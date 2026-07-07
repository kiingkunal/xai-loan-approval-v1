# Datasets

This directory is gitignored except for this file and `download_data.py`.
Run the download script to populate it locally.

## 1. UCI German Credit (Statlog)

- **Source:** UCI Machine Learning Repository, Statlog (German Credit Data)
- **Size:** 1,000 rows × 20 attributes + 1 target
- **License:** Public domain, free to use
- **Where it lands:** `data/german_credit/german.data` (whitespace-separated,
  no header — column names are assigned by `xai_loan.data.loader`)

Downloaded automatically by `download_data.py` — no account or API key needed.

## 2. Home Credit Default Risk (Kaggle)

- **Source:** Kaggle competition "Home Credit Default Risk"
- **Size:** ~307K rows in `application_train.csv` (main table; the
  competition also ships several auxiliary tables we don't use)
- **License:** Free, but requires a Kaggle account + accepting the
  competition rules before download
- **Where it lands:** `data/home_credit/application_train.csv`

Kaggle requires authentication, so this isn't a plain HTTP download:

1. Create a Kaggle account and accept the competition rules at
   `kaggle.com/c/home-credit-default-risk`.
2. Generate an API token (Kaggle account settings → "Create New Token"),
   which downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json`.
3. `pip install kaggle` (not in `requirements.txt` — it's only needed for
   this one-time download, not by the framework itself).
4. Run `python data/download_data.py --home-credit`.

If the `kaggle` CLI/package isn't installed or `kaggle.json` isn't found,
the script prints these same instructions and exits cleanly instead of
failing with a stack trace.

## Usage

```bash
# German Credit only (default, no auth needed)
python data/download_data.py

# Both datasets (Home Credit needs Kaggle auth, see above)
python data/download_data.py --home-credit
```
