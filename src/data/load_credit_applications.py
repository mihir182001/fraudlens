"""
load_credit_applications.py, Credit Risk Scorecard module: loads the real
Home Credit Default Risk application_train.csv and subsets it down to the
same reduced column set generate_synthetic_credit.py's synthetic stand-in
already produces (APPLICATION_FEATURE_COLUMNS there), so every downstream
module (WoE encoding, the points scorecard, the FastAPI CreditScoringRequest
schema) keeps working completely unchanged, now against real values instead
of synthetic ones.

Why subset rather than expose the real file's full 122 columns: the serving
API (CreditScoringRequest, src/serving/schemas.py) only accepts these named
18 fields, a deliberate choice made when this project only had synthetic
data to build against. Widening the API to the real file's full column set
is a real, separate design decision (a different request schema, different
WoE special-value handling for whichever additional columns turn out
predictive) that is not what this loader does; its job is narrowly to make
the EXISTING pipeline run on real values, not to redesign the API. This
mirrors generate_synthetic_credit.py's own module docstring, which notes
that module's code (and, by extension, this loader) is meant to run
unchanged against the real downloaded file, just with more columns present
in the file than are actually used.

Unlike the fraud detection side (src/data/load_and_merge.py), no merge is
needed here: the real Home Credit application_train.csv is already a single
flat file (one row per applicant), so this loader is a read plus a column
subset and validation, not a join.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_credit import APPLICATION_FEATURE_COLUMNS

REQUIRED_COLUMNS = {"SK_ID_CURR", "TARGET", *APPLICATION_FEATURE_COLUMNS}


def load_credit_applications(path: str | Path) -> pd.DataFrame:
    """Loads application_train.csv (or a synthetic stand-in with the same
    column names) and returns it subset to SK_ID_CURR, TARGET, and the same
    18 columns generate_synthetic_credit_applications produces, in that same
    order, so the returned DataFrame is schema-identical to the synthetic
    generator's own output regardless of how many extra columns the real
    file has (122 in the real dataset, versus this reduced 20-column
    result).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Credit application file not found: {path}. If you haven't "
            "downloaded the real Home Credit data yet, run "
            "`python -m src.data.generate_synthetic_credit` to create a "
            "synthetic stand-in at data/raw/application_train_synthetic.csv, "
            "or pass the path to your own file."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            "Is this really Home Credit's application_train.csv (or a file "
            "with the same schema)?"
        )
    if df["SK_ID_CURR"].duplicated().any():
        n_dupes = int(df["SK_ID_CURR"].duplicated().sum())
        raise ValueError(
            f"{path} has {n_dupes} duplicate SK_ID_CURR value(s); "
            "SK_ID_CURR is expected to be a unique key."
        )

    ordered_columns = ["SK_ID_CURR", "TARGET"] + APPLICATION_FEATURE_COLUMNS
    return df[ordered_columns].copy()
