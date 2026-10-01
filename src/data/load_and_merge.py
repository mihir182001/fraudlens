"""
load_and_merge.py, Week 1: load the IEEE-CIS transaction and identity
files and merge them the way the competition itself specifies.

This module is deliberately schema-agnostic beyond the one column it
actually depends on (`TransactionID`, plus `isFraud` where noted). It never
hardcodes an expected V-column count, C/D/M-column count, or id_-column
count. That's intentional: it's built and tested here against
generate_synthetic_ieee.py's reduced schema (V1-V20 instead of the real
V1-V339), and the exact same code runs unchanged against the real Kaggle
files once you've downloaded them. More columns just means more columns
in the resulting DataFrame, not a code change.

Merge semantics (per the competition's own data description, and mirrored
in generate_synthetic_ieee.py): NOT every transaction has a matching
identity record, so this is a LEFT join on TransactionID from the
transaction table. Every transaction row is kept, and rows with no
identity match get NaN in every identity column rather than being dropped.
An inner join would silently throw away the majority of transactions (real
identity coverage is roughly a quarter of transactions), which would badly
distort both the fraud rate and every downstream model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_TRANSACTION_COLUMNS = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
REQUIRED_IDENTITY_COLUMNS = {"TransactionID"}


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Loads train_transaction.csv (or a synthetic stand-in with the same
    column names) and validates that the columns this whole pipeline
    depends on are actually present, rather than failing confusingly deep
    inside a later step.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {path}. If you haven't downloaded "
            "the real IEEE-CIS data yet, run "
            "`python -m src.data.generate_synthetic_ieee` to create a "
            "synthetic stand-in at data/raw/train_transaction.csv."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_TRANSACTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            "Is this really the IEEE-CIS train_transaction.csv (or a file "
            "with the same schema)?"
        )
    if df["TransactionID"].duplicated().any():
        n_dupes = int(df["TransactionID"].duplicated().sum())
        raise ValueError(
            f"{path} has {n_dupes} duplicate TransactionID value(s); "
            "TransactionID is expected to be a unique key."
        )
    return df


def load_identity(path: str | Path) -> pd.DataFrame:
    """Loads train_identity.csv (or a synthetic stand-in). Unlike the
    transaction file, it's normal and expected for this file to cover only
    a subset of all transaction IDs. That's exactly what
    merge_transaction_identity's left join is for.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Identity file not found: {path}. If you haven't downloaded "
            "the real IEEE-CIS data yet, run "
            "`python -m src.data.generate_synthetic_ieee` to create a "
            "synthetic stand-in at data/raw/train_identity.csv."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_IDENTITY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            "Is this really the IEEE-CIS train_identity.csv (or a file "
            "with the same schema)?"
        )
    if df["TransactionID"].duplicated().any():
        n_dupes = int(df["TransactionID"].duplicated().sum())
        raise ValueError(
            f"{path} has {n_dupes} duplicate TransactionID value(s); "
            "TransactionID is expected to be a unique key in the identity "
            "file too."
        )
    return df


def merge_transaction_identity(
    transactions: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """Left-joins identity onto transactions on TransactionID.

    Every row of `transactions` is preserved (row count of the result
    always equals row count of `transactions`, checked below rather than
    assumed). Transactions with no identity match get NaN in every identity
    column instead of being dropped; see module docstring for why this
    must be a left join, not an inner join.
    """
    if "TransactionID" not in transactions.columns:
        raise ValueError("transactions is missing the TransactionID column.")
    if "TransactionID" not in identity.columns:
        raise ValueError("identity is missing the TransactionID column.")

    # Computed from the raw TransactionID sets rather than merge's own
    # indicator=True option: on a wide frame (the real data has 434+
    # columns once transaction + identity are combined) indicator=True
    # measurably triggers pandas' "DataFrame is highly fragmented"
    # PerformanceWarning internally, confirmed directly by running the
    # merge both ways and counting warnings. Set membership is also just
    # more direct for what this is actually checking.
    has_match = transactions["TransactionID"].isin(identity["TransactionID"])

    merged = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")

    if len(merged) != len(transactions):
        raise AssertionError(
            f"Left join changed row count (was {len(transactions)}, now "
            f"{len(merged)}); this should be impossible with "
            "validate='one_to_one' and would indicate a duplicate-key bug."
        )

    # Kept as an explicit boolean `has_identity_match` rather than relying
    # on NaN-ness of any single identity column (e.g. DeviceType), since
    # real identity columns can themselves be missing even for a matched
    # transaction. See identity_match_rate's docstring.
    merged = pd.concat(
        [merged, has_match.rename("has_identity_match").reset_index(drop=True)], axis=1
    )
    return merged


def load_and_merge(
    transaction_path: str | Path, identity_path: str | Path
) -> pd.DataFrame:
    """Convenience wrapper: load both files and merge them in one call."""
    transactions = load_transactions(transaction_path)
    identity = load_identity(identity_path)
    return merge_transaction_identity(transactions, identity)


def identity_match_rate(merged: pd.DataFrame) -> float:
    """Fraction of transactions that had a matching identity record.

    Reads the explicit `has_identity_match` column merge_transaction_identity
    adds, rather than checking whether some identity column (e.g.
    DeviceType) is non-null. An individual identity column can itself be
    missing even for a transaction that DID get an identity match, so
    NaN-checking a single column would understate coverage. Reported by the
    EDA step so "how much identity coverage did we actually get" is a real
    measured number, not assumed from the spec.
    """
    if "has_identity_match" not in merged.columns:
        raise ValueError(
            "merged has no has_identity_match column. Was this really "
            "produced by merge_transaction_identity?"
        )
    return float(merged["has_identity_match"].mean())
