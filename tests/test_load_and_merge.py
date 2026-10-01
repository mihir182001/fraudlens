"""Tests for src.data.load_and_merge.

Uses small hand-built DataFrames (not the synthetic generator) so the merge
semantics, especially "every transaction is kept even with no identity
match", are checked against exact, known-by-construction data rather than
random synthetic data where the right answer would itself need computing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_and_merge import (
    identity_match_rate,
    load_and_merge,
    load_identity,
    load_transactions,
    merge_transaction_identity,
)


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "isFraud": [0, 1, 0, 0],
        "TransactionDT": [100, 200, 300, 400],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
    })


def _sample_identity() -> pd.DataFrame:
    # Only TransactionID 1 and 3 have identity records. 2 and 4 do not.
    return pd.DataFrame({
        "TransactionID": [1, 3],
        "DeviceType": ["desktop", "mobile"],
        "id_01": [0.5, -1.2],
    })


def test_merge_keeps_every_transaction_row():
    merged = merge_transaction_identity(_sample_transactions(), _sample_identity())
    assert len(merged) == 4
    assert set(merged["TransactionID"]) == {1, 2, 3, 4}


def test_merge_fills_missing_identity_with_nan_not_dropped_rows():
    merged = merge_transaction_identity(_sample_transactions(), _sample_identity())
    row2 = merged[merged["TransactionID"] == 2].iloc[0]
    row4 = merged[merged["TransactionID"] == 4].iloc[0]
    assert pd.isna(row2["DeviceType"])
    assert pd.isna(row4["id_01"])


def test_merge_preserves_matched_identity_values():
    merged = merge_transaction_identity(_sample_transactions(), _sample_identity())
    row1 = merged[merged["TransactionID"] == 1].iloc[0]
    assert row1["DeviceType"] == "desktop"
    assert row1["id_01"] == pytest.approx(0.5)


def test_merge_adds_has_identity_match_flag():
    merged = merge_transaction_identity(_sample_transactions(), _sample_identity())
    flags = merged.set_index("TransactionID")["has_identity_match"]
    assert flags.loc[1] is True or flags.loc[1] == True  # noqa: E712
    assert flags.loc[2] is False or flags.loc[2] == False  # noqa: E712
    assert flags.loc[3] == True  # noqa: E712
    assert flags.loc[4] == False  # noqa: E712


def test_identity_match_rate_matches_known_fraction():
    merged = merge_transaction_identity(_sample_transactions(), _sample_identity())
    # 2 of 4 transactions matched, giving exactly 0.5, known by construction.
    assert identity_match_rate(merged) == pytest.approx(0.5)


def test_merge_raises_on_duplicate_identity_keys():
    transactions = _sample_transactions()
    dup_identity = pd.DataFrame({
        "TransactionID": [1, 1],
        "DeviceType": ["desktop", "mobile"],
    })
    with pytest.raises(Exception):
        # pandas' own validate="one_to_one" raises a MergeError subclass;
        # asserting on the base Exception keeps this test independent of
        # pandas' exact internal exception hierarchy.
        merge_transaction_identity(transactions, dup_identity)


def test_merge_requires_transaction_id_column():
    with pytest.raises(ValueError):
        merge_transaction_identity(pd.DataFrame({"foo": [1]}), _sample_identity())
    with pytest.raises(ValueError):
        merge_transaction_identity(_sample_transactions(), pd.DataFrame({"foo": [1]}))


def test_load_transactions_missing_file_raises_with_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_synthetic_ieee"):
        load_transactions(tmp_path / "does_not_exist.csv")


def test_load_transactions_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad_transactions.csv"
    pd.DataFrame({"TransactionID": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_transactions(path)


def test_load_transactions_rejects_duplicate_transaction_ids(tmp_path):
    path = tmp_path / "dup_transactions.csv"
    pd.DataFrame({
        "TransactionID": [1, 1],
        "isFraud": [0, 1],
        "TransactionDT": [100, 200],
        "TransactionAmt": [10.0, 20.0],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_transactions(path)


def test_load_identity_missing_file_raises_with_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_synthetic_ieee"):
        load_identity(tmp_path / "does_not_exist.csv")


def test_load_and_merge_end_to_end_from_csv_files(tmp_path):
    transactions_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    _sample_transactions().to_csv(transactions_path, index=False)
    _sample_identity().to_csv(identity_path, index=False)

    merged = load_and_merge(transactions_path, identity_path)

    assert len(merged) == 4
    assert identity_match_rate(merged) == pytest.approx(0.5)
