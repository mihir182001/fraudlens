"""
Tests for src.models.prepare_model_data.

The most important property under test here is one that is easy to get
wrong silently: velocity and card-usage features must be computed on the
FULL chronological history before the train/test split, not separately per
split, or a cards test-period transaction would incorrectly "forget" its
own real prior transactions from the train period. That property gets its
own dedicated test below rather than being left to an end-to-end smoke test
to catch by accident.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.prepare_model_data import (
    chronological_train_test_split,
    prepare_train_test_features,
    select_feature_columns,
)


def test_chronological_split_exact_known_boundary():
    df = pd.DataFrame({
        "TransactionDT": [0, 100, 200, 300, 400],
        "isFraud": [0, 0, 1, 0, 1],
    })
    train_df, test_df = chronological_train_test_split(df, test_size=0.4)
    # 5 rows, test_size=0.4 gives 2 test rows and 3 train rows: the split
    # index is round(5 * 0.6) = 3.
    assert len(train_df) == 3
    assert len(test_df) == 2
    assert train_df["TransactionDT"].tolist() == [0, 100, 200]
    assert test_df["TransactionDT"].tolist() == [300, 400]


def test_chronological_split_is_actually_chronological():
    df = pd.DataFrame({
        "TransactionDT": [500, 100, 300, 200, 400],
        "isFraud": [0, 1, 0, 1, 0],
    })
    train_df, test_df = chronological_train_test_split(df, test_size=0.4)
    assert train_df["TransactionDT"].max() <= test_df["TransactionDT"].min()


def test_chronological_split_rejects_invalid_test_size():
    df = pd.DataFrame({"TransactionDT": [0, 1], "isFraud": [0, 1]})
    with pytest.raises(ValueError):
        chronological_train_test_split(df, test_size=0.0)
    with pytest.raises(ValueError):
        chronological_train_test_split(df, test_size=1.0)


def test_chronological_split_rejects_empty_df():
    with pytest.raises(ValueError):
        chronological_train_test_split(pd.DataFrame({"TransactionDT": []}))


def test_chronological_split_requires_time_column():
    with pytest.raises(ValueError):
        chronological_train_test_split(pd.DataFrame({"x": [1, 2]}))


def _cross_boundary_sample() -> pd.DataFrame:
    # Card A transacts three times: twice before the split boundary and
    # once after. If velocity/card-usage features were (incorrectly)
    # computed separately per split instead of on the full history first,
    # the post-boundary A transaction would show 0 prior transactions
    # instead of the correct 2.
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "card1": ["A", "A", "B", "B", "B", "A"],
        "TransactionDT": [0, 100, 200, 300, 400, 500],
        "TransactionAmt": [10.0, 20.0, 5.0, 5.0, 5.0, 30.0],
        "addr1": [111, 111, 222, 222, 222, 111],
        "isFraud": [0, 0, 0, 0, 1, 0],
        "ProductCD": ["W", "W", "C", "C", "C", "W"],
        "card4": ["visa"] * 6,
        "card6": ["debit"] * 6,
        "P_emaildomain": ["gmail.com"] * 6,
        "R_emaildomain": [None] * 6,
    })


def test_velocity_and_card_usage_features_see_prior_history_across_the_split():
    # test_size chosen so the split boundary falls between card A's second
    # and third transaction (rows sorted by TransactionDT: 0,100,200,300,
    # 400 | 500 with test_size close to 1/6).
    train_df, test_df = prepare_train_test_features(_cross_boundary_sample(), test_size=1 / 6)

    assert len(test_df) == 1
    test_row = test_df.iloc[0]
    assert test_row["TransactionID"] == 6
    # Card A's third transaction should see 2 prior transactions (rows 1
    # and 2, both in the train period), not 0.
    assert test_row["card1_prior_txn_count"] == 2
    assert test_row["card1_running_avg_amt"] == pytest.approx(15.0)  # mean of 10, 20


def test_hour_of_day_and_frequency_features_are_fit_on_train_only():
    train_df, test_df = prepare_train_test_features(_cross_boundary_sample(), test_size=1 / 6)
    # Every ProductCD value in this sample is "W" or "C", both present in
    # train; if frequency encoding were (incorrectly) fit on df as a whole
    # or on test itself, the test-set proportions would differ from what
    # was actually observed in train alone. Here this test checks the
    # encoding runs through without error and produces values in the valid
    # [0, 1] range, with the real fit source verified directly in
    # test_engineer_features.py.
    assert "ProductCD_freq" in test_df.columns
    assert (test_df["ProductCD_freq"] >= 0).all()
    assert (test_df["ProductCD_freq"] <= 1).all()
    assert "hour_of_day_avg_amt" in test_df.columns


def test_select_feature_columns_excludes_target_and_id():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "isFraud": [0, 1],
        "amt": [10.0, 20.0],
        "has_identity_match": [True, False],
        "ProductCD": ["W", "C"],
    })
    columns = select_feature_columns(df)
    assert "TransactionID" not in columns
    assert "isFraud" not in columns
    assert "amt" in columns
    assert "has_identity_match" in columns
    assert "ProductCD" not in columns  # raw categorical, not numeric/bool
