"""
Tests for src.graph.build_transaction_graph.

The property most worth its own dedicated test here is one that is
easy to get backwards: a shared entity (the same card1 value) must
actually connect the train-side and test-side transactions that used it
through a real edge in the graph, or the entire premise of graph fraud
detection (a transaction is suspicious partly because of who it is
connected to) never gets exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graph.build_transaction_graph import build_hetero_graph

FEATURE_COLUMNS = ["f1", "f2"]


def _small_frames():
    train_df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "isFraud": [0, 0, 1, 1],
        "f1": [0.1, 0.2, 5.0, 5.1],
        "f2": [1.0, 1.1, 1.0, 1.2],
        "card1": ["A", "B", "X", "X"],
        "addr1": [111, 222, 333, 333],
        "P_emaildomain": ["gmail.com", "yahoo.com", "aol.com", "aol.com"],
    })
    test_df = pd.DataFrame({
        "TransactionID": [5, 6],
        "isFraud": [0, 1],
        "f1": [0.15, 5.2],
        "f2": [1.05, 0.9],
        "card1": ["C", "X"],  # row 6 shares the ring card "X" with train fraud rows
        "addr1": [444, 333],
        "P_emaildomain": ["hotmail.com", "aol.com"],
    })
    return train_df, test_df


def test_build_hetero_graph_basic_shapes():
    train_df, test_df = _small_frames()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])

    assert data["transaction"].x.shape == (6, 2)
    assert data["transaction"].y.tolist() == [0, 0, 1, 1, 0, 1]
    assert data["transaction"].train_mask.tolist() == [True, True, True, True, False, False]
    assert data["transaction"].test_mask.tolist() == [False, False, False, False, True, True]
    # 4 distinct card1 values across both frames: A, B, C, X.
    assert data["card1"].x.shape == (4, 1)


def test_build_hetero_graph_shared_entity_connects_train_and_test_rows():
    train_df, test_df = _small_frames()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])

    edge_index = data[("transaction", "uses_card1", "card1")].edge_index.numpy()
    # Row indices 2, 3 (train fraud rows) and 5 (test fraud row, 0-indexed
    # position 5 in the combined frame) all used card1 "X".
    tx_to_card = dict(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert tx_to_card[2] == tx_to_card[3] == tx_to_card[5]

    # And the reverse edge type must exist and be the mirror of the forward one.
    reverse_edge_index = data[("card1", "rev_uses_card1", "transaction")].edge_index.numpy()
    assert reverse_edge_index[0].tolist() == edge_index[1].tolist()
    assert reverse_edge_index[1].tolist() == edge_index[0].tolist()


def test_build_hetero_graph_entity_features_are_constant_not_label_derived():
    # Entity features must never encode anything about the transactions
    # connected to them (including their labels), or test-set labels could
    # leak into the graph through a shared entity's feature.
    train_df, test_df = _small_frames()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    assert (data["card1"].x == 1.0).all()


def test_build_hetero_graph_imputes_and_scales_using_train_only():
    train_df, test_df = _small_frames()
    train_df = train_df.copy()
    test_df = test_df.copy()
    # Give the test set a missing value and a value far outside train's
    # range, to make train-only fitting and test-only-transform observable.
    test_df.loc[0, "f1"] = np.nan
    test_df.loc[1, "f1"] = 1000.0

    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    x = data["transaction"].x.numpy()

    # Train rows (indices 0-3) should be standardized to roughly mean 0.
    assert abs(x[:4, 0].mean()) < 1e-5
    # The imputed test row should not be NaN, and the far-out-of-range test
    # value should still show up as a large positive scaled value (it was
    # never used to refit the scaler, so it does not get pulled back toward
    # its own mean).
    assert not np.isnan(x[4, 0])
    assert x[5, 0] > 5.0


def test_build_hetero_graph_rejects_missing_feature_column():
    train_df, test_df = _small_frames()
    with pytest.raises(ValueError, match="missing required columns"):
        build_hetero_graph(train_df, test_df, ["f1", "not_a_real_column"], entity_columns=["card1"])


def test_build_hetero_graph_rejects_empty_frames():
    train_df, test_df = _small_frames()
    with pytest.raises(ValueError, match="non-empty"):
        build_hetero_graph(train_df.iloc[0:0], test_df, FEATURE_COLUMNS, entity_columns=["card1"])


def test_build_hetero_graph_rejects_overlapping_transaction_ids():
    train_df, test_df = _small_frames()
    test_df = test_df.copy()
    test_df.loc[0, "TransactionID"] = train_df["TransactionID"].iloc[0]
    with pytest.raises(ValueError, match="share"):
        build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])


def test_build_hetero_graph_supports_multiple_entity_columns():
    train_df, test_df = _small_frames()
    data = build_hetero_graph(
        train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1", "addr1", "P_emaildomain"],
    )
    assert set(data.node_types) == {"transaction", "card1", "addr1", "P_emaildomain"}
    assert ("transaction", "uses_addr1", "addr1") in data.edge_types
    assert ("transaction", "uses_P_emaildomain", "P_emaildomain") in data.edge_types
