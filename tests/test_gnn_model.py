"""
Tests for src.graph.gnn_model.

The central test here builds a small hand-crafted graph with a real fraud
ring pattern that only graph structure can reveal (a test transaction's
own feature values look identical to an unrelated, non-fraud transaction;
the only difference is which card1 entity node it is connected to). A
passing "learns real signal" test on this data confirms the model is
actually using message passing, not just its own transaction node
features, which a plain feed-forward network on the same feature columns
could also fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.build_transaction_graph import build_hetero_graph
from src.graph.gnn_model import FraudGNN, evaluate_gnn, train_gnn

FEATURE_COLUMNS = ["f1", "f2"]


def _ring_only_graph():
    """Every transaction's own f1/f2 values are drawn from the same
    distribution regardless of fraud status, so a model that ignores graph
    structure entirely has no way to separate the classes. Fraud
    transactions are distinguished only by sharing one of a few ring card1
    values with each other; legitimate transactions each use their own
    unique card. The test-side fraud row shares its ring card with
    train-side fraud rows and nothing else marks it as fraud.
    """
    rng = np.random.default_rng(0)
    n_legit = 60
    n_ring = 20  # split across two rings, ten members each

    legit = pd.DataFrame({
        "TransactionID": np.arange(1, n_legit + 1),
        "isFraud": 0,
        "f1": rng.normal(size=n_legit),
        "f2": rng.normal(size=n_legit),
        "card1": [f"legit_{i}" for i in range(n_legit)],
        "addr1": [f"addr_{i}" for i in range(n_legit)],
        "P_emaildomain": ["gmail.com"] * n_legit,
    })
    ring_card = np.where(np.arange(n_ring) < n_ring // 2, "ring_a", "ring_b")
    ring = pd.DataFrame({
        "TransactionID": np.arange(n_legit + 1, n_legit + n_ring + 1),
        "isFraud": 1,
        "f1": rng.normal(size=n_ring),  # same distribution as legit, no tabular signal
        "f2": rng.normal(size=n_ring),
        "card1": ring_card,
        "addr1": [f"ring_addr_{i}" for i in range(n_ring)],
        "P_emaildomain": ["gmail.com"] * n_ring,
    })

    combined = pd.concat([legit, ring], ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
    train_df = combined.iloc[:64].reset_index(drop=True)
    test_df = combined.iloc[64:].reset_index(drop=True)
    return train_df, test_df


def test_fraud_gnn_rejects_zero_layers():
    with pytest.raises(ValueError):
        FraudGNN(edge_types=[("transaction", "uses_card1", "card1")], num_layers=0)


def test_train_gnn_learns_ring_structure_tabular_features_cannot():
    train_df, test_df = _ring_only_graph()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])

    model, history = train_gnn(
        data, hidden_channels=16, num_layers=2, epochs=150, lr=0.02, random_state=42,
    )

    # Training loss should have actually decreased, not just run.
    assert history[-1] < history[0]

    result = evaluate_gnn(model, data, split="test")
    # A model using only f1/f2 could not do better than chance here (both
    # classes share the same feature distribution by construction), so any
    # real separation has to come from the shared ring card1 structure.
    assert result["auc_roc"] > 0.7
    assert result["n"] == int(data["transaction"].test_mask.sum().item())
    assert result["n_fraud"] == int(data["transaction"].y[data["transaction"].test_mask].sum().item())


def test_train_gnn_raises_when_train_mask_has_one_class():
    train_df, test_df = _ring_only_graph()
    train_df = train_df.copy()
    train_df["isFraud"] = 0
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    with pytest.raises(ValueError, match="train_mask"):
        train_gnn(data, epochs=1)


def test_evaluate_gnn_raises_when_test_mask_has_one_class():
    train_df, test_df = _ring_only_graph()
    test_df = test_df.copy()
    test_df["isFraud"] = 0
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    model, _ = train_gnn(data, epochs=5)
    with pytest.raises(ValueError, match="test_mask"):
        evaluate_gnn(model, data, split="test")


def test_evaluate_gnn_rejects_unknown_split_name():
    train_df, test_df = _ring_only_graph()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    model, _ = train_gnn(data, epochs=5)
    with pytest.raises(ValueError, match="val_mask"):
        evaluate_gnn(model, data, split="val")


def test_train_gnn_is_reproducible_given_same_random_state():
    train_df, test_df = _ring_only_graph()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])

    model_a, history_a = train_gnn(data, epochs=20, random_state=7)
    model_b, history_b = train_gnn(data, epochs=20, random_state=7)

    assert history_a == pytest.approx(history_b)


def test_train_gnn_returns_model_in_eval_mode():
    # Regression guard for a real bug caught while building this module:
    # train_gnn used to return the model still in .train() mode, so two
    # back-to-back forward passes on the same untouched model disagreed
    # because dropout was still active.
    train_df, test_df = _ring_only_graph()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    model, _ = train_gnn(data, epochs=10)
    assert model.training is False


def test_mlflow_log_model_round_trip_for_gnn(tmp_path):
    """Exercises mlflow.pytorch.log_model for FraudGNN specifically.

    mlflow 3.16.1 defaults mlflow.pytorch.log_model to a traced-graph
    export format that requires a concrete input_example and cannot trace
    a forward method that takes dictionaries of tensors the way
    FraudGNN's does; this was confirmed directly (the default raised
    MlflowException asking for an input_example) before switching to
    serialization_format="pickle" here and in train_gnn.py.
    """
    mlflow = pytest.importorskip("mlflow")

    train_df, test_df = _ring_only_graph()
    data = build_hetero_graph(train_df, test_df, FEATURE_COLUMNS, entity_columns=["card1"])
    model, _ = train_gnn(data, epochs=10)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_gnn_test.db")
    mlflow.set_experiment("fraudlens_gnn_log_test")
    with mlflow.start_run(run_name="gnn_model_log_test"):
        model_info = mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    loaded_model = mlflow.pytorch.load_model(model_info.model_uri)
    loaded_model.eval()
    with torch.no_grad():
        original_output = model(data.x_dict, data.edge_index_dict)
        loaded_output = loaded_model(data.x_dict, data.edge_index_dict)
    assert torch.allclose(original_output, loaded_output)
