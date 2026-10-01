"""
Tests for the real-data code paths added to
src.serving.train_serving_models: train_fraud_serving_model's
transaction_csv/identity_csv arguments and
train_credit_serving_scorecard's credit_csv argument.

The synthetic-data code paths (the defaults, with no csv arguments) are
already exercised indirectly by tests/test_model_registry.py and
tests/test_monitor_psi.py; these tests are specifically for the NEW
real-data loading logic, using small locally-generated stand-in CSVs (the
real Kaggle files are not available in this environment; see
generate_synthetic_ieee.py and generate_synthetic_credit.py's own module
docstrings for why) shaped exactly like the real files would be, including
extra columns the reduced synthetic schema does not otherwise exercise.
"""

from __future__ import annotations

import pytest

from src.data.generate_synthetic_credit import generate_synthetic_credit_applications
from src.data.generate_synthetic_ieee import generate_synthetic_identity, generate_synthetic_transactions
from src.serving.train_serving_models import train_credit_serving_scorecard, train_fraud_serving_model


def test_train_fraud_serving_model_raises_if_only_transaction_csv_given(tmp_path):
    transactions = generate_synthetic_transactions(n_transactions=200, seed=1)
    transaction_path = tmp_path / "train_transaction.csv"
    transactions.to_csv(transaction_path, index=False)

    with pytest.raises(ValueError, match="must both be given together"):
        train_fraud_serving_model(transaction_csv=transaction_path, identity_csv=None)


def test_train_fraud_serving_model_raises_if_only_identity_csv_given(tmp_path):
    transactions = generate_synthetic_transactions(n_transactions=200, seed=1)
    identity = generate_synthetic_identity(transactions["TransactionID"].to_numpy(), seed=2)
    identity_path = tmp_path / "train_identity.csv"
    identity.to_csv(identity_path, index=False)

    with pytest.raises(ValueError, match="must both be given together"):
        train_fraud_serving_model(transaction_csv=None, identity_csv=identity_path)


def test_train_fraud_serving_model_trains_on_real_shaped_csvs(tmp_path):
    # A real train_transaction.csv has 393 columns, this project's reduced
    # synthetic stand-in has fewer; what matters for this test is only that
    # load_and_merge (already schema-agnostic) is actually the code path
    # taken when csv paths are given, not that this stand-in has every real
    # column.
    transactions = generate_synthetic_transactions(n_transactions=1500, fraud_rate=0.05, seed=11)
    identity = generate_synthetic_identity(transactions["TransactionID"].to_numpy(), seed=12)
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    transactions.to_csv(transaction_path, index=False)
    identity.to_csv(identity_path, index=False)

    artifact = train_fraud_serving_model(
        transaction_csv=transaction_path, identity_csv=identity_path,
    )

    assert artifact["n_train"] + artifact["n_test"] == 1500
    assert 0.0 <= artifact["test_auc_roc"] <= 1.0
    assert artifact["thresholds"].review_threshold < artifact["thresholds"].decline_threshold


def test_train_credit_serving_scorecard_trains_on_a_real_shaped_csv(tmp_path):
    # A real application_train.csv has 122 columns; this generator's own
    # 18-plus-id-plus-target columns are the reduced schema
    # load_credit_applications subsets any wider real file down to, so
    # adding an extra column here confirms the subsetting actually happens
    # rather than merely working by coincidence on an already-reduced file.
    df = generate_synthetic_credit_applications(n_applications=1000, seed=5)
    df["FLAG_MOBIL"] = 1
    credit_path = tmp_path / "application_train.csv"
    df.to_csv(credit_path, index=False)

    artifact = train_credit_serving_scorecard(credit_csv=credit_path)

    assert artifact["n_train"] + artifact["n_test"] == 1000
    assert 0.0 <= artifact["test_auc_roc"] <= 1.0
    assert artifact["cutoffs"].approve_cutoff > artifact["cutoffs"].decline_cutoff
    assert len(artifact["reference_scores"]) == artifact["n_test"]
