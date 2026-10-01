"""Tests for src.data.generate_synthetic_ieee.

These check that the generator actually produces the schema and statistical
properties it claims to (real column names, roughly the requested fraud
rate, partial identity coverage), not just that it runs without an
exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.generate_synthetic_ieee import (
    generate_synthetic_identity,
    generate_synthetic_transactions,
)

REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
}


def test_transactions_have_required_real_schema_columns():
    df = generate_synthetic_transactions(n_transactions=200, seed=1)
    missing = REQUIRED_TRANSACTION_COLUMNS - set(df.columns)
    assert not missing


def test_transactions_have_c_d_m_v_column_families():
    df = generate_synthetic_transactions(n_transactions=200, seed=1)
    assert all(f"C{i}" in df.columns for i in range(1, 15))
    assert all(f"D{i}" in df.columns for i in range(1, 16))
    assert all(f"M{i}" in df.columns for i in range(1, 10))
    assert any(col.startswith("V") for col in df.columns)


def test_transaction_id_is_unique():
    df = generate_synthetic_transactions(n_transactions=500, seed=2)
    assert df["TransactionID"].is_unique


def test_realized_fraud_rate_is_close_to_requested():
    df = generate_synthetic_transactions(n_transactions=20000, fraud_rate=0.035, seed=3)
    realized = df["isFraud"].mean()
    # Binomial noise over 20k draws at p=0.035, so this uses a generous
    # tolerance. It is checking "roughly right", not exact equality.
    assert abs(realized - 0.035) < 0.01


def test_transaction_dt_is_strictly_increasing():
    df = generate_synthetic_transactions(n_transactions=500, seed=4)
    assert (df["TransactionDT"].diff().dropna() > 0).all()


def test_fraud_rate_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=10, fraud_rate=0.0)
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=10, fraud_rate=1.5)


def test_n_transactions_must_be_positive():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=0)


def test_same_seed_is_reproducible():
    df1 = generate_synthetic_transactions(n_transactions=300, seed=99)
    df2 = generate_synthetic_transactions(n_transactions=300, seed=99)
    assert df1.equals(df2)


def test_identity_covers_only_a_subset_of_transaction_ids():
    transactions = generate_synthetic_transactions(n_transactions=1000, seed=5)
    identity = generate_synthetic_identity(
        transactions["TransactionID"].to_numpy(), coverage_rate=0.24, seed=6
    )
    assert len(identity) < len(transactions)
    assert set(identity["TransactionID"]).issubset(set(transactions["TransactionID"]))
    # Roughly the requested coverage, not exact.
    assert abs(len(identity) / len(transactions) - 0.24) < 0.02


def test_identity_has_required_columns():
    transactions = generate_synthetic_transactions(n_transactions=300, seed=7)
    identity = generate_synthetic_identity(transactions["TransactionID"].to_numpy(), seed=8)
    assert "DeviceType" in identity.columns
    assert "DeviceInfo" in identity.columns
    assert all(f"id_{i:02d}" in identity.columns for i in (1, 2, 38))


def test_identity_transaction_id_is_unique():
    transactions = generate_synthetic_transactions(n_transactions=1000, seed=9)
    identity = generate_synthetic_identity(transactions["TransactionID"].to_numpy(), seed=10)
    assert identity["TransactionID"].is_unique


def test_coverage_rate_must_be_in_valid_range():
    ids = np.arange(100)
    with pytest.raises(ValueError):
        generate_synthetic_identity(ids, coverage_rate=0.0)
    with pytest.raises(ValueError):
        generate_synthetic_identity(ids, coverage_rate=1.5)


def test_fraud_ring_injection_creates_shared_identifiers_among_fraud():
    # Week 5 needs the synthetic data to have real, graph-detectable fraud
    # ring clustering; see generate_synthetic_transactions's own comment on
    # why. This compares the same seed with and without ring injection, so
    # any difference is attributable to the injection itself, not to
    # incidental randomness.
    with_rings = generate_synthetic_transactions(
        n_transactions=5000, seed=42, fraud_ring_rate=0.3, n_rings=3,
    )
    without_rings = generate_synthetic_transactions(
        n_transactions=5000, seed=42, fraud_ring_rate=0.0, n_rings=3,
    )

    fraud_with = with_rings[with_rings["isFraud"] == 1]
    fraud_without = without_rings[without_rings["isFraud"] == 1]

    max_shared_card_with = fraud_with["card1"].value_counts().max()
    max_shared_card_without = fraud_without["card1"].value_counts().max()
    assert max_shared_card_with > max_shared_card_without

    # At least one ring identifier should be shared by multiple fraud rows.
    assert (fraud_with["card1"].value_counts() >= 2).sum() >= 1


def test_fraud_ring_rate_must_be_in_valid_range():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=10, fraud_ring_rate=-0.1)
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=10, fraud_ring_rate=1.1)


def test_n_rings_must_be_non_negative():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_transactions=10, n_rings=-1)


def test_zero_rings_produces_no_forced_sharing_beyond_chance():
    # A regression guard: n_rings=0 must fully disable the injection, not
    # silently apply some default ring count.
    df = generate_synthetic_transactions(
        n_transactions=1000, seed=11, fraud_ring_rate=0.3, n_rings=0,
    )
    df_no_injection = generate_synthetic_transactions(
        n_transactions=1000, seed=11, fraud_ring_rate=0.0, n_rings=0,
    )
    assert df["card1"].equals(df_no_injection["card1"])
    assert df["P_emaildomain"].equals(df_no_injection["P_emaildomain"])
