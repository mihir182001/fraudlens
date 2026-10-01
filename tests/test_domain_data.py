"""
Tests for src.transfer_learning.domain_data.

test_shift_columns_breaks_correlation_with_fraud is this module's central
claim: shift_columns must actually remove the shifted columns' fraud signal
in the resulting domain, not just rearrange rows cosmetically, since Week
8's whole transfer learning comparison depends on that being genuinely true
(see domain_data.py's module docstring for why the shift exists at all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.transfer_learning.domain_data import build_domain

import pytest


def test_build_domain_returns_expected_shapes():
    train_df, test_df, feature_columns = build_domain(
        n_transactions=500, fraud_rate=0.05, seed=1,
    )
    assert len(train_df) + len(test_df) == 500
    assert "isFraud" in train_df.columns
    assert len(feature_columns) > 0
    assert "isFraud" not in feature_columns
    assert "TransactionID" not in feature_columns


def test_build_domain_is_reproducible_given_same_seed():
    train_a, test_a, features_a = build_domain(n_transactions=500, fraud_rate=0.05, seed=7)
    train_b, test_b, features_b = build_domain(n_transactions=500, fraud_rate=0.05, seed=7)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)
    assert features_a == features_b


def test_two_domains_from_different_seeds_share_the_same_feature_schema():
    # Different seed/n_transactions/fraud_rate, no shift: this project's own
    # generator always emits the same column names (see module docstring),
    # so feature_columns must still match, a precondition transfer learning
    # depends on.
    _, _, source_features = build_domain(n_transactions=800, fraud_rate=0.035, seed=1)
    _, _, target_features = build_domain(n_transactions=300, fraud_rate=0.06, seed=2)
    assert source_features == target_features


def test_shift_columns_breaks_correlation_with_fraud():
    shift_columns = ["V1", "V2", "V3"]
    train_shifted, test_shifted, _ = build_domain(
        n_transactions=3000, fraud_rate=0.05, seed=3,
        shift_columns=shift_columns, shift_seed=99,
    )
    train_plain, test_plain, _ = build_domain(n_transactions=3000, fraud_rate=0.05, seed=3)

    combined_shifted = pd.concat([train_shifted, test_shifted])
    combined_plain = pd.concat([train_plain, test_plain])

    for column in shift_columns:
        corr_shifted = abs(combined_shifted[column].corr(combined_shifted["isFraud"]))
        corr_plain = abs(combined_plain[column].corr(combined_plain["isFraud"]))
        # The unshifted generator deliberately correlates every V-column
        # with isFraud (see generate_synthetic_ieee.py), so corr_plain is
        # real and sizeable (measured directly at 0.14-0.19 for V1-V3 on
        # this test's own seed); shifting must knock it down close to the
        # noise floor a random permutation would produce.
        assert corr_plain > 0.10
        assert corr_shifted < 0.05


def test_shift_columns_preserves_the_column_s_own_values():
    # Shifting must be a pure row permutation: the multiset of values in a
    # shifted column is unchanged, only which row each value lands on.
    shift_columns = ["V1"]
    train_shifted, test_shifted, _ = build_domain(
        n_transactions=500, fraud_rate=0.05, seed=4,
        shift_columns=shift_columns, shift_seed=1,
    )
    train_plain, test_plain, _ = build_domain(n_transactions=500, fraud_rate=0.05, seed=4)

    shifted_values = np.sort(
        pd.concat([train_shifted["V1"], test_shifted["V1"]]).to_numpy()
    )
    plain_values = np.sort(pd.concat([train_plain["V1"], test_plain["V1"]]).to_numpy())
    np.testing.assert_allclose(shifted_values, plain_values)


def test_shift_columns_with_different_shift_seeds_gives_different_assignment():
    train_a, test_a, _ = build_domain(
        n_transactions=500, fraud_rate=0.05, seed=4,
        shift_columns=["V1"], shift_seed=1,
    )
    train_b, test_b, _ = build_domain(
        n_transactions=500, fraud_rate=0.05, seed=4,
        shift_columns=["V1"], shift_seed=2,
    )
    combined_a = pd.concat([train_a["V1"], test_a["V1"]]).to_numpy()
    combined_b = pd.concat([train_b["V1"], test_b["V1"]]).to_numpy()
    assert not np.allclose(combined_a, combined_b)


def test_build_domain_rejects_unknown_shift_column():
    with pytest.raises(ValueError, match="unknown column"):
        build_domain(
            n_transactions=200, fraud_rate=0.05, seed=1,
            shift_columns=["not_a_real_column"],
        )


def test_no_shift_by_default():
    train_default, test_default, _ = build_domain(n_transactions=500, fraud_rate=0.05, seed=5)
    train_explicit_none, test_explicit_none, _ = build_domain(
        n_transactions=500, fraud_rate=0.05, seed=5, shift_columns=None,
    )
    pd.testing.assert_frame_equal(train_default, train_explicit_none)
    pd.testing.assert_frame_equal(test_default, test_explicit_none)
