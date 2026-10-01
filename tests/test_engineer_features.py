"""Tests for src.features.engineer_features.

Every feature function is tested against small, hand-built DataFrames where
the correct output is known by construction (not just "did it run"), plus
an explicit leakage guard test per causal feature: add a future
high-value transaction and confirm it does not change any earlier row's
feature value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer_features import (
    add_card_usage_features,
    add_hour_of_day_aggregates,
    add_velocity_features,
    engineer_features,
    fit_frequencies,
    frequency_encode,
    transform_frequencies,
)


def _velocity_sample() -> pd.DataFrame:
    # Not given in time order on purpose, since add_velocity_features must
    # sort internally. Sorted by TransactionDT this is:
    #   card A @ t=0     (amt 10)
    #   card B @ t=100   (amt 5)
    #   card A @ t=1800  (amt 20)
    #   card A @ t=7200  (amt 30)
    return pd.DataFrame({
        "card1": ["A", "A", "A", "B"],
        "TransactionDT": [0, 1800, 7200, 100],
        "TransactionAmt": [10.0, 20.0, 30.0, 5.0],
    })


def test_velocity_1h_window_exact_known_counts_and_sums():
    result = add_velocity_features(_velocity_sample())
    by_time = result.set_index("TransactionDT")

    # 1h window (3600s), trailing, inclusive of the row itself.
    assert by_time.loc[0, "card1_txn_count_1h"] == 1
    assert by_time.loc[0, "card1_amt_sum_1h"] == pytest.approx(10.0)
    # t=1800: window is (t-3600, t], which includes A@0 and A@1800.
    assert by_time.loc[1800, "card1_txn_count_1h"] == 2
    assert by_time.loc[1800, "card1_amt_sum_1h"] == pytest.approx(30.0)
    # t=7200: window is (3600, 7200], which excludes A@1800 (t=1800 is not
    # > 3600), so only A@7200 itself falls inside.
    assert by_time.loc[7200, "card1_txn_count_1h"] == 1
    assert by_time.loc[7200, "card1_amt_sum_1h"] == pytest.approx(30.0)
    assert by_time.loc[100, "card1_txn_count_1h"] == 1
    assert by_time.loc[100, "card1_amt_sum_1h"] == pytest.approx(5.0)


def test_velocity_1d_window_exact_known_counts_and_sums():
    result = add_velocity_features(_velocity_sample())
    by_time = result.set_index("TransactionDT")

    # 1d window (86400s) comfortably covers all three card A transactions.
    assert by_time.loc[0, "card1_txn_count_1d"] == 1
    assert by_time.loc[1800, "card1_txn_count_1d"] == 2
    assert by_time.loc[7200, "card1_txn_count_1d"] == 3
    assert by_time.loc[7200, "card1_amt_sum_1d"] == pytest.approx(60.0)


def test_velocity_features_never_see_a_future_transaction():
    baseline = add_velocity_features(_velocity_sample())
    baseline_by_time = baseline.set_index("TransactionDT")

    future_row = pd.DataFrame({"card1": ["A"], "TransactionDT": [100000], "TransactionAmt": [9999.0]})
    with_future = pd.concat([_velocity_sample(), future_row], ignore_index=True)
    with_future_result = add_velocity_features(with_future)
    with_future_by_time = with_future_result.set_index("TransactionDT")

    for t in (0, 1800, 7200, 100):
        assert with_future_by_time.loc[t, "card1_txn_count_1h"] == baseline_by_time.loc[t, "card1_txn_count_1h"]
        assert with_future_by_time.loc[t, "card1_txn_count_1d"] == baseline_by_time.loc[t, "card1_txn_count_1d"]
        assert with_future_by_time.loc[t, "card1_amt_sum_1d"] == pytest.approx(
            baseline_by_time.loc[t, "card1_amt_sum_1d"]
        )


def test_velocity_requires_columns():
    with pytest.raises(ValueError):
        add_velocity_features(pd.DataFrame({"x": [1]}))


def test_velocity_rejects_empty_df():
    with pytest.raises(ValueError):
        add_velocity_features(pd.DataFrame({"card1": [], "TransactionDT": [], "TransactionAmt": []}))


def _card_usage_sample() -> pd.DataFrame:
    # Sorted by TransactionDT this is:
    #   card A @ t=0    (amt 10, addr1 100)
    #   card B @ t=100  (amt 5,  addr1 300)
    #   card A @ t=1800 (amt 20, addr1 100)
    #   card A @ t=7200 (amt 30, addr1 200)
    return pd.DataFrame({
        "card1": ["A", "A", "A", "B"],
        "TransactionDT": [0, 1800, 7200, 100],
        "TransactionAmt": [10.0, 20.0, 30.0, 5.0],
        "addr1": [100, 100, 200, 300],
    })


def test_card_usage_prior_txn_count_exact_known_values():
    result = add_card_usage_features(_card_usage_sample())
    by_time = result.set_index("TransactionDT")
    assert by_time.loc[0, "card1_prior_txn_count"] == 0
    assert by_time.loc[1800, "card1_prior_txn_count"] == 1
    assert by_time.loc[7200, "card1_prior_txn_count"] == 2
    assert by_time.loc[100, "card1_prior_txn_count"] == 0


def test_card_usage_running_avg_excludes_current_row():
    result = add_card_usage_features(_card_usage_sample())
    by_time = result.set_index("TransactionDT")
    # First transaction for a card has no prior history: NaN, not its own amount.
    assert pd.isna(by_time.loc[0, "card1_running_avg_amt"])
    # At t=1800, the only prior A transaction is the one at t=0 (amt 10).
    assert by_time.loc[1800, "card1_running_avg_amt"] == pytest.approx(10.0)
    # At t=7200, prior A transactions are t=0 (10) and t=1800 (20): mean 15.
    assert by_time.loc[7200, "card1_running_avg_amt"] == pytest.approx(15.0)


def test_card_usage_amt_ratio_to_own_avg():
    result = add_card_usage_features(_card_usage_sample())
    by_time = result.set_index("TransactionDT")
    assert pd.isna(by_time.loc[0, "card1_amt_ratio_to_own_avg"])
    assert by_time.loc[1800, "card1_amt_ratio_to_own_avg"] == pytest.approx(2.0)
    assert by_time.loc[7200, "card1_amt_ratio_to_own_avg"] == pytest.approx(2.0)


def test_card_usage_prior_distinct_addr1_count():
    result = add_card_usage_features(_card_usage_sample())
    by_time = result.set_index("TransactionDT")
    assert by_time.loc[0, "card1_prior_distinct_addr1_count"] == 0
    # Prior A rows so far: just addr1=100.
    assert by_time.loc[1800, "card1_prior_distinct_addr1_count"] == 1
    # Prior A rows: addr1 100 and 100 again: still only 1 distinct value.
    assert by_time.loc[7200, "card1_prior_distinct_addr1_count"] == 1


def test_card_usage_features_never_see_a_future_transaction():
    baseline = add_card_usage_features(_card_usage_sample())
    baseline_by_time = baseline.set_index("TransactionDT")

    future_row = pd.DataFrame({
        "card1": ["A"], "TransactionDT": [100000], "TransactionAmt": [9999.0], "addr1": [999],
    })
    with_future = pd.concat([_card_usage_sample(), future_row], ignore_index=True)
    with_future_result = add_card_usage_features(with_future)
    with_future_by_time = with_future_result.set_index("TransactionDT")

    for t in (0, 1800, 7200, 100):
        base_avg = baseline_by_time.loc[t, "card1_running_avg_amt"]
        new_avg = with_future_by_time.loc[t, "card1_running_avg_amt"]
        if pd.isna(base_avg):
            assert pd.isna(new_avg)
        else:
            assert new_avg == pytest.approx(base_avg)
        assert (
            with_future_by_time.loc[t, "card1_prior_distinct_addr1_count"]
            == baseline_by_time.loc[t, "card1_prior_distinct_addr1_count"]
        )


def test_card_usage_requires_columns():
    with pytest.raises(ValueError):
        add_card_usage_features(pd.DataFrame({"x": [1]}))


def test_card_usage_rejects_empty_df():
    with pytest.raises(ValueError):
        add_card_usage_features(
            pd.DataFrame({"card1": [], "TransactionDT": [], "TransactionAmt": []})
        )


def test_hour_of_day_aggregates_fit_on_self_exact_known_values():
    df = pd.DataFrame({
        "TransactionDT": [0, 7200, 7210, 18000],  # hour buckets: 0, 2, 2, 5
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
    })
    result = add_hour_of_day_aggregates(df)
    assert result.loc[0, "hour_of_day_avg_amt"] == pytest.approx(10.0)
    assert result.loc[0, "hour_of_day_txn_count"] == 1
    assert result.loc[1, "hour_of_day_avg_amt"] == pytest.approx(25.0)
    assert result.loc[1, "hour_of_day_txn_count"] == 2
    assert result.loc[2, "hour_of_day_avg_amt"] == pytest.approx(25.0)
    assert result.loc[3, "hour_of_day_avg_amt"] == pytest.approx(40.0)


def test_hour_of_day_aggregates_uses_fit_df_not_df_itself():
    df = pd.DataFrame({
        "TransactionDT": [0, 7200, 7210, 18000],  # hour buckets: 0, 2, 2, 5
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
    })
    fit_df = pd.DataFrame({
        "TransactionDT": [0, 7200, 7210, 18000],
        "TransactionAmt": [100.0, 200.0, 200.0, 300.0],
    })
    result = add_hour_of_day_aggregates(df, fit_df=fit_df)
    # These values come from fit_df's amounts, not df's own (10/20/30/40).
    assert result.loc[0, "hour_of_day_avg_amt"] == pytest.approx(100.0)
    assert result.loc[1, "hour_of_day_avg_amt"] == pytest.approx(200.0)
    assert result.loc[3, "hour_of_day_avg_amt"] == pytest.approx(300.0)


def test_hour_of_day_aggregates_requires_columns():
    with pytest.raises(ValueError):
        add_hour_of_day_aggregates(pd.DataFrame({"x": [1]}))


def test_fit_and_transform_frequencies_exact_known_proportions():
    df = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
    freq_tables = fit_frequencies(df, ["cat"])
    result = transform_frequencies(df, freq_tables)
    assert result["cat_freq"].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25])


def test_transform_frequencies_unseen_value_maps_to_zero():
    fit_df = pd.DataFrame({"cat": ["a", "a", "b"]})
    freq_tables = fit_frequencies(fit_df, ["cat"])
    new_df = pd.DataFrame({"cat": ["a", "d"]})
    result = transform_frequencies(new_df, freq_tables)
    assert result["cat_freq"].iloc[0] == pytest.approx(2 / 3)
    assert result["cat_freq"].iloc[1] == pytest.approx(0.0)


def test_frequency_encode_convenience_wrapper_matches_fit_then_transform():
    df = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
    result = frequency_encode(df, ["cat"])
    assert result["cat_freq"].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25])


def test_fit_frequencies_requires_column_to_exist():
    with pytest.raises(ValueError):
        fit_frequencies(pd.DataFrame({"x": [1]}), ["cat"])


def test_engineer_features_smoke_test_produces_expected_new_columns():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "card1": ["A", "A", "B", "A"],
        "TransactionDT": [0, 1800, 100, 7200],
        "TransactionAmt": [10.0, 20.0, 5.0, 30.0],
        "addr1": [100, 100, 300, 200],
        "ProductCD": ["W", "W", "C", "W"],
        "card4": ["visa", "visa", "mastercard", "visa"],
        "card6": ["debit", "debit", "credit", "debit"],
        "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "gmail.com"],
        "R_emaildomain": [None, None, None, None],
    })
    result = engineer_features(df)

    expected_new_columns = {
        "card1_txn_count_1h", "card1_amt_sum_1h",
        "card1_txn_count_1d", "card1_amt_sum_1d",
        "card1_prior_txn_count", "card1_running_avg_amt",
        "card1_amt_ratio_to_own_avg", "card1_prior_distinct_addr1_count",
        "hour_of_day_avg_amt", "hour_of_day_txn_count",
        "ProductCD_freq", "card4_freq", "card6_freq", "P_emaildomain_freq",
    }
    assert expected_new_columns.issubset(set(result.columns))
    assert len(result) == len(df)
