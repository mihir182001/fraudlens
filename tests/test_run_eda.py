"""Tests for src.eda.run_eda.

Uses small hand-built DataFrames with known-by-construction properties
(exact fraud count, exact missing-value count, exact time buckets) so each
EDA function's output can be checked against an exact expected number,
rather than just "did it run."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda.run_eda import (
    compute_feature_distributions_by_fraud,
    compute_fraud_rate,
    compute_missing_value_rates,
    compute_time_patterns,
    run_eda,
)


def test_compute_fraud_rate_exact_known_value():
    df = pd.DataFrame({"isFraud": [0, 0, 0, 1]})
    result = compute_fraud_rate(df)
    assert result == {
        "n_total": 4,
        "n_fraud": 1,
        "n_legitimate": 3,
        "fraud_rate": 0.25,
    }


def test_compute_fraud_rate_requires_isfraud_column():
    with pytest.raises(ValueError):
        compute_fraud_rate(pd.DataFrame({"x": [1, 2]}))


def test_compute_fraud_rate_rejects_empty_df():
    with pytest.raises(ValueError):
        compute_fraud_rate(pd.DataFrame({"isFraud": []}))


def test_compute_missing_value_rates_exact_known_values():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],           # 0% missing
        "b": [1, None, None, None],  # 75% missing
        "c": [1, 2, None, 4],        # 25% missing
    })
    rates = compute_missing_value_rates(df)
    assert rates["a"] == pytest.approx(0.0)
    assert rates["b"] == pytest.approx(0.75)
    assert rates["c"] == pytest.approx(0.25)
    # Sorted descending.
    assert list(rates.index) == ["b", "c", "a"]


def test_compute_missing_value_rates_rejects_empty_df():
    with pytest.raises(ValueError):
        compute_missing_value_rates(pd.DataFrame())


def test_compute_feature_distributions_separates_fraud_groups():
    df = pd.DataFrame({
        "isFraud": [0, 0, 1, 1],
        "TransactionID": [1, 2, 3, 4],
        "amt": [10.0, 10.0, 100.0, 100.0],
    })
    result = compute_feature_distributions_by_fraud(df)
    # TransactionID must be excluded from auto-selected columns.
    assert "TransactionID" not in result.index.get_level_values(0)
    amt_mean = result.loc[("amt", "mean")]
    assert amt_mean[0] == pytest.approx(10.0)
    assert amt_mean[1] == pytest.approx(100.0)


def test_compute_feature_distributions_requires_isfraud():
    with pytest.raises(ValueError):
        compute_feature_distributions_by_fraud(pd.DataFrame({"x": [1, 2]}))


def test_compute_feature_distributions_raises_when_no_numeric_columns():
    df = pd.DataFrame({"isFraud": [0, 1], "label": ["a", "b"]})
    with pytest.raises(ValueError):
        compute_feature_distributions_by_fraud(df)


def test_compute_time_patterns_exact_known_buckets():
    # Two transactions one day apart (86400s), two more three hours apart
    # within that second day. Exact buckets computed by hand below.
    df = pd.DataFrame({
        "TransactionDT": [0, 86400, 86400 + 3600 * 3, 86400 + 3600 * 3],
        "isFraud": [0, 1, 0, 1],
    })
    result = compute_time_patterns(df)

    by_day = result["by_elapsed_day"]
    assert by_day.loc[0, "n_transactions"] == 1
    assert by_day.loc[0, "fraud_rate"] == pytest.approx(0.0)
    assert by_day.loc[1, "n_transactions"] == 3
    assert by_day.loc[1, "fraud_rate"] == pytest.approx(2 / 3)

    by_hour = result["by_hour_of_day"]
    assert by_hour.loc[0, "n_transactions"] == 2  # the two at hour-of-day 0
    assert by_hour.loc[3, "n_transactions"] == 2  # the two at hour-of-day 3


def test_compute_time_patterns_requires_columns():
    with pytest.raises(ValueError):
        compute_time_patterns(pd.DataFrame({"isFraud": [0, 1]}))
    with pytest.raises(ValueError):
        compute_time_patterns(pd.DataFrame({"TransactionDT": [0, 1]}))


def test_compute_time_patterns_never_produces_a_real_calendar_date():
    # Regression guard for the module's core caveat: nothing here should
    # try to interpret TransactionDT as a Unix timestamp.
    df = pd.DataFrame({"TransactionDT": [0, 100000], "isFraud": [0, 1]})
    result = compute_time_patterns(df)
    for frame in result.values():
        assert not any("date" in str(col).lower() for col in frame.columns)


def test_run_eda_returns_all_four_items():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "isFraud": [0, 1, 0, 1],
        "TransactionDT": [0, 1000, 2000, 3000],
        "amt": [10.0, 20.0, np.nan, 40.0],
    })
    result = run_eda(df)
    assert set(result.keys()) == {
        "fraud_rate", "missing_value_rates",
        "feature_distributions_by_fraud", "time_patterns",
    }
    assert result["fraud_rate"]["n_fraud"] == 2
