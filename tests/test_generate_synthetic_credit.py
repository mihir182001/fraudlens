"""
Tests for src.data.generate_synthetic_credit.

test_days_employed_anomaly_only_for_pensioner_and_unemployed checks the one
detail this generator exists to reproduce faithfully (see its module
docstring): the real Home Credit dataset's DAYS_EMPLOYED=365243 placeholder
for not-working applicants, which Week 9's WoE encoding is specifically
written to bin separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_synthetic_credit import (
    APPLICATION_FEATURE_COLUMNS,
    DAYS_EMPLOYED_ANOMALY,
    generate_synthetic_credit_applications,
)


def test_generate_returns_expected_shape_and_realized_rate_near_target():
    df = generate_synthetic_credit_applications(n_applications=4000, default_rate=0.08, seed=1)
    assert len(df) == 4000
    assert set(["SK_ID_CURR", "TARGET"]).issubset(df.columns)
    assert abs(df["TARGET"].mean() - 0.08) < 0.02


def test_application_feature_columns_constant_matches_generator_output():
    # src/data/load_credit_applications.py imports APPLICATION_FEATURE_COLUMNS
    # to subset the REAL application_train.csv down to this same schema, so
    # this constant drifting out of sync with what this generator actually
    # produces would silently break that subsetting rather than raising
    # anywhere obvious. This test is that drift's actual guard.
    df = generate_synthetic_credit_applications(n_applications=50, seed=1)
    expected_columns = ["SK_ID_CURR", "TARGET"] + APPLICATION_FEATURE_COLUMNS
    assert list(df.columns) == expected_columns


def test_reproducible_given_same_seed():
    df_a = generate_synthetic_credit_applications(n_applications=1000, seed=7)
    df_b = generate_synthetic_credit_applications(n_applications=1000, seed=7)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_days_employed_anomaly_only_for_pensioner_and_unemployed():
    df = generate_synthetic_credit_applications(n_applications=5000, seed=3)
    anomaly_rows = df[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY]
    assert len(anomaly_rows) > 0
    assert set(anomaly_rows["NAME_INCOME_TYPE"].unique()).issubset({"Pensioner", "Unemployed"})

    not_working = df[df["NAME_INCOME_TYPE"].isin(["Pensioner", "Unemployed"])]
    assert (not_working["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).all()

    working = df[~df["NAME_INCOME_TYPE"].isin(["Pensioner", "Unemployed"])]
    assert not (working["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).any()


def test_ext_source_columns_negatively_correlated_with_target():
    df = generate_synthetic_credit_applications(n_applications=6000, seed=5)
    for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        corr = df[column].corr(df["TARGET"])
        assert corr < -0.05


def test_education_default_rate_is_monotonic_by_education_level():
    df = generate_synthetic_credit_applications(n_applications=8000, seed=9)
    rates = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean()
    ordered = [
        rates["Academic degree"], rates["Higher education"], rates["Incomplete higher"],
        rates["Secondary / secondary special"], rates["Lower secondary"],
    ]
    assert ordered == sorted(ordered)


def test_unemployment_shift_increases_not_working_share():
    base = generate_synthetic_credit_applications(n_applications=4000, seed=1, start_id=1)
    stressed = generate_synthetic_credit_applications(
        n_applications=4000, seed=2, start_id=1_000_000, unemployment_shift=0.15,
    )
    base_share = base["NAME_INCOME_TYPE"].isin(["Pensioner", "Unemployed"]).mean()
    stressed_share = stressed["NAME_INCOME_TYPE"].isin(["Pensioner", "Unemployed"]).mean()
    assert stressed_share > base_share + 0.05


def test_ext_source_shift_lowers_mean_ext_source():
    base = generate_synthetic_credit_applications(n_applications=4000, seed=1, start_id=1)
    stressed = generate_synthetic_credit_applications(
        n_applications=4000, seed=2, start_id=1_000_000, ext_source_shift=-0.12,
    )
    for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        assert stressed[column].mean() < base[column].mean() - 0.05


def test_rejects_non_positive_n_applications():
    with pytest.raises(ValueError, match="n_applications must be positive"):
        generate_synthetic_credit_applications(n_applications=0)


def test_rejects_invalid_default_rate():
    with pytest.raises(ValueError, match="default_rate must be strictly between"):
        generate_synthetic_credit_applications(default_rate=1.5)
