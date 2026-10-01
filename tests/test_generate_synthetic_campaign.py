"""
Tests for src.data.generate_synthetic_campaign.

The test worth the most attention here is
test_observed_uplift_ordering_matches_segment_design: it checks that the
generator's random draws actually reproduce, in the observed data, the
segment ordering the module docstring promises (persuadable highest,
sleeping_dog negative). Everything downstream in this week's uplift models
and metrics is only meaningful if this holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.generate_synthetic_campaign import (
    DEFAULT_SEGMENT_SHARES,
    FEATURE_COLUMNS,
    SEGMENT_PROBABILITIES,
    SEGMENTS,
    generate_synthetic_campaign,
)


def test_required_columns_present():
    df = generate_synthetic_campaign(n_customers=500, seed=1)
    expected = set(FEATURE_COLUMNS) | {
        "customer_id", "treatment", "outcome", "segment", "true_uplift",
    }
    assert expected.issubset(set(df.columns))


def test_customer_id_is_unique():
    df = generate_synthetic_campaign(n_customers=500, seed=1)
    assert df["customer_id"].is_unique


def test_treatment_and_outcome_are_binary():
    df = generate_synthetic_campaign(n_customers=500, seed=1)
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["outcome"].unique()).issubset({0, 1})


def test_realized_treatment_rate_is_close_to_requested():
    df = generate_synthetic_campaign(n_customers=20000, treatment_rate=0.4, seed=2)
    assert abs(df["treatment"].mean() - 0.4) < 0.01


def test_segment_shares_are_close_to_requested():
    df = generate_synthetic_campaign(n_customers=50000, seed=3)
    realized = df["segment"].value_counts(normalize=True)
    for segment, expected_share in DEFAULT_SEGMENT_SHARES.items():
        assert abs(realized[segment] - expected_share) < 0.01


def test_same_seed_is_reproducible():
    df1 = generate_synthetic_campaign(n_customers=1000, seed=99)
    df2 = generate_synthetic_campaign(n_customers=1000, seed=99)
    assert df1.equals(df2)


def test_true_uplift_matches_segment_probability_definition_exactly():
    # true_uplift is a deterministic function of segment (p1 - p0), fixed
    # constants in SEGMENT_PROBABILITIES, so this must hold exactly, not
    # approximately.
    df = generate_synthetic_campaign(n_customers=2000, seed=4)
    for segment in SEGMENTS:
        expected = (
            SEGMENT_PROBABILITIES[segment]["p1"] - SEGMENT_PROBABILITIES[segment]["p0"]
        )
        rows = df[df["segment"] == segment]
        assert rows["true_uplift"].to_numpy() == pytest.approx(expected)


def test_observed_uplift_ordering_matches_segment_design():
    # With enough customers, the empirical (observed outcome rate under
    # treatment minus under control) uplift per segment should recover the
    # module's designed ordering: persuadable is the largest positive
    # effect, sleeping_dog is the only negative one, and sure_thing /
    # lost_cause are both small in magnitude. This is the property every
    # uplift model and metric in this module is being tested against.
    df = generate_synthetic_campaign(n_customers=200000, seed=5)

    observed_uplift = {}
    for segment in SEGMENTS:
        rows = df[df["segment"] == segment]
        treated_rate = rows.loc[rows["treatment"] == 1, "outcome"].mean()
        control_rate = rows.loc[rows["treatment"] == 0, "outcome"].mean()
        observed_uplift[segment] = treated_rate - control_rate

    assert observed_uplift["persuadable"] > 0.25
    assert observed_uplift["sleeping_dog"] < -0.10
    assert abs(observed_uplift["sure_thing"]) < 0.10
    assert abs(observed_uplift["lost_cause"]) < 0.10
    assert observed_uplift["persuadable"] > observed_uplift["sure_thing"]
    assert observed_uplift["persuadable"] > observed_uplift["lost_cause"]
    assert observed_uplift["sleeping_dog"] < observed_uplift["lost_cause"]


def test_tenure_months_is_not_segment_informative():
    # tenure_months is deliberately drawn identically regardless of segment;
    # this guards against that noise feature accidentally picking up signal.
    df = generate_synthetic_campaign(n_customers=50000, seed=6)
    means_by_segment = df.groupby("segment")["tenure_months"].mean()
    assert means_by_segment.max() - means_by_segment.min() < 2.0


def test_n_customers_must_be_positive():
    with pytest.raises(ValueError):
        generate_synthetic_campaign(n_customers=0)


def test_treatment_rate_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        generate_synthetic_campaign(n_customers=10, treatment_rate=0.0)
    with pytest.raises(ValueError):
        generate_synthetic_campaign(n_customers=10, treatment_rate=1.0)


def test_segment_shares_must_have_exactly_the_known_segments():
    with pytest.raises(ValueError, match="segment_shares"):
        generate_synthetic_campaign(
            n_customers=10, segment_shares={"persuadable": 1.0},
        )


def test_segment_shares_must_sum_to_one():
    bad_shares = {
        "persuadable": 0.5, "sure_thing": 0.5, "lost_cause": 0.5, "sleeping_dog": 0.5,
    }
    with pytest.raises(ValueError, match="sum to 1"):
        generate_synthetic_campaign(n_customers=10, segment_shares=bad_shares)


def test_segment_shares_must_be_non_negative():
    bad_shares = {
        "persuadable": -0.1, "sure_thing": 0.4, "lost_cause": 0.4, "sleeping_dog": 0.3,
    }
    with pytest.raises(ValueError, match="non-negative"):
        generate_synthetic_campaign(n_customers=10, segment_shares=bad_shares)
