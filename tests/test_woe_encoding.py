"""
Tests for src.credit_risk.woe_encoding.

test_woe_iv_table_matches_hand_computation is the module's central check: a
hand-worked two-bin example (80/20 split in one bin, 20/80 in the other)
with a known, independently computed IV, verified here against this
module's own smoothed formula rather than trusted on faith. The smoothing
constant (_EPSILON) is small enough that the smoothed IV should land close
to, but measurably below, the unsmoothed hand value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.credit_risk.woe_encoding import (
    MISSING_LABEL,
    _woe_iv_table,
    fit_woe_bins,
    iv_report,
    iv_strength_label,
    select_features_by_iv,
    transform_woe,
)

_UNSMOOTHED_HAND_IV = 2 * (0.8 - 0.2) * math.log(0.8 / 0.2)


def test_woe_iv_table_matches_hand_computation():
    bin_labels = pd.Series(["A"] * 100 + ["B"] * 100)
    target = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    table = _woe_iv_table(bin_labels, target)

    assert set(table["bin"]) == {"A", "B"}
    row_a = table[table["bin"] == "A"].iloc[0]
    assert row_a["woe"] > 0  # bin A is majority-good, so WoE = ln(good/bad) > 0.
    row_b = table[table["bin"] == "B"].iloc[0]
    assert row_b["woe"] < 0

    total_iv = table["iv"].sum()
    assert total_iv == pytest.approx(_UNSMOOTHED_HAND_IV, rel=0.03)
    assert total_iv < _UNSMOOTHED_HAND_IV  # smoothing pulls it down slightly, never up.


def test_woe_iv_table_rejects_single_class_target():
    bin_labels = pd.Series(["A", "A", "B", "B"])
    target = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError, match="Both classes"):
        _woe_iv_table(bin_labels, target)


@pytest.mark.parametrize("iv,expected", [
    (0.01, "not_useful"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong"), (0.9, "suspicious"),
])
def test_iv_strength_label_thresholds(iv, expected):
    assert iv_strength_label(iv) == expected


def _toy_credit_frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ext_source = rng.uniform(0, 1, size=n)
    ext_source_observed = ext_source.copy()
    ext_source_observed[rng.random(n) < 0.3] = np.nan

    days_employed = -rng.integers(30, 5000, size=n).astype(float)
    not_working = rng.random(n) < 0.2
    days_employed[not_working] = 365243

    category = rng.choice(["Married", "Single", "Divorced"], size=n)

    default_prob = 1.0 / (1.0 + np.exp(-(2.0 - 4.0 * ext_source)))
    target = (rng.random(n) < default_prob).astype(int)

    return pd.DataFrame({
        "EXT_SOURCE": ext_source_observed,
        "DAYS_EMPLOYED": days_employed,
        "FAMILY_STATUS": category,
        "TARGET": target,
    })


def test_fit_woe_bins_creates_missing_bin_for_nan_column():
    df = _toy_credit_frame()
    woe_bins = fit_woe_bins(df, ["EXT_SOURCE"], n_bins=5)
    bins = woe_bins["EXT_SOURCE"].bin_stats["bin"].tolist()
    assert MISSING_LABEL in bins


def test_fit_woe_bins_creates_special_bin_for_special_value():
    df = _toy_credit_frame()
    woe_bins = fit_woe_bins(
        df, ["DAYS_EMPLOYED"], n_bins=5, special_values={"DAYS_EMPLOYED": [365243]},
    )
    bins = woe_bins["DAYS_EMPLOYED"].bin_stats["bin"].tolist()
    assert any("365243" in b for b in bins)
    # the special bin's row count should match the injected not-working rows exactly.
    special_row = woe_bins["DAYS_EMPLOYED"].bin_stats[
        woe_bins["DAYS_EMPLOYED"].bin_stats["bin"].str.contains("365243")
    ]
    assert int(special_row["count"].iloc[0]) == int((df["DAYS_EMPLOYED"] == 365243).sum())


def test_fit_woe_bins_categorical_creates_one_bin_per_category():
    df = _toy_credit_frame()
    woe_bins = fit_woe_bins(df, ["FAMILY_STATUS"])
    bins = set(woe_bins["FAMILY_STATUS"].bin_stats["bin"])
    assert bins == {"Married", "Single", "Divorced"}


def test_transform_woe_never_refits_and_maps_unseen_category_to_zero():
    train_df = _toy_credit_frame(n=800, seed=1)
    test_df = _toy_credit_frame(n=200, seed=2)
    test_df = pd.concat(
        [test_df, pd.DataFrame({
            "EXT_SOURCE": [0.5], "DAYS_EMPLOYED": [-100.0],
            "FAMILY_STATUS": ["NeverSeenBefore"], "TARGET": [0],
        })],
        ignore_index=True,
    )

    woe_bins = fit_woe_bins(train_df, ["EXT_SOURCE", "DAYS_EMPLOYED", "FAMILY_STATUS"], n_bins=5)
    transformed = transform_woe(test_df, woe_bins)

    assert "EXT_SOURCE_woe" in transformed.columns
    assert transformed["FAMILY_STATUS_woe"].iloc[-1] == pytest.approx(0.0)
    assert not transformed.filter(like="_woe").isna().any().any()


def test_select_features_by_iv_orders_by_descending_iv_and_filters_threshold():
    df = _toy_credit_frame(n=3000, seed=3)
    woe_bins = fit_woe_bins(df, ["EXT_SOURCE", "DAYS_EMPLOYED", "FAMILY_STATUS"], n_bins=5)
    selected = select_features_by_iv(woe_bins, min_iv=0.02)

    assert selected[0] == "EXT_SOURCE"  # the only feature this toy frame makes genuinely predictive.
    ivs = [woe_bins[f].total_iv for f in selected]
    assert ivs == sorted(ivs, reverse=True)
    assert all(woe_bins[f].total_iv >= 0.02 for f in selected)


def test_iv_report_lists_every_feature_with_a_strength_label():
    df = _toy_credit_frame(n=1500, seed=4)
    woe_bins = fit_woe_bins(df, ["EXT_SOURCE", "DAYS_EMPLOYED", "FAMILY_STATUS"], n_bins=5)
    report = iv_report(woe_bins)

    assert set(report["feature"]) == {"EXT_SOURCE", "DAYS_EMPLOYED", "FAMILY_STATUS"}
    assert list(report["total_iv"]) == sorted(report["total_iv"], reverse=True)
    assert report["strength"].isin(["not_useful", "weak", "medium", "strong", "suspicious"]).all()
