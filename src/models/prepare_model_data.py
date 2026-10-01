"""
prepare_model_data.py, Week 3: builds a leakage-safe train and test feature
matrix for the baseline models, by chaining Week 2's feature functions in
the specific order a real train/test split requires.

Order matters here in a way that is easy to get wrong. The velocity and
card-usage features from src/features/engineer_features.py
(add_velocity_features, add_card_usage_features) are causal: a row's value
only ever depends on that same card's genuinely earlier transactions,
whether those earlier transactions end up on the train or the test side of
a later split. So they are computed ONCE, on the full chronological
history, before any split happens, exactly as they would be in production,
where a live transaction's velocity features are computed against real
past transactions regardless of how a model's training set happened to be
drawn.

The hour-of-day aggregates and frequency encoding are different: they are
statistics fit on a set of rows, not causal per-row computations. Like
Week 2's own docstrings already flagged, once a real train and test split
exists, these must be fit only on the train split and then applied (never
refit) to the test split, or test-set metrics would be inflated by
information the model would not actually have at real deployment time.
This module is where that switch happens.

Splitting itself is chronological, not random: TransactionDT is a real
ordering (even though it is not a real calendar date; see
src/eda/run_eda.py), so a random split would put some later rows in
training and some earlier rows in test, which does not match how this
model would actually be used in production (trained on transactions seen
so far, scored against transactions that have not happened yet).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from src.features.engineer_features import (
    DEFAULT_FREQUENCY_ENCODE_COLUMNS,
    add_card_usage_features,
    add_hour_of_day_aggregates,
    add_velocity_features,
    frequency_encode,
)

NON_FEATURE_COLUMNS = {"TransactionID", "isFraud"}


def chronological_train_test_split(
    df: pd.DataFrame, time_col: str = "TransactionDT", test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits df into a train and test set by time, not at random: every
    train row's time_col is less than or equal to every test row's, so
    the test set is strictly the later portion of the timeline.

    Returns copies sorted by time_col, with each side's original relative
    row order otherwise preserved. Raises if test_size is not in (0, 1), or
    if the resulting split would leave either side empty.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if time_col not in df.columns:
        raise ValueError(f"df has no column {time_col!r}.")
    if df.empty:
        raise ValueError("df is empty.")

    ordered = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    split_index = int(round(len(ordered) * (1.0 - test_size)))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError(
            f"test_size={test_size} leaves an empty train or test split for "
            f"{len(ordered)} rows."
        )

    train_df = ordered.iloc[:split_index].reset_index(drop=True)
    test_df = ordered.iloc[split_index:].reset_index(drop=True)

    if train_df[time_col].max() > test_df[time_col].min():
        raise AssertionError(
            "Chronological split invariant violated: a train row is later "
            "than a test row. This should be impossible given the sort "
            "above and indicates a real bug."
        )
    return train_df, test_df


def prepare_train_test_features(
    merged_df: pd.DataFrame,
    time_col: str = "TransactionDT",
    test_size: float = 0.2,
    frequency_encode_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds a leakage-safe engineered train and test feature pair from
    Week 1's merged transaction and identity data. See the module
    docstring for why the steps below happen in exactly this order.
    """
    if frequency_encode_columns is None:
        frequency_encode_columns = DEFAULT_FREQUENCY_ENCODE_COLUMNS

    # Causal features first, on the full history, before any split.
    with_velocity = add_velocity_features(merged_df, time_col=time_col)
    with_card_usage = add_card_usage_features(with_velocity, time_col=time_col)

    train_df, test_df = chronological_train_test_split(
        with_card_usage, time_col=time_col, test_size=test_size
    )

    # Fit-based features from here on: fit on train, apply to both.
    train_df = add_hour_of_day_aggregates(train_df, time_col=time_col)
    test_df = add_hour_of_day_aggregates(test_df, time_col=time_col, fit_df=train_df)

    encode_columns = [c for c in frequency_encode_columns if c in train_df.columns]
    if encode_columns:
        train_df = frequency_encode(train_df, encode_columns)
        test_df = frequency_encode(test_df, encode_columns, fit_df=train_df)

    return train_df, test_df


def select_feature_columns(df: pd.DataFrame, target_col: str = "isFraud") -> List[str]:
    """Returns the columns to actually feed a baseline model: every numeric
    or boolean column except the target and TransactionID (a row id with
    no predictive meaning of its own).

    Raw categorical columns (ProductCD, card4, card6, P_emaildomain,
    R_emaildomain, M1-M9, DeviceType, DeviceInfo, and so on) are
    deliberately excluded here rather than one-hot encoded: the
    frequency-encoded numeric version of each one that Week 2's
    frequency_encode produces already carries that column's signal in a
    form these baseline models can use directly, and adding a one-hot
    encoded version on top would just be redundant for this Week 3 floor.
    Boolean columns (such as load_and_merge's has_identity_match) are
    included explicitly, since pandas own "numeric" dtype selection
    excludes bool by default even though it is a perfectly usable 0/1
    feature for these models.
    """
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.discard(target_col)  # in case a caller passes a different target name
    excluded.add(target_col)
    numeric_and_bool = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [c for c in numeric_and_bool if c not in excluded]
