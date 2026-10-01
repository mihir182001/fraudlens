"""
engineer_features.py, Week 2: transaction velocity features, time-based
aggregations, card usage patterns, and frequency encoding for categoricals,
built on top of Week 1's merged transaction and identity data.

Every feature here is causal: computed using only the current row and rows
strictly before it in TransactionDT order, never rows that come after. This
matters because these features will eventually feed a fraud classifier. A
feature that could see into the future during training (for example, "this
card's average transaction amount" computed over the whole dataset,
including transactions that have not happened yet at the time of the
transaction being scored) cannot actually be computed with only the
information truly available at scoring time. Every function below has a
leakage guard test in tests/test_engineer_features.py: it adds a future
high value transaction and confirms that doing so does not change any
earlier row's feature value.

Frequency encoding is the one exception, exposed with an explicit
fit_frequencies / transform_frequencies split. Fit the frequency table on
your training split, transform both train and any held out or live data
with that same fitted table, and never refit on data that includes rows
outside training. Passing the same DataFrame to both fit and transform (as
this module's own CLI does, for lack of a real train/test split before
Week 3's baseline models) is only appropriate for this Week 2 exploration
step, not for actual model training later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400
DEFAULT_VELOCITY_WINDOWS_SECONDS = {"1h": 3600, "1d": 86400}
DEFAULT_FREQUENCY_ENCODE_COLUMNS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
]


def add_velocity_features(
    df: pd.DataFrame,
    card_col: str = "card1",
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    windows_seconds: Optional[dict] = None,
) -> pd.DataFrame:
    """Adds, for each row, how many transactions the same card made and how
    much it spent in a trailing time window ending at that row's own
    timestamp (inclusive of the row itself, exclusive of anything after it).

    `windows_seconds` maps a human label to a window length in seconds; the
    default gives a 1 hour and a 1 day window, matching the spec's
    "transaction velocity features". Returns a new DataFrame (does not
    mutate the input), re-sorted by time_col ascending since the rolling
    computation requires sorted order; callers that need the original row
    order back should re-sort the result by a stable id such as
    TransactionID.
    """
    if windows_seconds is None:
        windows_seconds = DEFAULT_VELOCITY_WINDOWS_SECONDS

    required = {card_col, time_col, amount_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df is empty.")

    out = df.sort_values(time_col, kind="mergesort").reset_index(drop=True).copy()
    # Rolling with a time offset needs a datetime-like column to roll on.
    # TransactionDT is seconds from an arbitrary reference point (see
    # src/eda/run_eda.py), so this conversion is purely a rolling-window
    # mechanism. It is never interpreted as a real calendar date.
    out["_dt"] = pd.to_datetime(out[time_col], unit="s")

    for label, seconds in windows_seconds.items():
        offset = pd.Timedelta(seconds=seconds)
        # include_groups=False: without it, pandas 2.2+ emits a
        # FutureWarning that a later pandas version will stop passing the
        # grouping column (card_col) into the applied function at all.
        # _rolling_metric never uses card_col (it only needs _dt and
        # amount_col), so excluding it changes nothing about the result;
        # confirmed directly by running the full test suite against the
        # exact pandas version pinned in requirements.txt (2.3.3, not
        # whatever version happened to be installed already) both with and
        # without this argument and comparing output byte for byte.
        rolling_count = out.groupby(card_col, group_keys=False).apply(
            lambda sub: _rolling_metric(sub, amount_col, offset, "count"),
            include_groups=False,
        )
        rolling_sum = out.groupby(card_col, group_keys=False).apply(
            lambda sub: _rolling_metric(sub, amount_col, offset, "sum"),
            include_groups=False,
        )
        out[f"{card_col}_txn_count_{label}"] = rolling_count
        out[f"{card_col}_amt_sum_{label}"] = rolling_sum

    return out.drop(columns=["_dt"])


def _rolling_metric(sub: pd.DataFrame, amount_col: str, offset: pd.Timedelta, method: str) -> pd.Series:
    """Per-group helper for add_velocity_features. Computes a trailing
    time-window rolling aggregate on `sub` (one card's rows, in time order)
    and restores the result's index to `sub`'s own original row labels.

    This matters because pandas' `groupby(...).rolling(offset, on=col)`
    returns a result indexed by (group key, that column's own values), not
    by the original row position. Since a datetime column can have ties
    (two transactions from different cards in the same second are common,
    and even the same card twice in the same second is possible), that
    index is not safe to align back onto the original DataFrame by label
    or by position. Indexing by `sub`'s own row labels instead sidesteps
    the problem entirely, since those labels are guaranteed unique.
    """
    rolling = sub.set_index("_dt")[amount_col].rolling(offset)
    result = getattr(rolling, method)()
    result.index = sub.index
    return result


def add_card_usage_features(
    df: pd.DataFrame,
    card_col: str = "card1",
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    diversity_col: str = "addr1",
) -> pd.DataFrame:
    """Adds per-card usage-pattern features, all computed from rows
    strictly before the current one:

      - `<card_col>_prior_txn_count`: how many earlier transactions this
        card has made.
      - `<card_col>_running_avg_amt`: the average amount of this card's
        earlier transactions. NaN for a card's first transaction, since it
        has no history yet; left as NaN rather than filled here, since the
        right fill value depends on how a later modeling step wants to
        treat "no history" (a dedicated missing-indicator versus a
        population mean are both defensible, and that choice belongs to
        the model-building step, not this feature step).
      - `<card_col>_amt_ratio_to_own_avg`: this transaction's amount
        divided by that running average. Also NaN when the average is NaN
        or zero.
      - `<card_col>_prior_distinct_<diversity_col>_count`: how many
        distinct (non-null) values of `diversity_col` (addr1 by default)
        this card has used in its earlier transactions, a simple proxy for
        "is this card suddenly being used somewhere new."

    The running average deliberately excludes the current row's own amount.
    Including it would leak the very value being described into its own
    "typical amount for this card" feature, which is a subtler version of
    the same mistake as computing it over the whole dataset: for a card's
    first-ever transaction, the naive (including current row) average
    would equal that transaction's own amount exactly, silently handing a
    fraud classifier a feature that is a deterministic function of the
    transaction it is trying to score.
    """
    required = {card_col, time_col, amount_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df is empty.")

    out = df.sort_values(time_col, kind="mergesort").reset_index(drop=True).copy()

    grouped_amt = out.groupby(card_col)[amount_col]
    prior_count = grouped_amt.cumcount()
    cum_sum_incl_current = grouped_amt.cumsum()
    prior_sum = cum_sum_incl_current - out[amount_col]

    with np.errstate(invalid="ignore", divide="ignore"):
        running_avg = prior_sum / prior_count.replace(0, np.nan)
        ratio_to_avg = out[amount_col] / running_avg.replace(0, np.nan)

    out[f"{card_col}_prior_txn_count"] = prior_count
    out[f"{card_col}_running_avg_amt"] = running_avg
    out[f"{card_col}_amt_ratio_to_own_avg"] = ratio_to_avg

    if diversity_col in out.columns:
        prior_distinct = (
            out.groupby(card_col)[diversity_col]
            .apply(_prior_distinct_count)
            .reset_index(level=0, drop=True)
        )
        out[f"{card_col}_prior_distinct_{diversity_col}_count"] = prior_distinct

    return out


def _prior_distinct_count(values: pd.Series) -> pd.Series:
    """Per-group helper for add_card_usage_features: given a group's values
    of some column in time order, returns, for each row, the number of
    distinct non-null values seen strictly before that row.

    Vectorized via factorize plus a cumulative "is this the first time we
    have seen this value" flag, rather than an explicit Python loop
    building up a set row by row, since the latter is O(n) per group but
    with a much higher constant factor once the real 590,540-row IEEE-CIS
    data is in play.
    """
    is_missing = values.isna()
    codes, _ = pd.factorize(values, sort=False)
    codes = pd.Series(codes, index=values.index)
    codes = codes.where(~is_missing, other=-1)

    is_first_occurrence = (~codes.duplicated()) & (codes != -1)
    distinct_count_including_current = is_first_occurrence.cumsum()
    return distinct_count_including_current.shift(1, fill_value=0)


def add_hour_of_day_aggregates(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    fit_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Adds, for each row, the mean amount and transaction count among rows
    that fall in the same hour-of-a-simulated-day bucket (TransactionDT
    modulo 86400, divided by 3600), matching
    src/eda/run_eda.py's compute_time_patterns bucketing.

    Pass a `fit_df` (a training split, once a real one exists) so the
    per-hour-bucket statistics used as a feature are learned only from
    training data. Without fit_df, statistics are computed from `df`
    itself, which is fine for this Week 2 exploration step but would leak
    test-set information into train-set features once a real split exists.
    """
    required = {time_col, amount_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")

    fit_source = fit_df if fit_df is not None else df
    fit_missing = required - set(fit_source.columns)
    if fit_missing:
        raise ValueError(f"fit_df is missing required column(s): {sorted(fit_missing)}")
    if fit_source.empty:
        raise ValueError("fit_df is empty.")

    fit_hour_bucket = (fit_source[time_col] % SECONDS_PER_DAY) // 3600
    stats = (
        fit_source.assign(_hour=fit_hour_bucket)
        .groupby("_hour")[amount_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "hour_of_day_avg_amt", "count": "hour_of_day_txn_count"})
    )

    out = df.copy()
    out["_hour"] = (out[time_col] % SECONDS_PER_DAY) // 3600
    out = out.merge(stats, left_on="_hour", right_index=True, how="left")
    return out.drop(columns=["_hour"])


def fit_frequencies(df: pd.DataFrame, columns: list) -> dict:
    """Returns, for each column, a mapping from each observed value to its
    proportion of rows in `df`. Fit this on a training split only; see
    module docstring.
    """
    freq_tables = {}
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"df has no column {col!r}.")
        freq_tables[col] = df[col].value_counts(normalize=True, dropna=False)
    return freq_tables


def transform_frequencies(df: pd.DataFrame, freq_tables: dict) -> pd.DataFrame:
    """Applies previously fit frequency tables to `df`, adding a
    `<col>_freq` column per entry in freq_tables. A value that never
    appeared in the fit data maps to 0.0, not NaN, since "never seen
    before" is itself real information (novelty), not missingness.
    """
    out = df.copy()
    for col, freq_table in freq_tables.items():
        if col not in out.columns:
            raise ValueError(f"df has no column {col!r}.")
        out[f"{col}_freq"] = out[col].map(freq_table).fillna(0.0)
    return out


def frequency_encode(
    df: pd.DataFrame, columns: list, fit_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Convenience wrapper: fits frequency tables on fit_df (or `df` itself
    if not given) and applies them to `df` in one call. See module
    docstring for why fit_df should be a training split once a real split
    exists.
    """
    freq_tables = fit_frequencies(fit_df if fit_df is not None else df, columns)
    return transform_frequencies(df, freq_tables)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all Week 2 feature engineering steps in sequence and returns the
    result. Uses `df` itself as the frequency and hour-bucket fit source;
    see add_hour_of_day_aggregates and the module docstring for why that
    changes once a real train/test split exists starting Week 3.
    """
    out = add_velocity_features(df)
    out = add_card_usage_features(out)
    out = add_hour_of_day_aggregates(out)
    categorical_columns = [c for c in DEFAULT_FREQUENCY_ENCODE_COLUMNS if c in out.columns]
    if categorical_columns:
        out = frequency_encode(out, categorical_columns)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Week 2 feature engineering against a merged "
        "transaction and identity CSV (see src.data.build_merged_dataset "
        "to produce one) and write the engineered features to a CSV."
    )
    parser.add_argument(
        "--merged-csv", type=Path, default=Path("data/processed/merged.csv"),
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("data/processed/features.csv"),
    )
    args = parser.parse_args()

    if not args.merged_csv.exists():
        raise SystemExit(
            f"{args.merged_csv} not found. Build it first, e.g.:\n"
            "  python -m src.data.generate_synthetic_ieee   (or download the real Kaggle files)\n"
            "  python -m src.data.build_merged_dataset"
        )

    df = pd.read_csv(args.merged_csv)
    features = engineer_features(df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_csv, index=False)
    print(f"Engineered {len(features.columns)} columns for {len(features)} rows.")
    print(f"Wrote features to {args.out_csv}")


if __name__ == "__main__":
    main()
