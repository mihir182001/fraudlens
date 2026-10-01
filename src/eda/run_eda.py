"""
run_eda.py, Week 1: the four EDA items the build spec calls for.
Fraud rate, missing-value rates, feature distributions by fraud label, and
time-based patterns, all computed generically off of whatever merged
transaction+identity DataFrame is passed in.

Every function here takes a DataFrame and returns real numbers computed
from it. Nothing in this module hardcodes or repeats the build spec's
target numbers (e.g. "AUC-ROC 0.927"). Those are planning figures, not
measurements, and this project's own discipline (established in the
sibling SearchLens project) is to report only what the code actually
measured, on whatever data it was actually run against. Run against
generate_synthetic_ieee.py's synthetic data, every number below is a real
number about that synthetic data. Run against the real Kaggle files, every
number becomes a real finding about real fraud.

`TransactionDT` caveat: per the IEEE-CIS competition's own data
description, this column is a count of seconds elapsed from an arbitrary,
undisclosed reference point, NOT a real Unix timestamp or calendar date.
compute_time_patterns therefore only derives *relative* time features from
it (elapsed-day bucket since the first transaction in the data, and
second-within-a-simulated-day, i.e. TransactionDT modulo 86400) rather than
ever converting it to a real date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400


def compute_fraud_rate(df: pd.DataFrame) -> dict:
    """Overall fraud rate plus the raw counts it's computed from, so the
    rate is always traceable back to n_fraud / n_total rather than just a
    bare float.
    """
    if "isFraud" not in df.columns:
        raise ValueError("df has no isFraud column.")
    n_total = len(df)
    if n_total == 0:
        raise ValueError("df is empty.")
    n_fraud = int(df["isFraud"].sum())
    return {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legitimate": n_total - n_fraud,
        "fraud_rate": n_fraud / n_total,
    }


def compute_missing_value_rates(df: pd.DataFrame) -> pd.Series:
    """Fraction of missing values per column, sorted descending. Works over
    every column actually present in `df`. There is no hardcoded column list, so
    this reports real numbers whether run against the reduced synthetic
    schema or the real 434-column IEEE-CIS files.
    """
    if df.empty:
        raise ValueError("df is empty.")
    return df.isna().mean().sort_values(ascending=False)


def compute_feature_distributions_by_fraud(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Per-column summary statistics (count, mean, std, min, quartiles,
    max) split by isFraud, for numeric columns.

    If `columns` isn't given, auto-selects every numeric column except
    TransactionID and isFraud themselves (an ID column has no meaningful
    distribution, and isFraud is the grouping key). This is how the
    function stays generic across the reduced synthetic V1-V20 and the
    real V1-V339 without a code change.
    """
    if "isFraud" not in df.columns:
        raise ValueError("df has no isFraud column.")

    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if c not in ("TransactionID", "isFraud")]

    if not columns:
        raise ValueError("No numeric feature columns available to summarize.")

    return df.groupby("isFraud")[columns].describe().transpose()


def compute_time_patterns(df: pd.DataFrame) -> dict:
    """Fraud rate broken down by two relative-time views of TransactionDT
    (see module docstring for why this never treats TransactionDT as a real
    date):

      - by_elapsed_day: fraud rate per day-since-first-transaction bucket
        (TransactionDT minus min(TransactionDT)) divided by 86400. Shows
        whether fraud rate drifts over the observed window.
      - by_second_of_day: fraud rate per hour-of-a-simulated-day bucket
        (TransactionDT modulo 86400) divided by 3600. Shows whether fraud
        clusters at particular times within a repeating daily cycle, a real pattern
        publicly documented for this competition even though the absolute
        calendar dates are undisclosed.
    """
    required = {"TransactionDT", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df is empty.")

    elapsed = df["TransactionDT"] - df["TransactionDT"].min()
    day_bucket = (elapsed // SECONDS_PER_DAY).astype(int)
    hour_of_day_bucket = ((df["TransactionDT"] % SECONDS_PER_DAY) // 3600).astype(int)

    # Group by the bucket Series directly (rather than df.assign()-ing it as
    # a new column onto a wide DataFrame first). This avoids pandas'
    # "highly fragmented" PerformanceWarning on the real 434-column IEEE-CIS
    # data, and only ever needs the isFraud column plus the bucket anyway.
    by_elapsed_day = (
        df["isFraud"].groupby(day_bucket).agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n_transactions"})
    )
    by_hour_of_day = (
        df["isFraud"].groupby(hour_of_day_bucket).agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n_transactions"})
    )

    return {
        "by_elapsed_day": by_elapsed_day,
        "by_hour_of_day": by_hour_of_day,
    }


def run_eda(df: pd.DataFrame) -> dict:
    """Runs all four EDA items and returns them together. Kept as one
    function so a CLI or a notebook can call this once and get everything
    the Week 1 spec asks for, while each piece above stays independently
    testable and reusable.
    """
    return {
        "fraud_rate": compute_fraud_rate(df),
        "missing_value_rates": compute_missing_value_rates(df),
        "feature_distributions_by_fraud": compute_feature_distributions_by_fraud(df),
        "time_patterns": compute_time_patterns(df),
    }


def _json_safe(obj):
    """Recursively converts the pandas/numpy objects run_eda() returns into
    plain JSON-serializable Python types, for writing the report to disk.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="index"))
    if isinstance(obj, pd.Series):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Week 1 EDA (fraud rate, missing values, feature "
        "distributions, time patterns) against a merged transaction+"
        "identity CSV and print a summary."
    )
    parser.add_argument(
        "--merged-csv", type=Path, default=Path("data/processed/merged.csv"),
        help="Path to a merged transaction+identity CSV (see "
        "src.data.load_and_merge to produce one).",
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("data/processed/eda_report.json"),
        help="Where to write the full EDA report as JSON.",
    )
    args = parser.parse_args()

    if not args.merged_csv.exists():
        raise SystemExit(
            f"{args.merged_csv} not found. Build it first, e.g.:\n"
            "  python -m src.data.generate_synthetic_ieee   (or download the real Kaggle files)\n"
            "  python -m src.data.build_merged_dataset"
        )

    df = pd.read_csv(args.merged_csv)
    report = run_eda(df)

    fraud = report["fraud_rate"]
    print(f"Fraud rate: {fraud['fraud_rate']:.4%} ({fraud['n_fraud']} / {fraud['n_total']})")
    print("\nTop 10 columns by missing-value rate:")
    print(report["missing_value_rates"].head(10).to_string())

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(_json_safe(report), f, indent=2)
    print(f"\nFull EDA report written to {args.out_json}")


if __name__ == "__main__":
    main()
