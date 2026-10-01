"""
train_transfer_learning.py, Week 8 CLI: builds a mature SOURCE fraud domain
and a small, newly-launched TARGET domain, trains a SOURCE model, then
compares three ways of getting a model for the target domain across a sweep
of target labeling budgets: zero-shot (no target labels at all), warm-start
transfer (continue the source booster on a little target data), and
training from scratch on the target data alone.

The SOURCE and TARGET domains are both built by src.transfer_learning.
domain_data.build_domain, calling this project's own Week 1 synthetic
generator twice with different n_transactions/fraud_rate/seed, modeling an
established product (SOURCE: 8000 transactions, 3.5% fraud rate, this
project's usual default) against a newly-launched, smaller, higher-risk one
(TARGET: 1200 transactions, 6% fraud rate, a disclosed, plausible
assumption that a new or less-vetted product starts out with a higher fraud
rate than a mature one, not a claim about any real product).

That prevalence difference alone was measured, directly in this project's
own sandbox testing, to NOT be enough of a domain shift for this comparison
to show anything real: this generator's V-column fraud signal is identical
in form regardless of seed or fraud_rate, so a source model applied
zero-shot to a target domain built only by varying those arguments scored
AUC-PR=0.9658, higher than the source model's own 0.8858 on its own domain,
because AUC-based ranking metrics do not see a prevalence-only shift. This
CLI's --n-shift-columns default (10, out of this generator's 20 V-columns)
exists to fix that: those columns are independently row-shuffled for the
TARGET domain only (see domain_data.py's build_domain and its module
docstring for the full explanation and the exact numbers this project
measured), breaking their correlation with isFraud in the target domain
specifically, so the source model's reliance on them becomes a genuine,
measurable liability rather than a hidden non-issue. Measured directly:
this drops zero-shot transfer's target AUC-PR from 0.9658 to 0.6058, the
real gap the rest of this comparison is measured against.

--query-additional-estimators (the number of new trees warm_start_transfer
adds on top of the source booster) and warm_start_transfer's own
min_child_weight default are both discussed at length in transfer_models.py's
module docstring, including a real, measured XGBoost anomaly this project
found and fixed: the library's own min_child_weight default (1.0) makes
warm-start continuation a complete, silent no-op whenever a small target
batch's total Hessian falls below it, which happens easily once a source
model is already confident. Read that docstring before changing either
default.

Every (strategy, budget) combination is logged to MLflow as its own run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.metrics import auc_pr
from src.transfer_learning.domain_data import build_domain
from src.transfer_learning.transfer_models import (
    apply_zero_shot,
    train_from_scratch,
    train_source_model,
    warm_start_transfer,
)

DEFAULT_TARGET_BUDGETS = [50, 100, 200, 400, 800]


def _subsample_target_train(target_train_df, budget: int, random_state: int):
    """Returns a stratified subsample of budget rows from target_train_df,
    or target_train_df itself unchanged if budget meets or exceeds its
    length (there is nothing left to subsample).
    """
    if budget >= len(target_train_df):
        return target_train_df
    sub_train, _ = train_test_split(
        target_train_df, train_size=budget,
        stratify=target_train_df["isFraud"], random_state=random_state,
    )
    return sub_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare zero-shot, warm-start transfer, and "
        "from-scratch models for a small, newly-launched fraud domain "
        "against a mature source domain's XGBoost model, across a sweep of "
        "target labeling budgets."
    )
    parser.add_argument("--source-n-transactions", type=int, default=8000)
    parser.add_argument("--source-fraud-rate", type=float, default=0.035)
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--target-n-transactions", type=int, default=1200)
    parser.add_argument("--target-fraud-rate", type=float, default=0.06)
    parser.add_argument("--target-seed", type=int, default=2)
    parser.add_argument(
        "--n-shift-columns", type=int, default=10,
        help="How many of the 20 V-columns (V1..V10 by default) are "
        "row-shuffled in the target domain only, to create a real pattern "
        "shift; see module docstring. 0 disables the shift entirely.",
    )
    parser.add_argument("--shift-seed", type=int, default=99)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--source-n-estimators", type=int, default=200)
    parser.add_argument("--n-additional-estimators", type=int, default=50)
    parser.add_argument("--from-scratch-n-estimators", type=int, default=50)
    parser.add_argument(
        "--target-budgets", type=int, nargs="+", default=DEFAULT_TARGET_BUDGETS,
        help="Target training set sizes to sweep over, each a fresh "
        "stratified subsample of the target domain's own training split.",
    )
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fraudlens_transfer_learning")
    args = parser.parse_args()

    shift_columns = [f"V{i}" for i in range(1, args.n_shift_columns + 1)] or None

    source_train, source_test, source_features = build_domain(
        n_transactions=args.source_n_transactions,
        fraud_rate=args.source_fraud_rate,
        seed=args.source_seed,
        test_size=args.test_size,
        start_transaction_id=1_000_000,
    )
    target_train, target_test, target_features = build_domain(
        n_transactions=args.target_n_transactions,
        fraud_rate=args.target_fraud_rate,
        seed=args.target_seed,
        test_size=args.test_size,
        start_transaction_id=5_000_000,
        shift_columns=shift_columns,
        shift_seed=args.shift_seed,
    )

    if source_features != target_features:
        raise ValueError(
            "Source and target domains produced different feature column "
            "sets; transfer learning requires a shared feature schema. "
            f"Only in source: {sorted(set(source_features) - set(target_features))}. "
            f"Only in target: {sorted(set(target_features) - set(source_features))}."
        )

    print(
        f"Source: {len(source_train)} train rows ({source_train['isFraud'].sum()} fraud), "
        f"{len(source_test)} test rows ({source_test['isFraud'].sum()} fraud)."
    )
    print(
        f"Target: {len(target_train)} train rows ({target_train['isFraud'].sum()} fraud), "
        f"{len(target_test)} test rows ({target_test['isFraud'].sum()} fraud). "
        f"{len(shift_columns) if shift_columns else 0} column(s) pattern-shifted: "
        f"{shift_columns}."
    )
    print(f"{len(source_features)} shared feature columns.")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    source_model = train_source_model(
        source_train, source_features, n_estimators=args.source_n_estimators,
    )
    source_on_source_proba = source_model.predict_proba(source_test)
    source_on_source_auc_pr = auc_pr(source_test["isFraud"], source_on_source_proba)
    source_on_source_auc_roc = roc_auc_score(source_test["isFraud"], source_on_source_proba)
    print(
        f"\nSource model on its OWN test set: AUC-ROC={source_on_source_auc_roc:.4f}, "
        f"AUC-PR={source_on_source_auc_pr:.4f}."
    )

    zero_shot_proba = apply_zero_shot(source_model, target_test)
    zero_shot_auc_pr = auc_pr(target_test["isFraud"], zero_shot_proba)
    zero_shot_auc_roc = roc_auc_score(target_test["isFraud"], zero_shot_proba)
    print(
        f"Zero-shot (source model, no target labels) on TARGET test set: "
        f"AUC-ROC={zero_shot_auc_roc:.4f}, AUC-PR={zero_shot_auc_pr:.4f}."
    )

    with mlflow.start_run(run_name="source_model"):
        mlflow.log_param("strategy", "source_on_source")
        mlflow.log_param("n_estimators", args.source_n_estimators)
        mlflow.log_metric("test_auc_roc", source_on_source_auc_roc)
        mlflow.log_metric("test_auc_pr", source_on_source_auc_pr)

    with mlflow.start_run(run_name="zero_shot"):
        mlflow.log_param("strategy", "zero_shot")
        mlflow.log_metric("target_test_auc_roc", zero_shot_auc_roc)
        mlflow.log_metric("target_test_auc_pr", zero_shot_auc_pr)

    results: List[Dict] = []
    print(f"\n{'budget':>6}  {'n_fraud':>7}  {'warm_auc_roc':>12}  {'warm_auc_pr':>11}  "
          f"{'scratch_auc_roc':>15}  {'scratch_auc_pr':>14}")
    for budget in args.target_budgets:
        sub_train = _subsample_target_train(target_train, budget, args.subsample_seed)
        n_fraud = int(sub_train["isFraud"].sum())
        if n_fraud < 2 or sub_train["isFraud"].nunique() < 2:
            print(f"{budget:>6}: skipped, only {n_fraud} fraud row(s) in this subsample.")
            continue

        warm_model = warm_start_transfer(
            source_model, sub_train, n_additional_estimators=args.n_additional_estimators,
        )
        scratch_model = train_from_scratch(
            sub_train, target_features, n_estimators=args.from_scratch_n_estimators,
        )

        warm_proba = warm_model.predict_proba(target_test)
        scratch_proba = scratch_model.predict_proba(target_test)

        entry = {
            "budget": len(sub_train),
            "n_fraud": n_fraud,
            "warm_auc_roc": float(roc_auc_score(target_test["isFraud"], warm_proba)),
            "warm_auc_pr": auc_pr(target_test["isFraud"], warm_proba),
            "scratch_auc_roc": float(roc_auc_score(target_test["isFraud"], scratch_proba)),
            "scratch_auc_pr": auc_pr(target_test["isFraud"], scratch_proba),
        }
        results.append(entry)

        print(
            f"{entry['budget']:>6}  {entry['n_fraud']:>7}  {entry['warm_auc_roc']:>12.4f}  "
            f"{entry['warm_auc_pr']:>11.4f}  {entry['scratch_auc_roc']:>15.4f}  "
            f"{entry['scratch_auc_pr']:>14.4f}"
        )

        for strategy, auc_roc_key, auc_pr_key in [
            ("warm_start", "warm_auc_roc", "warm_auc_pr"),
            ("from_scratch", "scratch_auc_roc", "scratch_auc_pr"),
        ]:
            with mlflow.start_run(run_name=f"{strategy}_budget_{entry['budget']}"):
                mlflow.log_param("strategy", strategy)
                mlflow.log_param("budget", entry["budget"])
                mlflow.log_param("n_fraud", entry["n_fraud"])
                mlflow.log_metric("target_test_auc_roc", entry[auc_roc_key])
                mlflow.log_metric("target_test_auc_pr", entry[auc_pr_key])

    if results:
        best_warm = max(results, key=lambda r: r["warm_auc_pr"])
        best_scratch = max(results, key=lambda r: r["scratch_auc_pr"])
        n_warm_wins = sum(1 for r in results if r["warm_auc_pr"] > r["scratch_auc_pr"])
        print(
            f"\nwarm_start beat from_scratch on target test AUC-PR in "
            f"{n_warm_wins}/{len(results)} budgets swept."
        )
        print(
            f"Best warm_start budget: {best_warm['budget']} "
            f"(AUC-PR={best_warm['warm_auc_pr']:.4f}). "
            f"Best from_scratch budget: {best_scratch['budget']} "
            f"(AUC-PR={best_scratch['scratch_auc_pr']:.4f})."
        )


if __name__ == "__main__":
    main()
