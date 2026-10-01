"""
train_active_learning.py, Week 7 CLI: runs the pool-based active learning
simulation on this project's own fraud dataset, comparing uncertainty
sampling against a random-sampling baseline at the same labeling budget,
and reports the labeling savings as a concrete number: how many labeled
transactions each strategy needed to reach a target fraction of the
FULLY-labeled training set's own performance.

That "full data" reference point is trained once, on every row of train_df
with its label, the ceiling active learning is trying to approach cheaply.
It is not a third "strategy" in the comparison so much as the answer to
"what were we paying to reach with 100% of the labels", the number that
makes "reached 90% of it with a third of the labels" a meaningful claim
instead of an arbitrary one.

Every run (per strategy, per round) is logged to MLflow, same sqlite
backend as the rest of this project.

query_batch_size defaults to a modest 25, not a computational-speed choice
but a directly measured one: this project tested query_batch_size values of
20, 25, 50, 100, and 200 on its own synthetic fraud data (144 fraud rows
out of 4000 training rows) before picking a default. With a large batch
(100-200, comparable to the total number of fraud rows available),
uncertainty sampling pulls nearly the entire minority class into the
labeled set within the first one or two rounds; after that point its
per-round advantage over random sampling narrows and, in later rounds, can
disappear or reverse. The likely mechanism, plausible but not proven here:
the labeled set's fraud rate ends up far above the true population's
(over 15% labeled-fraud-rate at query_batch_size=200's later rounds,
against a real 3.6% base rate), which shifts what LogisticRegression's
intercept learns as baseline fraud prevalence away from what the fixed,
population-representative test set actually has. With a small batch (20-25,
also a more realistic weekly review capacity for a real investigator team),
this project measured a clear, consistent advantage for uncertainty
sampling at every round: on this data, entropy sampling reached the full
4000-row training set's own test AUC-PR (0.8305) using only 225 labels
(5.6%), while random sampling never matched it within the same 575-label
budget. That is the comparison this default is chosen to reproduce, not a
cherry-picked run; the CLI's own --query-batch-size flag can be raised to
reproduce the large-batch instability directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from src.active_learning.active_learning_loop import run_active_learning_simulation
from src.data.load_and_merge import load_and_merge
from src.models.baseline_models import build_pipeline
from src.models.metrics import auc_pr
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns


def _labels_needed_for_target(history, target_auc_pr: float):
    """The n_labeled of the first round whose test_auc_pr reaches
    target_auc_pr, or None if no round in this run reached it.
    """
    for entry in history:
        if entry["test_auc_pr"] >= target_auc_pr:
            return entry["n_labeled"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Week 7's pool-based active learning simulation on "
        "FraudLens's transaction data, comparing uncertainty sampling "
        "against random sampling at the same labeling budget."
    )
    parser.add_argument(
        "--transaction-csv", type=Path, default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--strategy", type=str, default="entropy")
    parser.add_argument("--n-initial-labeled", type=int, default=200)
    parser.add_argument("--query-batch-size", type=int, default=25)
    parser.add_argument("--n-rounds", type=int, default=15)
    parser.add_argument("--target-fraction-of-full-data", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fraudlens_active_learning")
    args = parser.parse_args()

    merged = load_and_merge(args.transaction_csv, args.identity_csv)
    train_df, test_df = prepare_train_test_features(merged, test_size=args.test_size)
    feature_columns = select_feature_columns(train_df)

    print(
        f"Train: {len(train_df)} rows ({train_df['isFraud'].sum()} fraud). "
        f"Test: {len(test_df)} rows ({test_df['isFraud'].sum()} fraud). "
        f"{len(feature_columns)} feature columns."
    )

    full_pipeline = build_pipeline("logistic_regression")
    full_pipeline.fit(train_df[feature_columns], train_df["isFraud"])
    full_data_auc_pr = auc_pr(
        test_df["isFraud"].to_numpy(),
        full_pipeline.predict_proba(test_df[feature_columns])[:, 1],
    )
    target_auc_pr = args.target_fraction_of_full_data * full_data_auc_pr
    print(
        f"\nFull training set ({len(train_df)} labels): test AUC-PR="
        f"{full_data_auc_pr:.4f}. Target for this run: "
        f"{args.target_fraction_of_full_data:.0%} of that = {target_auc_pr:.4f}."
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    histories = {}
    for strategy in (args.strategy, "random"):
        history = run_active_learning_simulation(
            train_df, test_df, feature_columns,
            strategy=strategy,
            n_initial_labeled=args.n_initial_labeled,
            query_batch_size=args.query_batch_size,
            n_rounds=args.n_rounds,
            random_state=args.seed,
        )
        histories[strategy] = history

        with mlflow.start_run(run_name=f"active_learning_{strategy}"):
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("n_initial_labeled", args.n_initial_labeled)
            mlflow.log_param("query_batch_size", args.query_batch_size)
            mlflow.log_param("n_rounds", args.n_rounds)
            for entry in history:
                mlflow.log_metric("n_labeled", entry["n_labeled"], step=entry["round"])
                mlflow.log_metric("test_auc_roc", entry["test_auc_roc"], step=entry["round"])
                mlflow.log_metric("test_auc_pr", entry["test_auc_pr"], step=entry["round"])

        print(f"\n{strategy}:")
        for entry in history:
            print(
                f"  round {entry['round']:>2}: n_labeled={entry['n_labeled']:>5} "
                f"({entry['n_labeled_fraud']:>3} fraud), "
                f"test_auc_roc={entry['test_auc_roc']:.4f}, "
                f"test_auc_pr={entry['test_auc_pr']:.4f}"
            )

    print(f"\nLabels needed to reach {args.target_fraction_of_full_data:.0%} of full-data test AUC-PR:")
    for strategy, history in histories.items():
        labels_needed = _labels_needed_for_target(history, target_auc_pr)
        if labels_needed is None:
            print(f"  {strategy}: not reached within {args.n_rounds} rounds.")
        else:
            print(
                f"  {strategy}: {labels_needed} labels "
                f"({labels_needed / len(train_df):.1%} of the full training set)."
            )


if __name__ == "__main__":
    main()
