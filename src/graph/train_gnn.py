"""
train_gnn.py, Week 5 CLI: builds the transaction graph, trains the
GraphSAGE-based FraudGNN, evaluates it, and prints its metrics alongside
two quick tabular reference points computed on the exact same
chronological train/test split, so the comparison is real rather than
against a different run's numbers.

The tabular reference points are Logistic Regression (Week 3's simplest
model) and untuned XGBoost (Week 4's floor), both run without
hyperparameter tuning here deliberately: this script's job is to show
whether adding graph structure helps at all on this data, not to re-run
Week 4's own tuning work. Week 3 and Week 4's own scripts remain the place
to get this project's tuned tabular numbers.

Every run is logged to MLflow, same sqlite backend as the rest of this
project. The trained PyTorch model is logged with
serialization_format="pickle" rather than mlflow's newer default ("pt2",
a traced-graph export): pt2 tracing requires a concrete input_example and
FraudGNN's forward method takes two dictionaries of tensors (one per node
type and one per edge type), a shape torch.export's tracer is not built
to accept the way it accepts a single tensor or tuple of tensors. Pickle
is a plain, supported alternative for exactly this case, at the same
"only ever loading files this project itself produced" trust level this
project already accepted for mlflow.sklearn's skops trusted-types list in
Week 3 and Week 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from src.data.load_and_merge import load_and_merge
from src.graph.build_transaction_graph import DEFAULT_ENTITY_COLUMNS, build_hetero_graph
from src.graph.gnn_model import evaluate_gnn, train_gnn
from src.models.advanced_models import train_and_evaluate_advanced
from src.models.baseline_models import train_and_evaluate as train_and_evaluate_baseline
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Week 5's graph neural network "
        "fraud model, comparing it to quick, untuned tabular reference "
        "points on the same chronological train/test split."
    )
    parser.add_argument(
        "--transaction-csv", type=Path, default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--entity-columns", type=str, nargs="+", default=DEFAULT_ENTITY_COLUMNS,
    )
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fraudlens_graph_models")
    args = parser.parse_args()

    merged = load_and_merge(args.transaction_csv, args.identity_csv)
    train_df, test_df = prepare_train_test_features(merged, test_size=args.test_size)
    feature_columns = select_feature_columns(train_df)

    print(
        f"Train: {len(train_df)} rows ({train_df['isFraud'].sum()} fraud). "
        f"Test: {len(test_df)} rows ({test_df['isFraud'].sum()} fraud). "
        f"{len(feature_columns)} feature columns."
    )

    data = build_hetero_graph(
        train_df, test_df, feature_columns, entity_columns=args.entity_columns,
    )
    model, _ = train_gnn(
        data,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
    )
    gnn_result = evaluate_gnn(model, data, split="test")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name="graph_sage_gnn"):
        mlflow.log_param("entity_columns", ",".join(args.entity_columns))
        mlflow.log_param("hidden_channels", args.hidden_channels)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        mlflow.log_metric("test_auc_roc", gnn_result["auc_roc"])
        mlflow.log_metric("test_auc_pr", gnn_result["auc_pr"])
        mlflow.log_metric("test_ks_statistic", gnn_result["ks_statistic"])
        mlflow.log_metric("test_gini", gnn_result["gini"])
        mlflow.log_metric("n_test", gnn_result["n"])
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    print(
        f"\nGraphSAGE GNN: test AUC-ROC={gnn_result['auc_roc']:.4f}, "
        f"AUC-PR={gnn_result['auc_pr']:.4f}, KS={gnn_result['ks_statistic']:.4f}, "
        f"Gini={gnn_result['gini']:.4f}"
    )

    print(
        "\nQuick tabular reference points on the identical train/test split "
        "(untuned, for comparison only; see Week 3/4 for this project's "
        "tuned tabular numbers):"
    )
    logistic_result = train_and_evaluate_baseline(
        "logistic_regression", train_df, test_df, feature_columns,
    )
    print(
        f"Logistic Regression (Week 3 floor): test AUC-ROC="
        f"{logistic_result['test_auc_roc']:.4f}"
    )

    xgboost_result = train_and_evaluate_advanced(
        "xgboost", train_df, test_df, feature_columns, use_smote=False, tune=False,
    )
    print(
        f"XGBoost (Week 4 floor): test AUC-ROC={xgboost_result['test_auc_roc']:.4f}, "
        f"test AUC-PR={xgboost_result['test_auc_pr']:.4f}"
    )

    print(
        f"\nGraphSAGE GNN vs XGBoost floor, test AUC-PR: "
        f"{gnn_result['auc_pr']:.4f} vs {xgboost_result['test_auc_pr']:.4f}."
    )


if __name__ == "__main__":
    main()
