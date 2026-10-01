"""
baseline_models.py, Week 3: Logistic Regression, Decision Tree, and Random
Forest baselines, evaluated by AUC-ROC to establish the performance floor
the spec calls for, with every run logged to MLflow.

These are deliberately left as plain, unweighted, untuned scikit-learn
defaults (aside from a couple of stability settings noted inline). Fraud
detection has severe class imbalance (a real, measured 3.5 percent positive
rate on the synthetic data this project runs against; the real IEEE-CIS
rate will be whatever Week 1's real EDA measures once you have the actual
Kaggle files), and Week 4's spec explicitly introduces SMOTE, hyperparameter
tuning, and threshold optimisation as the next steps. Establishing an
honest, unimproved floor here is what makes Week 4's improvement over it a
real, measurable claim instead of an assumed one.

Missing values (several engineered features are NaN by construction, such
as a card's first transaction having no prior average) are imputed with
the training set's median, since none of these three model types accept
NaN directly. The imputer is part of each model's own scikit-learn
Pipeline and is fit only on the training data passed to it, the same
train-only-fitting discipline used throughout Week 2 and Week 3's feature
code.

Every run is logged to MLflow against a local sqlite database
(sqlite:///mlflow.db by default), not a plain filesystem store
(file:./mlruns). That is not a style choice: mlflow 3.16.1, the version
actually pinned in requirements.txt, was confirmed directly to raise an
MlflowException for the filesystem store, since MLflow has put that
backend into maintenance mode ahead of removing it. A local sqlite file
is just as local and account-free as a folder of files, so it keeps this
project's "no external server needed" goal intact while tracking real
MLflow behavior instead of a deprecated code path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.load_and_merge import load_and_merge
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns

RANDOM_STATE = 42

# mlflow 3.16.1 serializes sklearn models with skops by default instead of
# pickle, and skops refuses to load a file back unless every type inside it
# is explicitly marked trusted, since a pickle-style file can otherwise run
# arbitrary code on load. Confirmed directly against this project's own
# three model types: LogisticRegression's fitted pipeline only needs
# numpy.dtype trusted, while DecisionTreeClassifier and RandomForestClassifier
# additionally store their fitted trees in sklearn.tree._tree.Tree, which
# skops also flags. Trusting these two specific types is safe here because
# every model logged by this module was just trained in this same process,
# not loaded from an external or untrusted file; the trust list is kept to
# exactly what was confirmed necessary rather than trusting every type skops
# reports.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def build_pipeline(model_name: str) -> Pipeline:
    """Returns a scikit-learn Pipeline (imputer plus model) for one of
    "logistic_regression", "decision_tree", or "random_forest".

    Logistic Regression additionally needs its inputs on a comparable
    scale to converge reliably (the tree-based models are invariant to
    monotonic per-feature scaling, so they skip that step). max_iter is
    raised from scikit-learn's default of 100 to 1000, a plain convergence
    fix rather than a modeling choice: the default was verified directly to
    raise a ConvergenceWarning on this project's engineered features before
    this change.
    """
    if model_name == "logistic_regression":
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
    if model_name == "decision_tree":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])
    raise ValueError(
        f"Unknown model_name {model_name!r}; expected one of "
        "'logistic_regression', 'decision_tree', 'random_forest'."
    )


def train_and_evaluate(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "isFraud",
) -> Dict:
    """Fits one baseline model on train_df and evaluates it on test_df.

    Returns a dict with the fitted pipeline, the real measured AUC-ROC
    (train and test), and the train/test row and fraud counts the AUC-ROC
    is computed from, so the number is always traceable back to what it was
    measured on rather than reported as a bare float.
    """
    x_train = train_df[feature_columns]
    y_train = train_df[target_col]
    x_test = test_df[feature_columns]
    y_test = test_df[target_col]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training set has only one class present; AUC-ROC is undefined. "
            "This can happen with a very small or unluckily split synthetic "
            "sample; try a larger --n-transactions or a different seed."
        )
    if y_test.nunique() < 2:
        raise ValueError(
            "Test set has only one class present; AUC-ROC is undefined. "
            "This can happen with a very small or unluckily split synthetic "
            "sample; try a larger --n-transactions or a different seed."
        )

    pipeline = build_pipeline(model_name)
    pipeline.fit(x_train, y_train)

    train_proba = pipeline.predict_proba(x_train)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "train_auc_roc": roc_auc_score(y_train, train_proba),
        "test_auc_roc": roc_auc_score(y_test, test_proba),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_fraud": int(y_train.sum()),
        "n_test_fraud": int(y_test.sum()),
        "n_features": len(feature_columns),
    }


def run_all_baselines(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: List[str]
) -> List[Dict]:
    """Trains and evaluates all three Week 3 baseline models and returns
    their results in the order the spec lists them.
    """
    model_names = ["logistic_regression", "decision_tree", "random_forest"]
    return [train_and_evaluate(name, train_df, test_df, feature_columns) for name in model_names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Week 3's three baseline models "
        "(Logistic Regression, Decision Tree, Random Forest) against a "
        "chronological train/test split of the merged transaction and "
        "identity data, logging every run to MLflow."
    )
    parser.add_argument(
        "--transaction-csv", type=Path, default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
        help="Where MLflow stores run data. The default keeps everything "
        "local to this project, no server or account needed. A plain "
        "filesystem URI such as file:./mlruns is deliberately not used: "
        "mlflow 3.16.1 (the version pinned in requirements.txt) has put "
        "that backend into maintenance mode and raises an error unless a "
        "database backend such as this sqlite one is used instead.",
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default="fraudlens_baseline_models",
    )
    args = parser.parse_args()

    merged = load_and_merge(args.transaction_csv, args.identity_csv)
    train_df, test_df = prepare_train_test_features(merged, test_size=args.test_size)
    feature_columns = select_feature_columns(train_df)

    print(
        f"Train: {len(train_df)} rows ({train_df['isFraud'].sum()} fraud). "
        f"Test: {len(test_df)} rows ({test_df['isFraud'].sum()} fraud). "
        f"{len(feature_columns)} feature columns."
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    results = []
    for model_name in ["logistic_regression", "decision_tree", "random_forest"]:
        with mlflow.start_run(run_name=model_name):
            result = train_and_evaluate(model_name, train_df, test_df, feature_columns)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_features", result["n_features"])
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_metric("train_auc_roc", result["train_auc_roc"])
            mlflow.log_metric("test_auc_roc", result["test_auc_roc"])
            mlflow.log_metric("n_train", result["n_train"])
            mlflow.log_metric("n_test", result["n_test"])
            mlflow.sklearn.log_model(
                result["pipeline"], name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            results.append(result)
            print(
                f"{model_name}: train AUC-ROC={result['train_auc_roc']:.4f}, "
                f"test AUC-ROC={result['test_auc_roc']:.4f}"
            )

    best = max(results, key=lambda r: r["test_auc_roc"])
    print(
        f"\nBest Week 3 baseline by test AUC-ROC: {best['model_name']} "
        f"({best['test_auc_roc']:.4f}). This is the floor Week 4's "
        "XGBoost/LightGBM plus tuning needs to beat."
    )


if __name__ == "__main__":
    main()
