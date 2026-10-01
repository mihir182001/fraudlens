"""
advanced_models.py, Week 4: XGBoost and LightGBM, with SMOTE for the class
imbalance Week 3's floor left untouched, and hyperparameter tuning, logged
to MLflow with the full Week 4 metric set from src/models/metrics.py.

Week 3 established an honest, unimproved floor (plain scikit-learn
defaults, no class-imbalance handling, no tuning). This module is built to
make the improvement over that floor a real, measured claim rather than an
assumed one: for each of XGBoost and LightGBM, it trains both an
equally-unimproved "floor" version (same philosophy as Week 3, default
hyperparameters, no SMOTE) and a "tuned_smote" version (SMOTE-resampled
training data plus hyperparameters chosen by cross-validated search), so
the reported lift is a real before/after on the same train and test split,
not a comparison against a different project's numbers.

SMOTE (Synthetic Minority Oversampling Technique) is applied only to the
training data, and only inside a resampling-aware pipeline
(imblearn.pipeline.Pipeline, not scikit-learn's own Pipeline): imblearn's
Pipeline knows to run the resampling step during fit but skip it during
predict, so the test set is never resampled, and SMOTE is never allowed to
manufacture synthetic test rows a real deployment would never see. SMOTE
also cannot handle NaN, so the imputer runs before it in every pipeline
here, same imputer, same train-only fit discipline as Week 3.

Hyperparameter tuning uses RandomizedSearchCV with a TimeSeriesSplit
cross-validator rather than the default random K-fold. This project's
train_df is already sorted chronologically by chronological_train_test_split
(src/models/prepare_model_data.py), so TimeSeriesSplit's fold boundaries
follow that same real time ordering, keeping tuning's own internal
validation folds honest to how this model would actually be deployed
(never validated against data that is, in real time, still in its future).

One limitation worth stating plainly: optimal_threshold_by_f1 (Week 4's
metrics module) is applied here to the same test set the final AUC-ROC,
AUC-PR, KS, and Gini numbers are reported on, not a separate validation
split. That is a simplification made to avoid carving a third split out of
an already-small synthetic sample; the threshold this project reports
should be read as "the best threshold in hindsight on this test set," not
as an unbiased estimate of the best threshold for a transaction this model
has never seen. A production deployment would pick the threshold on its
own held-out validation set, distinct from whatever set reports the final
numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.data.load_and_merge import load_and_merge
from src.models.metrics import auc_pr, gini_coefficient, ks_statistic, optimal_threshold_by_f1
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns

RANDOM_STATE = 42

# See baseline_models.py's own SKOPS_TRUSTED_TYPES comment for the general
# background (mlflow 3.16.1 serializes sklearn-compatible models with
# skops, which refuses to load a file back unless every type inside it is
# explicitly trusted). Confirmed directly against this module's own
# pipelines: an imblearn Pipeline containing SMOTE additionally stores a
# KD-tree and distance-metric object it uses internally to find each
# minority-class point's nearest neighbors, and XGBoost/LightGBM each store
# their own fitted booster type. This list is the union of what all four
# of this module's pipeline variants (xgboost/lightgbm, with/without
# SMOTE) were confirmed to need; a pipeline missing SMOTE simply does not
# use the SMOTE-related entries, which is harmless since skops only checks
# the types actually present in a given file.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "imblearn.over_sampling._smote.base.SMOTE",
    "imblearn.pipeline.Pipeline",
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    "sklearn.neighbors._kd_tree.KDTree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]

XGBOOST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
}

LIGHTGBM_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__num_leaves": [15, 31, 63],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
}


def build_advanced_pipeline(model_name: str, use_smote: bool) -> ImbPipeline:
    """Returns an imblearn Pipeline (imputer, optional SMOTE, model) for
    "xgboost" or "lightgbm", built with untuned, library-default
    hyperparameters (aside from a couple of plain stability/reproducibility
    settings noted inline).

    LightGBM's subsample argument is silently ignored unless
    subsample_freq is also set to a positive number, a real scikit-learn
    integration quirk confirmed directly (a subsample=0.5 run produced
    identical trees to subsample=1.0 until subsample_freq=1 was added), so
    subsample_freq=1 is fixed here rather than left at its 0 default.
    """
    if model_name == "xgboost":
        model = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        )
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1, subsample_freq=1,
        )
    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected 'xgboost' or 'lightgbm'."
        )

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def _param_distributions_for(model_name: str) -> Dict[str, list]:
    if model_name == "xgboost":
        return XGBOOST_PARAM_DISTRIBUTIONS
    if model_name == "lightgbm":
        return LIGHTGBM_PARAM_DISTRIBUTIONS
    raise ValueError(
        f"Unknown model_name {model_name!r}; expected 'xgboost' or 'lightgbm'."
    )


def tune_hyperparameters(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    n_iter: int = 8,
    n_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Runs RandomizedSearchCV over model_name's pipeline, scored by
    AUC-ROC, using a TimeSeriesSplit cross-validator so every validation
    fold is strictly later in time than the fold it is validated against.

    x_train and y_train must already be in chronological order (true of
    the train_df this project's chronological_train_test_split produces);
    TimeSeriesSplit trusts row order to reflect real time and does not
    re-sort anything itself.
    """
    pipeline = build_advanced_pipeline(model_name, use_smote=use_smote)
    param_distributions = _param_distributions_for(model_name)
    cv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def train_and_evaluate_advanced(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    use_smote: bool = True,
    tune: bool = True,
    n_iter: int = 8,
    n_splits: int = 3,
    target_col: str = "isFraud",
) -> Dict:
    """Trains one Week 4 model configuration and evaluates it on test_df
    with the full metric set: AUC-ROC (for continuity with Week 3),
    AUC-PR, the KS statistic and its threshold, the Gini coefficient, and
    an F1-optimal decision threshold (see the module docstring's
    limitation note on that last one).
    """
    x_train = train_df[feature_columns]
    y_train = train_df[target_col]
    x_test = test_df[feature_columns]
    y_test = test_df[target_col]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training set has only one class present; these metrics are "
            "undefined. This can happen with a very small or unluckily "
            "split synthetic sample; try a larger --n-transactions or a "
            "different seed."
        )
    if y_test.nunique() < 2:
        raise ValueError(
            "Test set has only one class present; these metrics are "
            "undefined. This can happen with a very small or unluckily "
            "split synthetic sample; try a larger --n-transactions or a "
            "different seed."
        )

    best_params: Optional[Dict] = None
    if tune:
        search = tune_hyperparameters(
            model_name, x_train, y_train, use_smote=use_smote,
            n_iter=n_iter, n_splits=n_splits,
        )
        pipeline = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline = build_advanced_pipeline(model_name, use_smote=use_smote)
        pipeline.fit(x_train, y_train)

    train_proba = pipeline.predict_proba(x_train)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]

    ks = ks_statistic(y_test, test_proba)
    threshold = optimal_threshold_by_f1(y_test, test_proba)

    return {
        "model_name": model_name,
        "use_smote": use_smote,
        "tuned": tune,
        "pipeline": pipeline,
        "best_params": best_params,
        "train_auc_roc": roc_auc_score(y_train, train_proba),
        "test_auc_roc": roc_auc_score(y_test, test_proba),
        "test_auc_pr": auc_pr(y_test, test_proba),
        "test_ks_statistic": ks["ks_statistic"],
        "test_ks_threshold": ks["ks_threshold"],
        "test_gini": gini_coefficient(y_test, test_proba),
        "optimal_threshold": threshold["threshold"],
        "optimal_threshold_precision": threshold["precision"],
        "optimal_threshold_recall": threshold["recall"],
        "optimal_threshold_f1": threshold["f1"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_fraud": int(y_train.sum()),
        "n_test_fraud": int(y_test.sum()),
        "n_features": len(feature_columns),
    }


def run_all_advanced_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    tune: bool = True,
    n_iter: int = 8,
    n_splits: int = 3,
) -> List[Dict]:
    """Trains both an unimproved floor (no SMOTE, no tuning) and an
    improved (SMOTE plus tuning) version of both XGBoost and LightGBM, in
    that order, so every model's own before/after lift is directly
    comparable on the same train and test split.
    """
    results = []
    for model_name in ["xgboost", "lightgbm"]:
        results.append(
            train_and_evaluate_advanced(
                model_name, train_df, test_df, feature_columns,
                use_smote=False, tune=False,
            )
        )
        results.append(
            train_and_evaluate_advanced(
                model_name, train_df, test_df, feature_columns,
                use_smote=True, tune=tune, n_iter=n_iter, n_splits=n_splits,
            )
        )
    return results


def _run_label(result: Dict) -> str:
    variant = "tuned_smote" if (result["use_smote"] or result["tuned"]) else "floor"
    return f"{result['model_name']}_{variant}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Week 4's XGBoost and LightGBM "
        "models, each as an unimproved floor and as a SMOTE-resampled, "
        "hyperparameter-tuned version, against a chronological train/test "
        "split, logging every run to MLflow with the full Week 4 metric set."
    )
    parser.add_argument(
        "--transaction-csv", type=Path, default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=8, help="RandomizedSearchCV iterations.")
    parser.add_argument("--n-splits", type=int, default=3, help="TimeSeriesSplit folds.")
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
        help="Where MLflow stores run data. See baseline_models.py for why "
        "this is a sqlite database rather than a file:./mlruns path.",
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default="fraudlens_advanced_models",
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

    results = run_all_advanced_models(
        train_df, test_df, feature_columns,
        tune=True, n_iter=args.n_iter, n_splits=args.n_splits,
    )

    for result in results:
        run_name = _run_label(result)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", result["model_name"])
            mlflow.log_param("use_smote", result["use_smote"])
            mlflow.log_param("tuned", result["tuned"])
            mlflow.log_param("n_features", result["n_features"])
            mlflow.log_param("test_size", args.test_size)
            if result["best_params"]:
                for key, value in result["best_params"].items():
                    mlflow.log_param(key, value)
            mlflow.log_metric("train_auc_roc", result["train_auc_roc"])
            mlflow.log_metric("test_auc_roc", result["test_auc_roc"])
            mlflow.log_metric("test_auc_pr", result["test_auc_pr"])
            mlflow.log_metric("test_ks_statistic", result["test_ks_statistic"])
            mlflow.log_metric("test_ks_threshold", result["test_ks_threshold"])
            mlflow.log_metric("test_gini", result["test_gini"])
            mlflow.log_metric("optimal_threshold", result["optimal_threshold"])
            mlflow.log_metric("optimal_threshold_f1", result["optimal_threshold_f1"])
            mlflow.log_metric("n_train", result["n_train"])
            mlflow.log_metric("n_test", result["n_test"])
            mlflow.sklearn.log_model(
                result["pipeline"], name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            print(
                f"{run_name}: test AUC-ROC={result['test_auc_roc']:.4f}, "
                f"test AUC-PR={result['test_auc_pr']:.4f}, "
                f"KS={result['test_ks_statistic']:.4f}, "
                f"Gini={result['test_gini']:.4f}"
            )

    best = max(results, key=lambda r: r["test_auc_roc"])
    print(
        f"\nBest Week 4 model by test AUC-ROC: {_run_label(best)} "
        f"({best['test_auc_roc']:.4f})."
    )


if __name__ == "__main__":
    main()
