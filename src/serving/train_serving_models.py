"""
train_serving_models.py, FastAPI/Streamlit serving module: trains one
fraud model and one credit scorecard, exactly like the earlier weeks'
own training scripts, but saves the fitted result to disk (models/*.joblib)
so src/api/main.py can load a model once at startup instead of retraining
on every request.

This intentionally reuses this project's own existing training code rather
than reimplementing anything: the fraud model is trained with
src.models.advanced_models' own XGBoost pipeline builder and evaluated with
src.models.metrics, on data built the same way src.models.advanced_models'
own main() builds it (src.data.generate_synthetic_ieee ->
merge_transaction_identity -> src.models.prepare_model_data). The credit
scorecard is trained with this project's Week 9 WoE/scorecard modules, the
same way src.credit_risk.train_scorecard's own main() does. Nothing about
either model changes for serving; only hyperparameter TUNING is skipped
here (use_smote=True, tune=False), a deliberate choice to keep this script
fast and deterministic for repeated local runs; Week 4's own tuned numbers
remain the project's research benchmark, and this serving artifact reports
its own real (untuned) test metrics rather than borrowing Week 4's.

Each saved artifact is a plain dict (joblib.dump), not a bespoke class, so
loading it back in src/api/main.py needs no import of this module:
  fraud_model.joblib:
    pipeline, feature_columns, thresholds (FraudDecisionThresholds),
    trained_at, test_auc_roc, test_auc_pr, test_ks_statistic, test_gini,
    n_train, n_test.
  credit_scorecard.joblib:
    model, woe_bins, base_points, points_tables, feature_columns, cutoffs
    (CreditCutoffs), trained_at, test_auc_roc, test_auc_pr,
    test_ks_statistic, test_gini, n_train, n_test, reference_scores (the
    scorecard's own test-set scores, used by
    src/credit_risk/monitor_psi.py as the "expected" PSI distribution).

See decision_policy.py's own module docstring for exactly how
thresholds/cutoffs are chosen from each model's own train-set score
distribution (fraud) or train-set F1-optimal point (credit's percentile
cutoffs are computed on train scores here for the same fit-on-train-only
reason).

Unless --skip-registry is passed, main() also registers each trained model
as a new version in MLflow's own Model Registry and promotes it with a
"champion" alias (see model_registry.py's own module docstring for why),
then adds run_id, registered_model_name, registered_model_version, and
alias to that same artifact dict before it is saved, so the joblib file on
disk always carries a durable pointer back to the exact MLflow run and
registry version that produced it.

Both models default to synthetic data (as above), but can instead train on
real Kaggle data once downloaded: --transaction-csv and --identity-csv
together (both required, or neither) point the fraud model at the real
IEEE-CIS train_transaction.csv/train_identity.csv, loaded unchanged by
src.data.load_and_merge (schema-agnostic, needs no code change for the
real file's full column set; see that module's own docstring); --credit-csv
points the credit scorecard at the real Home Credit application_train.csv,
loaded by src.data.load_credit_applications, which subsets the real file's
122 columns down to the same 18 this project's serving API
(CreditScoringRequest) already accepts, so switching to real data changes
nothing about the API contract, only the values training sees.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.credit_risk.scorecard_model import (
    build_points_scorecard,
    fit_scorecard_logistic_regression,
    score_applications,
)
from src.credit_risk.woe_encoding import fit_woe_bins, select_features_by_iv, transform_woe
from src.data.generate_synthetic_credit import DAYS_EMPLOYED_ANOMALY, generate_synthetic_credit_applications
from src.data.generate_synthetic_ieee import generate_synthetic_identity, generate_synthetic_transactions
from src.data.load_and_merge import load_and_merge, merge_transaction_identity
from src.data.load_credit_applications import load_credit_applications
from src.models.advanced_models import SKOPS_TRUSTED_TYPES, train_and_evaluate_advanced
from src.models.metrics import auc_pr, gini_coefficient, ks_statistic
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns
from src.serving.decision_policy import CreditCutoffs, FraudDecisionThresholds
from src.serving.model_registry import register_and_promote

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
SERVING_EXPERIMENT_NAME = "fraudlens_serving_models"
REGISTERED_FRAUD_MODEL_NAME = "fraudlens_fraud_model"
REGISTERED_CREDIT_MODEL_NAME = "fraudlens_credit_scorecard"
SPECIAL_VALUES = {"DAYS_EMPLOYED": [DAYS_EMPLOYED_ANOMALY]}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_fraud_serving_model(
    n_transactions: int = 5000,
    fraud_rate: float = 0.035,
    identity_coverage: float = 0.24,
    seed: int = 42,
    test_size: float = 0.2,
    transaction_csv: Optional[Path] = None,
    identity_csv: Optional[Path] = None,
) -> Dict:
    """Trains the SMOTE-resampled, untuned XGBoost pipeline
    (src.models.advanced_models.train_and_evaluate_advanced) and returns a
    serving artifact dict (see module docstring).

    By default this trains on synthetic IEEE-CIS-schema data
    (n_transactions, fraud_rate, identity_coverage, seed all apply only to
    that synthetic generation). Passing BOTH transaction_csv and
    identity_csv instead loads and merges the real files with
    src.data.load_and_merge.load_and_merge, which needs no code changes of
    its own to handle the real schema (see that module's own docstring for
    why it is schema-agnostic); n_transactions, fraud_rate, and
    identity_coverage are then ignored, since the real files determine
    their own row count, fraud rate, and identity coverage. Passing only
    one of the two raises, since a merge needs both files.
    """
    if transaction_csv is not None or identity_csv is not None:
        if transaction_csv is None or identity_csv is None:
            raise ValueError(
                "transaction_csv and identity_csv must both be given "
                "together (or neither, to train on synthetic data instead)."
            )
        merged = load_and_merge(transaction_csv, identity_csv)
    else:
        transactions = generate_synthetic_transactions(
            n_transactions=n_transactions, fraud_rate=fraud_rate, seed=seed,
        )
        identity = generate_synthetic_identity(
            transactions["TransactionID"].to_numpy(), coverage_rate=identity_coverage, seed=seed + 1,
        )
        merged = merge_transaction_identity(transactions, identity)
    train_df, test_df = prepare_train_test_features(merged, test_size=test_size)
    feature_columns = select_feature_columns(train_df)

    result = train_and_evaluate_advanced(
        "xgboost", train_df, test_df, feature_columns, use_smote=True, tune=False,
    )

    review_threshold = float(result["optimal_threshold"])
    decline_threshold = review_threshold + (1.0 - review_threshold) * 0.5
    thresholds = FraudDecisionThresholds(
        review_threshold=review_threshold, decline_threshold=decline_threshold,
    )

    return {
        "pipeline": result["pipeline"],
        "feature_columns": feature_columns,
        "thresholds": thresholds,
        "trained_at": _now_iso(),
        "test_auc_roc": result["test_auc_roc"],
        "test_auc_pr": result["test_auc_pr"],
        "test_ks_statistic": result["test_ks_statistic"],
        "test_gini": result["test_gini"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
    }


def train_credit_serving_scorecard(
    n_applications: int = 8000,
    default_rate: float = 0.08,
    seed: int = 1,
    test_size: float = 0.2,
    n_bins: int = 5,
    min_iv: float = 0.02,
    pdo: float = 20.0,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    credit_csv: Optional[Path] = None,
) -> Dict:
    """Trains the WoE/IV/logistic-regression scorecard
    (src.credit_risk.train_scorecard's own approach) and returns a serving
    artifact dict (see module docstring). approve_cutoff/decline_cutoff are
    the 60th/20th percentile of the TRAIN split's own scores, fit-on-train-only
    like every other cutoff and bin edge in this project.

    By default this trains on synthetic Home Credit-schema data
    (n_applications and default_rate apply only to that synthetic
    generation). Passing credit_csv instead loads the real
    application_train.csv with src.data.load_credit_applications, which
    subsets the real file's 122 columns down to the same 18 this project's
    pipeline already uses (see that module's own docstring for why); seed
    still applies, to the train/test split.
    """
    if credit_csv is not None:
        df = load_credit_applications(credit_csv)
    else:
        df = generate_synthetic_credit_applications(
            n_applications=n_applications, default_rate=default_rate, seed=seed,
        )
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["TARGET"], random_state=0,
    )
    feature_columns = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]

    woe_bins = fit_woe_bins(train_df, feature_columns, n_bins=n_bins, special_values=SPECIAL_VALUES)
    selected_features = select_features_by_iv(woe_bins, min_iv=min_iv)

    woe_train = transform_woe(train_df, woe_bins)
    woe_test = transform_woe(test_df, woe_bins)
    model = fit_scorecard_logistic_regression(woe_train, selected_features)

    base_points, points_tables = build_points_scorecard(
        model, selected_features, woe_bins, pdo=pdo, base_score=base_score, base_odds=base_odds,
    )

    train_scores = score_applications(train_df, selected_features, woe_bins, base_points, points_tables)
    test_scores = score_applications(test_df, selected_features, woe_bins, base_points, points_tables)
    test_proba = model.predict_proba(woe_test[[f"{f}_woe" for f in selected_features]])[:, 1]

    approve_cutoff = float(np.percentile(train_scores, 60))
    decline_cutoff = float(np.percentile(train_scores, 20))
    cutoffs = CreditCutoffs(approve_cutoff=approve_cutoff, decline_cutoff=decline_cutoff)

    ks = ks_statistic(test_df["TARGET"], test_proba)
    return {
        "model": model,
        "woe_bins": woe_bins,
        "base_points": base_points,
        "points_tables": points_tables,
        "feature_columns": selected_features,
        "cutoffs": cutoffs,
        "trained_at": _now_iso(),
        "test_auc_roc": roc_auc_score(test_df["TARGET"], test_proba),
        "test_auc_pr": auc_pr(test_df["TARGET"], test_proba),
        "test_ks_statistic": ks["ks_statistic"],
        "test_gini": gini_coefficient(test_df["TARGET"], test_proba),
        "n_train": len(train_df),
        "n_test": len(test_df),
        # The scorecard's own TEST-set scores, saved so
        # src/credit_risk/monitor_psi.py can check a later population for
        # drift against the exact population this scorecard was validated
        # on, without needing to retrain (or even re-run this function) to
        # get a reference distribution.
        "reference_scores": test_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the fraud model and credit scorecard used by "
        "the FastAPI serving layer, and save both to models/*.joblib."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--n-transactions", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument("--n-applications", type=int, default=8000)
    parser.add_argument("--default-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--transaction-csv", type=Path, default=None,
        help="Real IEEE-CIS train_transaction.csv. Must be given together "
        "with --identity-csv; when both are given, the fraud model trains "
        "on this real data instead of synthetic data, and "
        "--n-transactions/--fraud-rate are ignored.",
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=None,
        help="Real IEEE-CIS train_identity.csv. See --transaction-csv.",
    )
    parser.add_argument(
        "--credit-csv", type=Path, default=None,
        help="Real Home Credit application_train.csv. When given, the "
        "credit scorecard trains on this real data (subset to this "
        "project's existing 18-column schema; see "
        "src/data/load_credit_applications.py) instead of synthetic data, "
        "and --n-applications/--default-rate are ignored.",
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument(
        "--skip-registry", action="store_true",
        help="Skip MLflow Model Registry registration (models/*.joblib is still written as "
        "usual). Useful for fast local iteration without touching mlflow.db.",
    )
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    using_real_fraud_data = args.transaction_csv is not None or args.identity_csv is not None
    print(
        "Training fraud serving model (XGBoost, SMOTE, untuned) on "
        + ("real IEEE-CIS data..." if using_real_fraud_data else "synthetic data...")
    )
    fraud_artifact = train_fraud_serving_model(
        n_transactions=args.n_transactions, fraud_rate=args.fraud_rate, seed=args.seed,
        transaction_csv=args.transaction_csv, identity_csv=args.identity_csv,
    )
    if not args.skip_registry:
        registry_info = register_and_promote(
            fraud_artifact["pipeline"],
            registered_model_name=REGISTERED_FRAUD_MODEL_NAME,
            params={
                "n_transactions": args.n_transactions,
                "fraud_rate": args.fraud_rate,
                "seed": args.seed,
                "n_features": len(fraud_artifact["feature_columns"]),
            },
            metrics={
                "test_auc_roc": fraud_artifact["test_auc_roc"],
                "test_auc_pr": fraud_artifact["test_auc_pr"],
                "test_ks_statistic": fraud_artifact["test_ks_statistic"],
                "test_gini": fraud_artifact["test_gini"],
            },
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=SERVING_EXPERIMENT_NAME,
            run_name="fraud_model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        fraud_artifact.update(registry_info)
        print(
            f"Registered {REGISTERED_FRAUD_MODEL_NAME} version "
            f"{registry_info['registered_model_version']} and promoted it to "
            f"'{registry_info['alias']}'."
        )
    fraud_path = args.models_dir / "fraud_model.joblib"
    joblib.dump(fraud_artifact, fraud_path)
    print(
        f"Wrote {fraud_path}: test AUC-ROC={fraud_artifact['test_auc_roc']:.4f}, "
        f"review_threshold={fraud_artifact['thresholds'].review_threshold:.4f}, "
        f"decline_threshold={fraud_artifact['thresholds'].decline_threshold:.4f}."
    )

    print(
        "\nTraining credit serving scorecard (WoE + logistic regression) on "
        + ("real Home Credit data..." if args.credit_csv is not None else "synthetic data...")
    )
    credit_artifact = train_credit_serving_scorecard(
        n_applications=args.n_applications, default_rate=args.default_rate, seed=args.seed,
        credit_csv=args.credit_csv,
    )
    if not args.skip_registry:
        # The credit scorecard's model (a plain scikit-learn
        # LogisticRegression, with no ensemble internals like the fraud
        # pipeline's SMOTE/XGBoost steps) was confirmed directly to need no
        # skops_trusted_types at all, unlike the fraud model above.
        registry_info = register_and_promote(
            credit_artifact["model"],
            registered_model_name=REGISTERED_CREDIT_MODEL_NAME,
            params={
                "n_applications": args.n_applications,
                "default_rate": args.default_rate,
                "seed": args.seed,
                "n_features": len(credit_artifact["feature_columns"]),
            },
            metrics={
                "test_auc_roc": credit_artifact["test_auc_roc"],
                "test_auc_pr": credit_artifact["test_auc_pr"],
                "test_ks_statistic": credit_artifact["test_ks_statistic"],
                "test_gini": credit_artifact["test_gini"],
            },
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=SERVING_EXPERIMENT_NAME,
            run_name="credit_scorecard",
        )
        credit_artifact.update(registry_info)
        print(
            f"Registered {REGISTERED_CREDIT_MODEL_NAME} version "
            f"{registry_info['registered_model_version']} and promoted it to "
            f"'{registry_info['alias']}'."
        )
    credit_path = args.models_dir / "credit_scorecard.joblib"
    joblib.dump(credit_artifact, credit_path)
    print(
        f"Wrote {credit_path}: test AUC-ROC={credit_artifact['test_auc_roc']:.4f}, "
        f"approve_cutoff={credit_artifact['cutoffs'].approve_cutoff:.1f}, "
        f"decline_cutoff={credit_artifact['cutoffs'].decline_cutoff:.1f}."
    )


if __name__ == "__main__":
    main()
