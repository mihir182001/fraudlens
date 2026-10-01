"""
train_uplift.py, Week 6 CLI: generates (or loads) a randomized campaign
dataset, fits an SLearner and a TLearner, evaluates both with
qini_coefficient and uplift_at_k, and reports both scores against an oracle
upper bound computed from the dataset's ground-truth true_uplift column.

That oracle comparison is only possible because this project controls the
data-generating process end to end (src/data/generate_synthetic_campaign.py
records each row's true, unobservable individual treatment effect). No real
campaign has that column; a real deployment validates a model by running it
as the targeting rule in a fresh randomized holdout and reading the actual
Qini curve/uplift-at-k off the outcomes, not by comparing to some oracle.
Here, the oracle exists purely to sanity-check that this module's own
metric and model code is measuring what it claims to: if SLearner or
TLearner scored ABOVE the oracle, or the oracle itself did not clearly beat
random, that would mean a bug in qini_coefficient or in how the segments
were designed, not a genuinely superb model.

Unlike Weeks 1 through 5's transaction data, a randomized campaign snapshot
has no natural time ordering to respect, so this script uses a plain
stratified random split (stratified on treatment, to keep both splits at
close to the requested treatment_rate) rather than Week 3's
chronological_train_test_split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split

from src.data.generate_synthetic_campaign import FEATURE_COLUMNS, generate_synthetic_campaign
from src.uplift.uplift_metrics import qini_coefficient, uplift_at_k
from src.uplift.uplift_models import SLearner, TLearner


def _evaluate(name: str, uplift_score, test_df, top_k: float) -> dict:
    result = {
        "qini_coefficient": qini_coefficient(
            test_df["outcome"].to_numpy(), test_df["treatment"].to_numpy(), uplift_score,
        ),
        "uplift_at_k": uplift_at_k(
            test_df["outcome"].to_numpy(), test_df["treatment"].to_numpy(), uplift_score, k=top_k,
        ),
    }
    print(
        f"{name}: qini_coefficient={result['qini_coefficient']:.5f}, "
        f"uplift@top{top_k:.0%}={result['uplift_at_k']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Week 6's uplift models (SLearner, "
        "TLearner) on a synthetic randomized marketing campaign, reporting "
        "both against the dataset's own ground-truth oracle uplift."
    )
    parser.add_argument("--n-customers", type=int, default=20000)
    parser.add_argument("--treatment-rate", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-k", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fraudlens_uplift_models")
    args = parser.parse_args()

    df = generate_synthetic_campaign(
        n_customers=args.n_customers, treatment_rate=args.treatment_rate, seed=args.seed,
    )
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["treatment"],
    )
    print(
        f"Train: {len(train_df)} rows ({train_df['treatment'].mean():.3f} treatment rate). "
        f"Test: {len(test_df)} rows ({test_df['treatment'].mean():.3f} treatment rate)."
    )

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    slearner = SLearner().fit(X_train, train_df["treatment"].to_numpy(), train_df["outcome"].to_numpy())
    tlearner = TLearner().fit(X_train, train_df["treatment"].to_numpy(), train_df["outcome"].to_numpy())

    slearner_uplift = slearner.predict_uplift(X_test)
    tlearner_uplift = tlearner.predict_uplift(X_test)
    oracle_uplift = test_df["true_uplift"].to_numpy()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    print()
    results = {}
    with mlflow.start_run(run_name="uplift_models"):
        mlflow.log_param("n_customers", args.n_customers)
        mlflow.log_param("treatment_rate", args.treatment_rate)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("top_k", args.top_k)

        for name, uplift_score in [
            ("SLearner (logistic)", slearner_uplift),
            ("TLearner (logistic)", tlearner_uplift),
            ("Oracle (ground-truth true_uplift)", oracle_uplift),
        ]:
            result = _evaluate(name, uplift_score, test_df, args.top_k)
            results[name] = result
            metric_prefix = name.split(" ")[0].lower()
            mlflow.log_metric(f"{metric_prefix}_qini_coefficient", result["qini_coefficient"])
            mlflow.log_metric(f"{metric_prefix}_uplift_at_k", result["uplift_at_k"])

    print(
        "\nSanity check: both models should score positive (better than random) "
        "and no higher than the oracle's own score, since the oracle uses the "
        "true, otherwise-unobservable individual treatment effect this dataset "
        "was generated from."
    )
    oracle_qini = results["Oracle (ground-truth true_uplift)"]["qini_coefficient"]
    for name in ("SLearner (logistic)", "TLearner (logistic)"):
        model_qini = results[name]["qini_coefficient"]
        print(
            f"{name}: qini_coefficient={model_qini:.5f} "
            f"({'within' if 0 <= model_qini <= oracle_qini else 'OUTSIDE'} "
            f"[0, oracle={oracle_qini:.5f}])"
        )


if __name__ == "__main__":
    main()
