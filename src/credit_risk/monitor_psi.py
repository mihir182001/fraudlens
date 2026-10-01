"""
monitor_psi.py, Credit Risk Scorecard module: a standalone PSI monitoring
check, decoupled from training so it can run on its own schedule (a cron
job, a scheduled CI workflow, or similar) against whatever the credit
scorecard currently deployed for serving actually is.

train_scorecard.py already demonstrates PSI monitoring (see that module's
own docstring), but only as one more step inside its own training run,
comparing its own freshly-trained scorecard against a synthetic drifted
population it also builds in that same run. That is a useful demonstration
of the mechanism, but it is not automation: nothing there runs on its own,
independently of a person rerunning that whole training script.

This script instead:
  1. Loads models/credit_scorecard.joblib, the exact artifact
     src/api/main.py serves requests from (see train_serving_models.py),
     rather than retraining anything. reference_scores inside that
     artifact (the scorecard's own test-set scores, saved at training time)
     is the "expected" distribution PSI compares against, so this checks
     for drift against the population the scorecard was actually validated
     on, not against a new, arbitrary reference recomputed on every run.
  2. Scores a new batch of applications against that same, unchanged
     scorecard. By default this is a freshly generated synthetic batch
     (disclosed as such below and in its own printed output); --input-csv
     points this at a real file instead, once one exists, with the same
     raw columns CreditScoringRequest lists in src/serving/schemas.py.
  3. Computes PSI and logs it as its own MLflow run, in the same
     "fraudlens_credit_scorecard" experiment src/dashboard/app.py's
     load_psi_history() already reads from, so a scheduled run of this
     script shows up in the existing dashboard with no dashboard change.
  4. Exits with status 1 if the resulting PSI verdict is
     "significant_shift" (see psi_monitoring.py), 0 for "stable" or
     "moderate_shift", and 2 if it could not run the check at all (no
     model file, or a model file trained before reference_scores existed).
     A cron job or scheduled CI workflow can alert on a non-zero exit
     rather than needing to parse this script's own printed output.

Deliberately out of scope here, unlike train_scorecard.py's own PSI demo:
a per-feature PSI breakdown (psi_by_feature) showing which raw inputs
drove a shift. That breakdown needs the ORIGINAL training population's raw
feature values to compare against, which credit_scorecard.joblib does not
currently store (only its resulting scores, in reference_scores); adding
that would mean saving a full reference dataframe into the artifact, a
real storage and scope tradeoff left for whoever wires this into an actual
schedule to decide is worth it. This script's own score-level PSI is the
actionable, alertable signal; a full root-cause breakdown remains
train_scorecard.py's own demo, run by hand when score PSI here signals a
shift worth investigating.

Run locally with:
    python -m src.credit_risk.monitor_psi
or against a real batch of applications once one exists:
    python -m src.credit_risk.monitor_psi --input-csv path/to/applications.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import mlflow
import pandas as pd

from src.credit_risk.psi_monitoring import (
    PSI_SIGNIFICANT_THRESHOLD,
    population_stability_index,
    psi_verdict,
)
from src.credit_risk.scorecard_model import score_applications
from src.data.generate_synthetic_credit import generate_synthetic_credit_applications

DEFAULT_MODEL_PATH = Path("models") / "credit_scorecard.joblib"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MONITORING_EXPERIMENT_NAME = "fraudlens_credit_scorecard"

EXIT_OK = 0
EXIT_SIGNIFICANT_SHIFT = 1
EXIT_COULD_NOT_RUN = 2


def _load_monitoring_batch(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv is not None:
        print(f"Reading monitoring batch from {args.input_csv}.")
        return pd.read_csv(args.input_csv)
    # No real batch available yet: this project still runs entirely on
    # synthetic data (see generate_synthetic_credit.py's own docstring for
    # why). A fresh batch is generated instead, with the same
    # unemployment_shift/ext_source_shift knobs train_scorecard.py's own
    # PSI demo uses, so a scheduled run made without --input-csv still
    # exercises this script meaningfully rather than trivially comparing a
    # population against a near-identical resample of itself.
    print(
        f"No --input-csv given: generating a synthetic monitoring batch of "
        f"{args.n_applications} applications instead (disclosed as such; "
        "not a claim about any real population)."
    )
    return generate_synthetic_credit_applications(
        n_applications=args.n_applications,
        default_rate=args.default_rate,
        seed=args.seed,
        start_id=900001,
        unemployment_shift=args.unemployment_shift,
        ext_source_shift=args.ext_source_shift,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check the currently deployed credit scorecard "
        "(models/credit_scorecard.joblib) for population drift (PSI) "
        "against a new batch of applications, without retraining anything."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="CSV of new applications to check, with the same raw columns "
        "CreditScoringRequest lists (src/serving/schemas.py). If omitted, "
        "a fresh synthetic batch is generated instead (disclosed as such).",
    )
    parser.add_argument("--n-applications", type=int, default=3000)
    parser.add_argument("--default-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--unemployment-shift", type=float, default=0.15,
        help="Only used for the generated synthetic batch; see generate_synthetic_credit.py.",
    )
    parser.add_argument(
        "--ext-source-shift", type=float, default=-0.12,
        help="Only used for the generated synthetic batch; see generate_synthetic_credit.py.",
    )
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-experiment", type=str, default=MONITORING_EXPERIMENT_NAME)
    parser.add_argument(
        "--skip-mlflow", action="store_true",
        help="Print the PSI result without logging it to MLflow.",
    )
    args = parser.parse_args(argv)

    if not args.model_path.exists():
        print(
            f"{args.model_path} not found. Run "
            "`python -m src.serving.train_serving_models` first to create it.",
            file=sys.stderr,
        )
        sys.exit(EXIT_COULD_NOT_RUN)

    artifact = joblib.load(args.model_path)
    if "reference_scores" not in artifact:
        print(
            f"{args.model_path} has no 'reference_scores' (it was trained "
            "before monitor_psi.py existed). Retrain with "
            "`python -m src.serving.train_serving_models` to add it.",
            file=sys.stderr,
        )
        sys.exit(EXIT_COULD_NOT_RUN)

    monitoring_df = _load_monitoring_batch(args)
    print(f"Scoring {len(monitoring_df)} applications against {args.model_path}...")

    monitoring_scores = score_applications(
        monitoring_df, artifact["feature_columns"], artifact["woe_bins"],
        artifact["base_points"], artifact["points_tables"],
    )

    score_psi = population_stability_index(
        artifact["reference_scores"], monitoring_scores, bins=args.psi_bins,
    )
    verdict = psi_verdict(score_psi)
    print(f"\nScore PSI: {score_psi:.4f} ({verdict}).")

    if not args.skip_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="psi_monitoring"):
            mlflow.log_param("n_monitoring_applications", len(monitoring_df))
            mlflow.log_param("input_csv", str(args.input_csv) if args.input_csv else "synthetic")
            mlflow.log_metric("score_psi", score_psi)
        print(f"Logged to MLflow experiment '{args.mlflow_experiment}'.")

    if verdict == "significant_shift":
        print(
            "\nSignificant population shift detected (PSI >= "
            f"{PSI_SIGNIFICANT_THRESHOLD}). This scorecard is a real "
            "candidate for retraining or at least a fresh validation.",
            file=sys.stderr,
        )
        sys.exit(EXIT_SIGNIFICANT_SHIFT)

    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
