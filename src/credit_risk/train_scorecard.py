"""
train_scorecard.py, Credit Risk Scorecard CLI: builds a synthetic Home
Credit-style application population, fits WoE bins and selects features by
Information Value, fits a logistic regression scorecard and converts it to
points, evaluates it with this project's existing credit-risk metrics
(reused unchanged from src.models.metrics, the same module Week 4's
XGBoost/LightGBM models are scored with), then demonstrates PSI monitoring
against a second, deliberately drifted "monitoring" population.

Random, stratified train/test split, not chronological: Home Credit's
application_train.csv (real or synthetic) has no time column at all (unlike
the transaction fraud modules' TransactionDT), so there is no ordering to
respect the way chronological_train_test_split enforces for the fraud
detection modules. A stratified random split (on TARGET, since the ~8%
default rate is imbalanced enough that an unstratified split risks an
unlucky test fold) is the correct, standard choice here instead.

The PSI monitoring demo builds a second population with
generate_synthetic_credit.py's unemployment_shift and ext_source_shift
knobs (see that module's docstring), modeling a plausible economic
downturn: more applicants out of work, and systematically lower
credit-bureau scores across the whole applicant pool. This project's own
scorecard (fit ONLY on the original population, never refit) is applied to
this drifted population unchanged, and PSI compares the two resulting score
distributions, plus a per-feature PSI breakdown (psi_by_feature) to show
WHICH inputs actually drove the shift, not just that the overall score
distribution moved. This mirrors Week 8's disclosed synthetic domain-shift
design: the drift is a stated, chosen scenario for exercising the
monitoring code meaningfully, not a claim about any real economic period.
"""

from __future__ import annotations

import argparse

import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.generate_synthetic_credit import DAYS_EMPLOYED_ANOMALY, generate_synthetic_credit_applications
from src.credit_risk.psi_monitoring import population_stability_index, psi_by_feature, psi_verdict
from src.credit_risk.scorecard_model import build_points_scorecard, fit_scorecard_logistic_regression, score_applications
from src.credit_risk.woe_encoding import fit_woe_bins, iv_report, select_features_by_iv, transform_woe
from src.models.metrics import auc_pr, gini_coefficient, ks_statistic

SPECIAL_VALUES = {"DAYS_EMPLOYED": [DAYS_EMPLOYED_ANOMALY]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a WoE-encoded logistic regression credit "
        "scorecard on synthetic Home-Credit-schema data, select features "
        "by Information Value, evaluate it, and demonstrate PSI monitoring "
        "against a deliberately drifted population."
    )
    parser.add_argument("--n-applications", type=int, default=8000)
    parser.add_argument("--default-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--min-iv", type=float, default=0.02)
    parser.add_argument("--pdo", type=float, default=20.0)
    parser.add_argument("--base-score", type=float, default=600.0)
    parser.add_argument("--base-odds", type=float, default=50.0)
    parser.add_argument(
        "--monitoring-n-applications", type=int, default=3000,
        help="Size of the deliberately drifted population used for the PSI demo.",
    )
    parser.add_argument("--monitoring-seed", type=int, default=2)
    parser.add_argument(
        "--unemployment-shift", type=float, default=0.15,
        help="See generate_synthetic_credit.py: added to the Pensioner+Unemployed "
        "income-type probability for the monitoring population.",
    )
    parser.add_argument(
        "--ext-source-shift", type=float, default=-0.12,
        help="See generate_synthetic_credit.py: added to the mean of every "
        "EXT_SOURCE column for the monitoring population.",
    )
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fraudlens_credit_scorecard")
    args = parser.parse_args()

    df = generate_synthetic_credit_applications(
        n_applications=args.n_applications, default_rate=args.default_rate, seed=args.seed,
    )
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["TARGET"], random_state=0,
    )
    feature_columns = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]

    print(
        f"Train: {len(train_df)} applications ({train_df['TARGET'].sum()} defaults). "
        f"Test: {len(test_df)} applications ({test_df['TARGET'].sum()} defaults)."
    )

    woe_bins = fit_woe_bins(
        train_df, feature_columns, n_bins=args.n_bins, special_values=SPECIAL_VALUES,
    )
    report = iv_report(woe_bins)
    print("\nInformation Value by feature (fit on train only):")
    print(report.to_string(index=False))

    suspicious = report[report["strength"] == "suspicious"]
    if not suspicious.empty:
        print(
            f"\nWARNING: {len(suspicious)} feature(s) have IV above 0.5, "
            "conventionally a target-leakage red flag, not a cause for "
            f"celebration: {suspicious['feature'].tolist()}."
        )

    selected_features = select_features_by_iv(woe_bins, min_iv=args.min_iv)
    print(f"\n{len(selected_features)}/{len(feature_columns)} features selected at IV >= {args.min_iv}:")
    print(selected_features)

    woe_train = transform_woe(train_df, woe_bins)
    woe_test = transform_woe(test_df, woe_bins)
    model = fit_scorecard_logistic_regression(woe_train, selected_features)

    base_points, points_tables = build_points_scorecard(
        model, selected_features, woe_bins,
        pdo=args.pdo, base_score=args.base_score, base_odds=args.base_odds,
    )

    test_proba = model.predict_proba(
        woe_test[[f"{f}_woe" for f in selected_features]]
    )[:, 1]
    test_scores = score_applications(
        test_df, selected_features, woe_bins, base_points, points_tables,
    )

    ks = ks_statistic(test_df["TARGET"], test_proba)
    print(
        f"\nTest set: AUC-ROC={roc_auc_score(test_df['TARGET'], test_proba):.4f}, "
        f"AUC-PR={auc_pr(test_df['TARGET'], test_proba):.4f}, "
        f"KS={ks['ks_statistic']:.4f}, Gini={gini_coefficient(test_df['TARGET'], test_proba):.4f}."
    )
    print(
        f"Scorecard points: base={base_points:.1f}, "
        f"test score range=[{test_scores.min():.1f}, {test_scores.max():.1f}], "
        f"mean={test_scores.mean():.1f}."
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="scorecard"):
        mlflow.log_param("n_applications", args.n_applications)
        mlflow.log_param("n_bins", args.n_bins)
        mlflow.log_param("min_iv", args.min_iv)
        mlflow.log_param("n_features_selected", len(selected_features))
        mlflow.log_metric("test_auc_roc", roc_auc_score(test_df["TARGET"], test_proba))
        mlflow.log_metric("test_auc_pr", auc_pr(test_df["TARGET"], test_proba))
        mlflow.log_metric("test_ks_statistic", ks["ks_statistic"])
        mlflow.log_metric("test_gini", gini_coefficient(test_df["TARGET"], test_proba))

    # --- PSI monitoring demo against a deliberately drifted population ---
    monitoring_df = generate_synthetic_credit_applications(
        n_applications=args.monitoring_n_applications,
        default_rate=args.default_rate,
        seed=args.monitoring_seed,
        start_id=900001,
        unemployment_shift=args.unemployment_shift,
        ext_source_shift=args.ext_source_shift,
    )
    monitoring_scores = score_applications(
        monitoring_df, selected_features, woe_bins, base_points, points_tables,
    )

    score_psi = population_stability_index(test_scores, monitoring_scores, bins=args.psi_bins)
    print(
        f"\nPSI (test scores vs monitoring population's scores): {score_psi:.4f} "
        f"({psi_verdict(score_psi)})."
    )

    feature_psi = psi_by_feature(
        test_df, monitoring_df,
        columns=[c for c in feature_columns if test_df[c].dtype.kind in "fi"],
        bins=args.psi_bins,
    )
    print("\nPer-feature PSI (test vs monitoring population), top 5:")
    print(feature_psi.head(5).to_string(index=False))

    with mlflow.start_run(run_name="psi_monitoring"):
        mlflow.log_param("monitoring_n_applications", args.monitoring_n_applications)
        mlflow.log_param("unemployment_shift", args.unemployment_shift)
        mlflow.log_param("ext_source_shift", args.ext_source_shift)
        mlflow.log_metric("score_psi", score_psi)


if __name__ == "__main__":
    main()
