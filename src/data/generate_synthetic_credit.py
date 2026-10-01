"""
generate_synthetic_credit.py, Credit Risk Scorecard module: a synthetic
stand-in for the real Home Credit Default Risk dataset's main application
table (application_train.csv), used for the same reason
generate_synthetic_ieee.py exists: this sandbox cannot reach kaggle.com to
download the real files. Nothing here is a substitute for real EDA. Every
number this produces is synthetic and is never reported as a real finding
about credit risk. Once this pipeline runs against the real downloaded
data, report whatever it actually measures.

What this mimics from the real dataset (per the Home Credit Default Risk
Kaggle competition's own data description and the many public exploratory
kernels written about it, most notably Will Koehrsen's widely-cited "Start
Here: A Gentle Introduction"):
  - application_train.csv's real column names: SK_ID_CURR, TARGET (1 =
    client had payment difficulty), AMT_INCOME_TOTAL, AMT_CREDIT,
    AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_BIRTH, DAYS_EMPLOYED, CODE_GENDER,
    FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, CNT_FAM_MEMBERS,
    NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_INCOME_TYPE,
    NAME_HOUSING_TYPE, REGION_RATING_CLIENT, EXT_SOURCE_1, EXT_SOURCE_2,
    EXT_SOURCE_3.
  - The real dataset's documented ~8% default rate, used here purely as a
    plausible default (see DEFAULT_TARGET_RATE), not a claim about the real
    dataset's exact rate.
  - EXT_SOURCE_1/2/3 (normalized external credit bureau scores) are
    consistently reported, across essentially every public exploratory
    kernel on this competition, as the three single most predictive
    features, each negatively associated with default risk. Modeled here
    the same way, including EXT_SOURCE_1's real, heavily-documented
    missingness (roughly half of applicants have no EXT_SOURCE_1 value in
    the real data).
  - DAYS_BIRTH and DAYS_EMPLOYED are both stored as negative day counts in
    the real data (days before the application). DAYS_EMPLOYED carries a
    real, widely-documented data-quality artifact: unemployed, retired, and
    pensioner applicants are recorded with DAYS_EMPLOYED = 365243 (roughly
    1000 years), a placeholder value rather than a real employment length.
    This is not a synthetic invention; it is one of the most commonly
    discussed cleaning steps in public Home Credit kernels. It is
    reproduced here on purpose, because Week 9's WoE encoding module is
    written to handle it as its own bin rather than silently averaging it
    into "long employment," and that handling needs a real anomaly to be
    tested against.
  - NAME_EDUCATION_TYPE's real, documented association: higher education
    levels have a measurably lower default rate in the real data.

What is deliberately simplified from the real schema: the real
application_train.csv has 122 columns, many describing the applicant's
home/building characteristics (with heavy missingness) that this generator
omits entirely, keeping only the columns most public analyses treat as the
core, informative feature set. load_and_merge.py-style schema-agnostic
downstream code is not needed here since Week 9's own modules read this
generator's DataFrame directly, but the column NAMES above match the real
schema exactly, so this module's own code is meant to run unchanged against
the real downloaded CSV, just with more columns present.

Run directly to write a CSV to data/raw/ for local testing:
    python -m src.data.generate_synthetic_credit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TARGET_RATE = 0.08
DAYS_EMPLOYED_ANOMALY = 365243

# The 18 real Home Credit column names this generator's DataFrame carries,
# in the same order CreditScoringRequest (src/serving/schemas.py) lists
# them, excluding the id and target columns. Kept as a named constant, not
# just inline dict keys below, so src/data/load_credit_applications.py can
# import the exact same list to subset the REAL application_train.csv down
# to this same reduced schema, rather than maintaining a second, easily
# drifting copy of these 18 names. tests/test_generate_synthetic_credit.py
# asserts this constant matches this module's own DataFrame output columns.
APPLICATION_FEATURE_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE", "REGION_RATING_CLIENT", "EXT_SOURCE_1",
    "EXT_SOURCE_2", "EXT_SOURCE_3",
]

EDUCATION_TYPES = [
    "Lower secondary", "Secondary / secondary special", "Incomplete higher",
    "Higher education", "Academic degree",
]
EDUCATION_RISK_SHIFT = {
    "Lower secondary": 0.5, "Secondary / secondary special": 0.2,
    "Incomplete higher": 0.0, "Higher education": -0.35, "Academic degree": -0.6,
}
FAMILY_STATUS_TYPES = [
    "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
]
INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
HOUSING_TYPES = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"]


def generate_synthetic_credit_applications(
    n_applications: int = 5000,
    default_rate: float = DEFAULT_TARGET_RATE,
    seed: int = 42,
    start_id: int = 100001,
    unemployment_shift: float = 0.0,
    ext_source_shift: float = 0.0,
) -> pd.DataFrame:
    """Builds a synthetic application table with the real Home Credit
    schema's column names (reduced column count; see module docstring).

    default_rate is the target proportion of TARGET == 1 rows; the actual
    realized rate is reported by the caller from the returned DataFrame
    rather than assumed to exactly equal this input, since it is drawn
    from a Bernoulli process over a per-row risk score that itself has
    randomness in it.

    unemployment_shift and ext_source_shift exist purely for Week 9's PSI
    monitoring demo (see psi_monitoring.py and train_scorecard.py): calling
    this function twice with the same seed but a nonzero shift on the
    second call produces a genuinely different APPLICANT population, not
    just a different default_rate recalibration, so PSI has real covariate
    drift to detect rather than comparing two draws from the identical
    generating distribution. unemployment_shift adds directly to the
    "Unemployed" and "Pensioner" income-type probabilities (renormalized
    afterward), modeling a plausible economic downturn increasing the
    not-working share of applicants; ext_source_shift adds directly to the
    mean of all three EXT_SOURCE bureau scores, modeling a systematic drop
    in the applicant pool's credit-bureau-assessed quality. Both default to
    0.0 (no shift, the normal "development population" case).
    """
    if n_applications <= 0:
        raise ValueError("n_applications must be positive.")
    if not 0.0 < default_rate < 1.0:
        raise ValueError("default_rate must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = n_applications

    sk_id_curr = np.arange(start_id, start_id + n)

    income_type_probabilities = np.array([0.52, 0.23, 0.18, 0.06, 0.01])
    if unemployment_shift != 0.0:
        # Index 2 = "Pensioner", index 4 = "Unemployed" in INCOME_TYPES.
        income_type_probabilities = income_type_probabilities.copy()
        income_type_probabilities[2] += unemployment_shift / 2.0
        income_type_probabilities[4] += unemployment_shift / 2.0
        income_type_probabilities = np.clip(income_type_probabilities, 0.001, None)
        income_type_probabilities = income_type_probabilities / income_type_probabilities.sum()

    income_type = rng.choice(INCOME_TYPES, size=n, p=income_type_probabilities)
    is_not_working = np.isin(income_type, ["Pensioner", "Unemployed"])

    # EXT_SOURCE_1/2/3: the real dataset's three strongest predictors (see
    # module docstring). Each is an independent noisy draw, all negatively
    # associated with the eventual default risk score below. EXT_SOURCE_1
    # gets the real dataset's own heavy, documented missingness; EXT_SOURCE_2
    # and EXT_SOURCE_3 are missing far less often, also matching the real
    # data's own relative missingness ordering.
    ext_source_1 = np.clip(rng.normal(0.5 + ext_source_shift, 0.20, size=n), 0.0, 1.0)
    ext_source_2 = np.clip(rng.normal(0.5 + ext_source_shift, 0.18, size=n), 0.0, 1.0)
    ext_source_3 = np.clip(rng.normal(0.5 + ext_source_shift, 0.19, size=n), 0.0, 1.0)
    ext_source_1_observed = ext_source_1.copy()
    ext_source_1_observed[rng.random(n) < 0.56] = np.nan
    ext_source_2_observed = ext_source_2.copy()
    ext_source_2_observed[rng.random(n) < 0.002] = np.nan
    ext_source_3_observed = ext_source_3.copy()
    ext_source_3_observed[rng.random(n) < 0.20] = np.nan

    # DAYS_BIRTH: age 21-69, stored as a negative day count as in the real
    # schema. Older applicants get a lower default risk below, the real
    # dataset's own documented direction of effect.
    age_years = rng.uniform(21, 69, size=n)
    days_birth = -(age_years * 365.25).astype(int)

    # DAYS_EMPLOYED: real employment length for working applicants, but the
    # real dataset's DAYS_EMPLOYED_ANOMALY placeholder for anyone not
    # working (see module docstring); this is reproduced here on purpose; it
    # is not noise this generator forgot to clean.
    employed_years = np.clip(rng.exponential(6.0, size=n), 0, age_years - 16)
    days_employed = -(employed_years * 365.25).astype(int)
    days_employed[is_not_working] = DAYS_EMPLOYED_ANOMALY

    education_type = rng.choice(
        EDUCATION_TYPES, size=n, p=[0.04, 0.71, 0.10, 0.13, 0.02],
    )
    education_risk_shift = np.array([EDUCATION_RISK_SHIFT[e] for e in education_type])

    amt_income_total = np.round(rng.gamma(shape=2.2, scale=65000, size=n), 2)
    amt_credit = np.round(amt_income_total * rng.uniform(1.5, 6.0, size=n), 2)
    amt_annuity = np.round(amt_credit / rng.uniform(10, 30, size=n), 2)
    amt_goods_price = np.round(amt_credit * rng.uniform(0.85, 1.0, size=n), 2)
    credit_income_ratio = amt_credit / np.maximum(amt_income_total, 1.0)

    code_gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
    flag_own_car = rng.choice(["Y", "N"], size=n, p=[0.34, 0.66])
    flag_own_realty = rng.choice(["Y", "N"], size=n, p=[0.69, 0.31])
    cnt_children = rng.poisson(0.4, size=n)
    cnt_fam_members = np.clip(cnt_children + rng.integers(1, 3, size=n), 1, None)
    family_status = rng.choice(
        FAMILY_STATUS_TYPES, size=n, p=[0.64, 0.15, 0.10, 0.06, 0.05],
    )
    housing_type = rng.choice(
        HOUSING_TYPES, size=n, p=[0.88, 0.04, 0.04, 0.04],
    )
    region_rating_client = rng.choice([1, 2, 3], size=n, p=[0.10, 0.70, 0.20])

    # The per-row default risk score below is this generator's own,
    # disclosed synthetic construction: a plausible combination of
    # directions of effect that public Home Credit analyses report as real
    # (EXT_SOURCE scores dominate; older/longer-employed/more-educated
    # applicants are lower risk; a higher credit-to-income ratio is higher
    # risk), NOT a claim to reproduce the real dataset's actual coefficients.
    risk_score = (
        -3.2 * (ext_source_1 - 0.5)
        - 3.4 * (ext_source_2 - 0.5)
        - 3.0 * (ext_source_3 - 0.5)
        - 0.55 * ((age_years - 45) / 15)
        - 0.35 * np.clip((employed_years - 5) / 5, -1.5, 1.5)
        + education_risk_shift
        + 0.30 * np.clip((credit_income_ratio - 3.0) / 2.0, -1.0, 2.0)
        + rng.normal(0, 0.35, size=n)
    )
    # Calibrate the intercept by bisection on the synthetic risk_score so the
    # realized TARGET rate lands close to default_rate, the same calibration
    # idea generate_synthetic_campaign.py already used in Week 6 for its own
    # segment probabilities, applied here to a continuous score instead of
    # fixed per-segment probabilities.
    intercept = _calibrate_intercept(risk_score, default_rate)
    default_probability = 1.0 / (1.0 + np.exp(-(intercept + risk_score)))
    target = (rng.random(n) < default_probability).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr,
        "TARGET": target,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "CODE_GENDER": code_gender,
        "FLAG_OWN_CAR": flag_own_car,
        "FLAG_OWN_REALTY": flag_own_realty,
        "CNT_CHILDREN": cnt_children,
        "CNT_FAM_MEMBERS": cnt_fam_members,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_INCOME_TYPE": income_type,
        "NAME_HOUSING_TYPE": housing_type,
        "REGION_RATING_CLIENT": region_rating_client,
        "EXT_SOURCE_1": ext_source_1_observed,
        "EXT_SOURCE_2": ext_source_2_observed,
        "EXT_SOURCE_3": ext_source_3_observed,
    })
    return df


def _calibrate_intercept(risk_score: np.ndarray, target_rate: float) -> float:
    """Finds, by bisection, the intercept c such that
    mean(sigmoid(c + risk_score)) is as close as possible to target_rate.

    A closed-form solution does not exist once risk_score has real spread
    (the mean of a sigmoid over a distribution has no simple inverse), so
    this bisects on c directly: mean(sigmoid(c + risk_score)) is monotonic
    increasing in c, so the search always converges.
    """
    low, high = -20.0, 20.0
    for _ in range(60):
        mid = (low + high) / 2.0
        realized = 1.0 / (1.0 + np.exp(-(mid + risk_score)))
        if realized.mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Home-Credit-schema application "
        "CSV into data/raw/ for local testing (see module docstring for "
        "why real Kaggle data can't be fetched from this environment)."
    )
    parser.add_argument("--n-applications", type=int, default=5000)
    parser.add_argument("--default-rate", type=float, default=DEFAULT_TARGET_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw"),
        help="Directory to write application_train_synthetic.csv into.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_credit_applications(
        n_applications=args.n_applications, default_rate=args.default_rate, seed=args.seed,
    )
    out_path = args.out_dir / "application_train_synthetic.csv"
    df.to_csv(out_path, index=False)

    realized_rate = df["TARGET"].mean()
    print(f"Wrote {len(df)} synthetic applications to {out_path}")
    print(f"Realized synthetic default rate: {realized_rate:.4%} (target was {args.default_rate:.2%})")
    print(
        "Reminder: this is SYNTHETIC data for testing the pipeline only. "
        "Download the real Home Credit Default Risk files from Kaggle to "
        "get real numbers."
    )


if __name__ == "__main__":
    main()
