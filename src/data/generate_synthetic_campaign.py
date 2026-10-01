"""
generate_synthetic_campaign.py, Week 6: synthetic data for FraudLens's
marketing uplift module.

Uplift modeling answers a different question than Weeks 1 through 5's fraud
models. Those models estimate P(fraud | transaction): a plain prediction.
Uplift modeling estimates P(convert | treated) - P(convert | not treated)
for the SAME customer: a causal, incremental effect. That effect cannot be
observed directly for any single customer (nobody is both treated and
untreated at once, the fundamental problem of causal inference), so this
generator plays the role real data never can: it knows and records each
customer's true, unobservable, individual-level uplift, which is what lets
tests below check the estimators against ground truth instead of only
checking that they run.

To make that possible without confounding, every customer is assigned to
treatment or control independently of their features, at random, exactly
the way a real randomized marketing holdout works (the standard practice
this project assumes: a control group withheld from the offer, purely by
chance, not by any business rule). Uplift modeling on data where treatment
was NOT randomly assigned (say, a bank's front-line staff already targeted
its best customers) requires causal-inference machinery, propensity
adjustment, this project deliberately does not attempt. Every model here
inherits validity from that random assignment; it is the single most
important assumption in this module, and it is enforced here, in the data
itself, rather than merely asserted.

Every customer belongs to one of four canonical uplift segments from the
marketing-response literature, and the segment (never given to a model as
a feature) fully determines two fixed conversion probabilities: p0 (chance
of converting if left alone) and p1 (chance of converting if treated).
    - "persuadable": p0 low, p1 high. The offer is the reason they convert;
      this is who a targeted campaign exists to find.
    - "sure_thing": p0 and p1 both high. They convert either way, so
      treating them is wasted spend, even though a plain response model
      (which cannot see the untreated counterfactual) would rate them as
      great targets.
    - "lost_cause": p0 and p1 both low. Nothing moves them either way.
    - "sleeping_dog": p1 LOWER than p0. The offer actively drives them away
      (a real, well-documented pattern, e.g. a renewal offer reminding an
      otherwise-inertial customer to shop around). A plain response model
      cannot even represent a negative effect; only comparing treated
      against control can reveal one.
A model that only ranks customers by predicted response (Weeks 1-5's kind
of model, pointed at this data) would rank sure_things at the top and
completely miss that persuadables are the only segment worth spending the
offer on, and would never notice sleeping_dogs should be left alone. That
contrast is the entire reason this module exists, and
tests/test_generate_synthetic_campaign.py checks it directly.

Each segment also has its own feature distribution (below), overlapping
between segments on purpose: a real uplift model has to work with
imperfect, overlapping evidence about who is a persuadable, not a clean
label. tenure_months is deliberately drawn the same way for every segment,
a pure noise feature with no relationship to segment or outcome, included
because a realistic feature set always has a few of these and a model
should not be rewarded or punished for how it handles them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

SEGMENTS = ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]

DEFAULT_SEGMENT_SHARES: Dict[str, float] = {
    "persuadable": 0.25,
    "sure_thing": 0.25,
    "lost_cause": 0.30,
    "sleeping_dog": 0.20,
}

# Fixed, hand-chosen conversion probabilities without (p0) and with (p1)
# treatment, per segment. true_uplift = p1 - p0. These are constants, not
# estimated from anything, so tests can check them exactly.
SEGMENT_PROBABILITIES: Dict[str, Dict[str, float]] = {
    "persuadable": {"p0": 0.10, "p1": 0.45},
    "sure_thing": {"p0": 0.75, "p1": 0.80},
    "lost_cause": {"p0": 0.05, "p1": 0.07},
    "sleeping_dog": {"p0": 0.35, "p1": 0.15},
}

# Mean of each feature by segment, with a shared standard deviation so the
# segments' distributions genuinely overlap (see module docstring). Only
# engagement_score, avg_monthly_spend, credit_utilization, and
# prior_offers_accepted carry any segment signal; tenure_months does not.
FEATURE_MEANS: Dict[str, Dict[str, float]] = {
    "engagement_score": {
        "persuadable": 50.0, "sure_thing": 80.0, "lost_cause": 20.0, "sleeping_dog": 55.0,
    },
    "avg_monthly_spend": {
        "persuadable": 120.0, "sure_thing": 200.0, "lost_cause": 60.0, "sleeping_dog": 110.0,
    },
    "credit_utilization": {
        "persuadable": 0.45, "sure_thing": 0.30, "lost_cause": 0.70, "sleeping_dog": 0.40,
    },
    "prior_offers_accepted": {
        "persuadable": 1.5, "sure_thing": 2.5, "lost_cause": 0.5, "sleeping_dog": 4.0,
    },
}
FEATURE_STDS: Dict[str, float] = {
    "engagement_score": 15.0,
    "avg_monthly_spend": 40.0,
    "credit_utilization": 0.15,
    "prior_offers_accepted": 1.2,
}

FEATURE_COLUMNS = [
    "engagement_score",
    "tenure_months",
    "avg_monthly_spend",
    "credit_utilization",
    "prior_offers_accepted",
]


def _validate_segment_shares(segment_shares: Dict[str, float]) -> None:
    if set(segment_shares.keys()) != set(SEGMENTS):
        raise ValueError(f"segment_shares must have exactly the keys {sorted(SEGMENTS)}.")
    if any(share < 0 for share in segment_shares.values()):
        raise ValueError("segment_shares values must all be non-negative.")
    total = sum(segment_shares.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"segment_shares must sum to 1.0, got {total}.")


def generate_synthetic_campaign(
    n_customers: int = 10000,
    treatment_rate: float = 0.5,
    segment_shares: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a synthetic randomized marketing campaign dataset.

    Returns a DataFrame with one row per customer:
      - customer_id: a unique integer id.
      - The columns in FEATURE_COLUMNS: the only columns an uplift model is
        allowed to see.
      - treatment: 1 if this customer was (randomly) offered the campaign,
        0 if held out as control. Independent of segment and features; see
        module docstring for why that independence is load-bearing.
      - outcome: 1 if this customer converted, drawn as
        Bernoulli(p1) if treatment == 1 else Bernoulli(p0), using this
        customer's segment's fixed probabilities.
      - segment, true_uplift: ground truth, present ONLY for evaluation and
        testing. Real campaign data never has these columns; production
        code must not use them as model inputs.

    Raises ValueError if n_customers is not positive, if treatment_rate is
    not strictly between 0 and 1, or if segment_shares (when given) does not
    have exactly SEGMENTS as keys with non-negative values summing to 1.0.
    """
    if n_customers <= 0:
        raise ValueError("n_customers must be positive.")
    if not (0.0 < treatment_rate < 1.0):
        raise ValueError("treatment_rate must be strictly between 0 and 1.")
    if segment_shares is None:
        segment_shares = DEFAULT_SEGMENT_SHARES
    _validate_segment_shares(segment_shares)

    rng = np.random.default_rng(seed)

    segment = rng.choice(
        SEGMENTS, size=n_customers, p=[segment_shares[s] for s in SEGMENTS],
    )

    data: Dict[str, np.ndarray] = {"customer_id": np.arange(1, n_customers + 1)}

    for feature in FEATURE_COLUMNS:
        if feature == "tenure_months":
            # Deliberately segment-independent noise feature; see docstring.
            data[feature] = rng.integers(1, 121, size=n_customers).astype(float)
            continue
        means_by_segment = FEATURE_MEANS[feature]
        means = np.array([means_by_segment[s] for s in segment])
        values = rng.normal(loc=means, scale=FEATURE_STDS[feature])
        if feature in ("credit_utilization",):
            values = np.clip(values, 0.0, 1.0)
        elif feature == "prior_offers_accepted":
            values = np.clip(np.round(values), 0, None)
        else:
            values = np.clip(values, 0.0, None)
        data[feature] = values

    treatment = (rng.random(n_customers) < treatment_rate).astype(int)

    p0 = np.array([SEGMENT_PROBABILITIES[s]["p0"] for s in segment])
    p1 = np.array([SEGMENT_PROBABILITIES[s]["p1"] for s in segment])
    conversion_prob = np.where(treatment == 1, p1, p0)
    outcome = (rng.random(n_customers) < conversion_prob).astype(int)

    data["treatment"] = treatment
    data["outcome"] = outcome
    data["segment"] = segment
    data["true_uplift"] = p1 - p0

    df = pd.DataFrame(data)
    return df[
        ["customer_id"] + FEATURE_COLUMNS + ["treatment", "outcome", "segment", "true_uplift"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic randomized marketing campaign "
        "dataset for FraudLens's Week 6 uplift module."
    )
    parser.add_argument("--n-customers", type=int, default=10000)
    parser.add_argument("--treatment-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/campaign.csv"),
    )
    args = parser.parse_args()

    df = generate_synthetic_campaign(
        n_customers=args.n_customers, treatment_rate=args.treatment_rate, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(df)} rows to {args.output}. "
        f"Treatment rate: {df['treatment'].mean():.3f}. "
        f"Overall conversion rate: {df['outcome'].mean():.3f}."
    )


if __name__ == "__main__":
    main()
