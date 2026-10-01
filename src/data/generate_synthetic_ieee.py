"""
generate_synthetic_ieee.py, Week 1: a synthetic stand-in for the real
IEEE-CIS Fraud Detection dataset (train_transaction.csv / train_identity.csv),
used ONLY because this sandbox cannot reach kaggle.com to download the real
files (confirmed directly: `curl` against kaggle.com and its API both return
`CONNECT tunnel failed, response 403`, the same class of restriction this
project has already hit for huggingface.co and the real Anthropic API in the
sibling SearchLens project).

This generator exists so the loading/merging/EDA code in this package
(load_and_merge.py, run_eda.py) can be built and tested for real in the
sandbox, then run unchanged against the real Kaggle CSVs once you've
downloaded them on your own machine. Nothing here is a substitute for real
EDA. Every number this produces is synthetic and is never reported as a
real finding about fraud. Once you run this same pipeline against the real
data, report whatever it actually measures.

What this mimics from the real dataset (per the IEEE-CIS Kaggle competition's
own data description):
  - train_transaction.csv columns: TransactionID, isFraud, TransactionDT,
    TransactionAmt, ProductCD, card1-card6, addr1-addr2, dist1-dist2,
    P_emaildomain, R_emaildomain, C1-C14, D1-D15, M1-M9, V1-V339.
  - train_identity.csv columns: TransactionID, id_01-id_38, DeviceType,
    DeviceInfo.
  - Only a MINORITY of transactions have a matching identity record (the
    real competition data page states this explicitly), modeled here via
    `identity_coverage_rate`.
  - TransactionDT is seconds elapsed from an arbitrary, undisclosed
    reference point, NOT a real Unix timestamp. It is modeled here as a
    plain increasing integer counter with jitter, deliberately not treated
    as a real calendar date anywhere downstream.
  - Real-world fraud rate for this competition is documented (by many public
    writeups of the competition) as roughly 3.5%; used as this generator's
    default purely as a plausible realistic value for testing code paths
    (imbalance handling, EDA on a rare positive class), NOT as a claim about
    the real dataset's exact rate.

What is deliberately reduced from the real schema, and why that's safe:
  - The real data has V1-V339, C1-C14, D1-D15, M1-M9. This generator emits
    C1-C14, D1-D15, and M1-M9 in full (cheap), but only V1-V20 rather than
    all 339, purely to keep the synthetic files small and fast to test
    against. This is safe because load_and_merge.py and run_eda.py are
    written to work off of whatever columns are actually present in a CSV
    (they never hardcode "V1..V339"), so pointing them at the real files
    with all 339 V columns exercises the exact same code paths.

Run directly to write CSVs to data/raw/ for local testing:
    python -m src.data.generate_synthetic_ieee
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reduced from the real dataset's 339 V-columns; see module docstring for why.
N_V_COLUMNS = 20
N_C_COLUMNS = 14
N_D_COLUMNS = 15
N_M_COLUMNS = 9
N_ID_COLUMNS = 38  # id_01..id_38, matches the real schema exactly

PRODUCT_CODES = ["W", "C", "R", "H", "S"]
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "anonymous.com", "icloud.com",
]
DEVICE_TYPES = ["desktop", "mobile"]
DEVICE_INFO_SAMPLES = [
    "Windows", "iOS Device", "MacOS", "SM-G960U Build/PPR1.180610.011",
    "Trident/7.0", "SM-N960U Build/PQ3A.190801.002", "Linux",
]


def generate_synthetic_transactions(
    n_transactions: int = 5000,
    fraud_rate: float = 0.035,
    seed: int = 42,
    start_transaction_id: int = 2987000,
    fraud_ring_rate: float = 0.3,
    n_rings: int = 3,
) -> pd.DataFrame:
    """Builds a synthetic transactions table with the real schema's column
    names (reduced V-column count; see module docstring).

    `fraud_rate` is the target proportion of isFraud == 1 rows; the actual
    realized rate is reported by the caller from the returned DataFrame
    rather than assumed to exactly equal this input, since it's drawn from
    a Bernoulli process.

    `fraud_ring_rate` and `n_rings` control a Week 5 addition: the fraction
    of fraud rows (not of all rows) that get their card1 and P_emaildomain
    overridden to one of a small, fixed set of "ring" identifiers, shared
    across otherwise unrelated fraud transactions. See the inline comment
    below for why this exists.
    """
    if n_transactions <= 0:
        raise ValueError("n_transactions must be positive.")
    if not 0.0 < fraud_rate < 1.0:
        raise ValueError("fraud_rate must be strictly between 0 and 1.")
    if not 0.0 <= fraud_ring_rate <= 1.0:
        raise ValueError("fraud_ring_rate must be between 0 and 1.")
    if n_rings < 0:
        raise ValueError("n_rings must be non-negative.")

    rng = np.random.default_rng(seed)

    transaction_id = np.arange(
        start_transaction_id, start_transaction_id + n_transactions
    )
    is_fraud = rng.binomial(1, fraud_rate, size=n_transactions)

    # TransactionDT: seconds since an arbitrary reference point, strictly
    # increasing with random gaps. This matches the real column's documented
    # semantics (not a real calendar timestamp). Spread over ~180 days.
    gaps = rng.integers(1, 600, size=n_transactions)
    transaction_dt = np.cumsum(gaps)

    # Fraudulent transactions skew toward higher amounts and specific
    # products/domains, purely so the EDA code has real, nonzero group
    # differences to detect and report. These are synthetic patterns baked
    # in on purpose, not a claim about real fraud behavior.
    base_amt = rng.gamma(shape=2.0, scale=40.0, size=n_transactions)
    fraud_amt_boost = is_fraud * rng.gamma(shape=2.0, scale=60.0, size=n_transactions)
    transaction_amt = np.round(base_amt + fraud_amt_boost, 2)

    product_cd = rng.choice(PRODUCT_CODES, size=n_transactions, p=[0.55, 0.15, 0.1, 0.1, 0.1])

    # card1 identifies a card, and real cards get reused across many
    # transactions (that reuse is the entire premise of Week 2's velocity
    # and card-usage features). Drawing independently and uniformly per
    # row from the full 1000-18000 range, as an earlier version of this
    # generator did, gives almost no repeats at typical synthetic sizes
    # (confirmed directly: 5000 rows produced 4313 distinct card1 values,
    # over 86 percent singletons), which made every velocity and
    # card-usage feature degenerate to "no prior history" for nearly every
    # row and made Week 2's groupby-per-card feature code pay Python-level
    # per-group overhead for thousands of one-row groups (a real, measured
    # 33 second runtime on just 5000 rows, confirmed with a profiler before
    # this fix). Drawing from a bounded pool of card values instead, with
    # replacement, gives realistic reuse (about 20 transactions per card on
    # average here), which is both faster and a more meaningful exercise of
    # the Week 2 feature code.
    n_distinct_cards = max(1, n_transactions // 20)
    card_pool = rng.integers(1000, 18000, size=n_distinct_cards)
    card1 = rng.choice(card_pool, size=n_transactions)
    card2 = rng.integers(100, 600, size=n_transactions).astype(float)
    card3 = rng.choice([150.0, 185.0, 100.0], size=n_transactions)
    card4 = rng.choice(["visa", "mastercard", "american express", "discover"], size=n_transactions)
    card5 = rng.integers(100, 240, size=n_transactions).astype(float)
    card6 = rng.choice(["debit", "credit"], size=n_transactions, p=[0.7, 0.3])

    addr1 = rng.integers(100, 500, size=n_transactions).astype(float)
    addr2 = rng.choice([87.0, 60.0, 96.0], size=n_transactions)

    # dist1/dist2 are sparse in the real data (mostly missing), modeled
    # the same way here so missing-value-rate EDA has something real to find.
    dist1 = np.where(rng.random(n_transactions) < 0.6, np.nan, rng.exponential(20, n_transactions))
    dist2 = np.where(rng.random(n_transactions) < 0.9, np.nan, rng.exponential(50, n_transactions))

    p_email = rng.choice(EMAIL_DOMAINS + [None], size=n_transactions, p=[0.12] * 7 + [0.16])
    r_email = rng.choice(EMAIL_DOMAINS + [None], size=n_transactions, p=[0.07] * 7 + [0.51])

    # Week 5 needs a pattern in this data that a graph model can find and a
    # row-by-row tabular model cannot: several fraud transactions sharing
    # an identifier specifically because they belong to the same ring, not
    # because that identifier happens to be popular overall. Without this,
    # every card1 and P_emaildomain value here is used by fraud and
    # non-fraud rows in the same proportion (no more than chance reuse), so
    # a graph built on shared entities would carry no fraud signal at all,
    # and testing graph fraud detection against this data would not
    # exercise it meaningfully either way.
    #
    # A small, fixed number of "ring" card1 and P_emaildomain values are
    # picked below, and a fraud_ring_rate fraction of fraud rows (not of
    # all rows) are reassigned to reuse one of them instead of their
    # originally sampled value. This creates real, detectable clustering:
    # several otherwise-unrelated fraud transactions sharing an identifier
    # at a rate far above what chance reuse of a popular card or common
    # email domain would produce, which is exactly what entity-linked graph
    # fraud detection is built to exploit.
    fraud_indices = np.flatnonzero(is_fraud == 1)
    n_ring_members = int(round(len(fraud_indices) * fraud_ring_rate))
    if n_rings > 0 and n_ring_members > 0:
        ring_member_indices = rng.choice(fraud_indices, size=n_ring_members, replace=False)
        ring_assignment = rng.integers(0, n_rings, size=n_ring_members)

        ring_cards = rng.choice(card_pool, size=min(n_rings, len(card_pool)), replace=False)
        ring_emails = rng.choice(EMAIL_DOMAINS, size=min(n_rings, len(EMAIL_DOMAINS)), replace=False)

        card1[ring_member_indices] = ring_cards[ring_assignment % len(ring_cards)]
        p_email[ring_member_indices] = ring_emails[ring_assignment % len(ring_emails)]

    data = {
        "TransactionID": transaction_id,
        "isFraud": is_fraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "addr1": addr1,
        "addr2": addr2,
        "dist1": dist1,
        "dist2": dist2,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
    }

    # C1-C14: counting features in the real data (e.g. how many addresses
    # tied to a card), modeled as small non-negative integer counts.
    for i in range(1, N_C_COLUMNS + 1):
        data[f"C{i}"] = rng.poisson(lam=2.0, size=n_transactions)

    # D1-D15: time-delta features in the real data (e.g. days since last
    # transaction with this card), modeled as non-negative floats with
    # realistic missingness, since several D-columns are heavily missing in
    # the real dataset.
    for i in range(1, N_D_COLUMNS + 1):
        vals = rng.exponential(scale=30.0, size=n_transactions)
        missing_mask = rng.random(n_transactions) < 0.3
        vals[missing_mask] = np.nan
        data[f"D{i}"] = vals

    # M1-M9: match flags in the real data (e.g. does the billing name match
    # the card), modeled as T/F/missing categoricals.
    for i in range(1, N_M_COLUMNS + 1):
        data[f"M{i}"] = rng.choice(["T", "F", None], size=n_transactions, p=[0.45, 0.35, 0.2])

    # V1-V20: Vesta-engineered features in the real data (V1-V339), kept
    # small here for speed; see module docstring for why this is a safe
    # reduction. Slightly correlated with isFraud so hybrid feature-vs-label
    # EDA has something real to show.
    for i in range(1, N_V_COLUMNS + 1):
        base = rng.normal(loc=0.0, scale=1.0, size=n_transactions)
        data[f"V{i}"] = base + is_fraud * rng.normal(loc=0.8, scale=0.3, size=n_transactions)

    df = pd.DataFrame(data)
    return df


def generate_synthetic_identity(
    transaction_ids: np.ndarray,
    coverage_rate: float = 0.24,
    seed: int = 43,
) -> pd.DataFrame:
    """Builds a synthetic identity table covering only a subset of the given
    transaction IDs. The real dataset's identity file covers roughly a
    quarter of all transactions (per the competition's own data
    description), which is exactly the property load_and_merge.py's left
    join needs to be tested against (most transactions have NO identity
    match and must come through the merge with NaN identity columns, not
    get silently dropped).
    """
    if not 0.0 < coverage_rate <= 1.0:
        raise ValueError("coverage_rate must be in (0, 1].")

    rng = np.random.default_rng(seed)
    n_covered = int(round(len(transaction_ids) * coverage_rate))
    covered_ids = rng.choice(transaction_ids, size=n_covered, replace=False)
    covered_ids.sort()

    data = {"TransactionID": covered_ids}
    for i in range(1, N_ID_COLUMNS + 1):
        col = f"id_{i:02d}"
        vals = rng.normal(loc=0.0, scale=1.0, size=n_covered)
        missing_mask = rng.random(n_covered) < 0.4
        vals = vals.astype(object)
        vals[missing_mask] = None
        data[col] = vals

    data["DeviceType"] = rng.choice(DEVICE_TYPES + [None], size=n_covered, p=[0.45, 0.45, 0.1])
    data["DeviceInfo"] = rng.choice(
        DEVICE_INFO_SAMPLES + [None], size=n_covered, p=[0.13] * 7 + [0.09]
    )

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic IEEE-CIS-schema CSVs into data/raw/ "
        "for local testing (see module docstring for why real Kaggle data "
        "can't be fetched from this environment)."
    )
    parser.add_argument("--n-transactions", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument("--identity-coverage", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fraud-ring-rate", type=float, default=0.3,
        help="Fraction of fraud rows reassigned to a shared ring identifier; see module docstring.",
    )
    parser.add_argument("--n-rings", type=int, default=3)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw"),
        help="Directory to write train_transaction.csv / train_identity.csv into.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    transactions = generate_synthetic_transactions(
        n_transactions=args.n_transactions, fraud_rate=args.fraud_rate, seed=args.seed,
        fraud_ring_rate=args.fraud_ring_rate, n_rings=args.n_rings,
    )
    identity = generate_synthetic_identity(
        transactions["TransactionID"].to_numpy(),
        coverage_rate=args.identity_coverage,
        seed=args.seed + 1,
    )

    transactions_path = args.out_dir / "train_transaction.csv"
    identity_path = args.out_dir / "train_identity.csv"
    transactions.to_csv(transactions_path, index=False)
    identity.to_csv(identity_path, index=False)

    realized_fraud_rate = transactions["isFraud"].mean()
    print(f"Wrote {len(transactions)} synthetic transactions to {transactions_path}")
    print(f"Wrote {len(identity)} synthetic identity records to {identity_path}")
    print(f"Realized synthetic fraud rate: {realized_fraud_rate:.4%} (target was {args.fraud_rate:.2%})")
    print(
        "Reminder: this is SYNTHETIC data for testing the pipeline only. "
        "Download the real IEEE-CIS files from Kaggle to get real numbers."
    )


if __name__ == "__main__":
    main()
