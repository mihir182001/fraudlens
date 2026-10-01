"""
build_merged_dataset.py, Week 1: thin CLI wrapper that loads
train_transaction.csv + train_identity.csv (real or synthetic), merges them
with load_and_merge.load_and_merge, and writes the result to
data/processed/merged.csv for run_eda.py (and later weeks' feature
engineering) to read.

Kept separate from load_and_merge.py itself so that module stays pure
functions with no filesystem-path defaults baked in, and separate from
generate_synthetic_ieee.py so this exact command is what you'll also run
against the real Kaggle files. Only the input CSVs change, not this
script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load_and_merge import identity_match_rate, load_and_merge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and merge the IEEE-CIS transaction + identity "
        "CSVs (real or synthetic) into a single processed file."
    )
    parser.add_argument(
        "--transaction-csv", type=Path, default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"),
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("data/processed/merged.csv"),
    )
    args = parser.parse_args()

    merged = load_and_merge(args.transaction_csv, args.identity_csv)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    match_rate = identity_match_rate(merged)
    print(f"Merged {len(merged)} transactions into {args.out_csv}")
    print(f"Identity match rate: {match_rate:.2%} of transactions had a matching identity record")


if __name__ == "__main__":
    main()
