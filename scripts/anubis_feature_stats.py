# scripts/anubis_feature_stats.py
#
# ANUBIS Feature Distribution Quick Audit
# ───────────────────────────────────────
# Utility script to inspect train/val feature arrays and print channel-wise
# mean and std for quick domain-shift diagnostics.

from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect ANUBIS feature distribution")
    parser.add_argument(
        "--features-dir",
        default="data/anubis",
        help="Directory containing <split>_features_cleaned.npy files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["trn", "val"],
        help="Split name prefixes to inspect (default: trn val)",
    )
    return parser.parse_args()


def inspect_split(features_dir: str, split: str) -> None:
    """Load one split feature file and print summary stats."""
    path = os.path.join(features_dir, f"{split}_features_cleaned.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}\nCheck --features-dir and --splits values."
        )

    x = np.load(path)
    if x.ndim not in (4, 5):
        raise ValueError(
            f"Expected feature tensor to be 4D or 5D, got shape {x.shape}."
        )

    print(f"{split} shape {x.shape} dtype {x.dtype}")

    # x expected as (N, C, T, J) or (N, C, T, J, M)
    reduce_axes = (0, 2, 3) if x.ndim == 4 else (0, 2, 3, 4)
    ch_mean = x.mean(axis=reduce_axes)
    ch_std = x.std(axis=reduce_axes)

    print("  channel mean:", np.round(ch_mean, 4))
    print("  channel std :", np.round(ch_std, 4))


def main() -> None:
    """Run feature-distribution checks for requested splits."""
    args = parse_args()
    for split in args.splits:
        inspect_split(args.features_dir, split)


if __name__ == "__main__":
    main()
