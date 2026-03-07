# scripts/anubis_label_stats.py
#
# ANUBIS Label Distribution Quick Audit
# ─────────────────────────────────────
# Utility script to inspect class distributions for train/val label files.
# Prints basic stats per split:
#   - shape / min / max label
#   - number of unique classes
#   - number of zero-count classes
#   - min nonzero class count
#   - max class count

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect ANUBIS label distribution")
    parser.add_argument(
        "--labels-dir",
        default="data/anubis",
        help="Directory containing <split>_labels_cleaned.npy files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["trn", "val"],
        help="Split name prefixes to inspect (default: trn val)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=102,
        help="Total number of classes for bincount minlength",
    )
    return parser.parse_args()


def inspect_split(labels_dir: str, split: str, num_classes: int) -> None:
    """Load one split label file and print distribution summary."""
    path = os.path.join(labels_dir, f"{split}_labels_cleaned.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Label file not found: {path}\nCheck --labels-dir and --splits values."
        )

    y = np.load(path)
    counts = np.bincount(y, minlength=num_classes)
    nonzero = counts[counts > 0]

    print(
        f"{split} shape {y.shape} min {y.min()} max {y.max()} "
        f"unique {len(np.unique(y))}"
    )
    print(
        "  zero-count classes:",
        int((counts == 0).sum()),
        "min nonzero:",
        int(nonzero.min()) if nonzero.size > 0 else 0,
        "max:",
        int(counts.max()),
    )


def main() -> None:
    """Run label-distribution checks for requested splits."""
    args = parse_args()
    for split in args.splits:
        inspect_split(args.labels_dir, split, args.num_classes)


if __name__ == "__main__":
    main()
