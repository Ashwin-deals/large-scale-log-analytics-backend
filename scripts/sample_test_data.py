"""Sample random blocks from features.csv into an unlabeled test CSV.

features.csv already has no label column (anomaly_label.csv is only ever
joined in memory for training/evaluation, never written back to disk), so
this is a plain random sample -- it exists as a separate script so "new,
unlabeled data" has an obvious, reproducible source to run predict.py or
the API against without touching the training set's ground truth.

Usage:
    python3 scripts/sample_test_data.py
    python3 scripts/sample_test_data.py --n 200 --seed 7
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from feature_engineering.feature_extractor import FEATURE_COLUMNS

DEFAULT_FEATURES_PATH = Path("data/features/features.csv")
DEFAULT_OUTPUT_PATH = Path("data/samples/hdfs_test_sample.csv")


def sample_test_data(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    features = pd.read_csv(features_path)

    unexpected_label_columns = [c for c in features.columns if "label" in c.lower()]
    if unexpected_label_columns:
        raise ValueError(f"features.csv unexpectedly contains label-like columns: {unexpected_label_columns}")

    sample = features.sample(n=min(n, len(features)), random_state=seed).reset_index(drop=True)

    output_columns = ["block_id"] + FEATURE_COLUMNS
    sample = sample[output_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"Wrote {len(sample)} unlabeled rows to {output_path}")
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample an unlabeled test CSV from features.csv.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--n", type=int, default=500, help="Number of blocks to sample.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_test_data(Path(args.features), Path(args.output), n=args.n, seed=args.seed)
