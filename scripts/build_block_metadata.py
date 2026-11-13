import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from feature_engineering.block_metadata import (
    DEFAULT_BLOCK_METADATA_PATH,
    DEFAULT_CLEANED_PATH,
    DEFAULT_DATASET_STATS_PATH,
    build_block_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-block metadata (human-readable component/event_type/IPs/"
        "timestamp) and dataset-wide distribution stats for the API to serve, without "
        "re-scanning the full cleaned-log CSV on every request."
    )
    parser.add_argument("--cleaned-path", default=str(DEFAULT_CLEANED_PATH))
    parser.add_argument("--block-metadata-output", default=str(DEFAULT_BLOCK_METADATA_PATH))
    parser.add_argument("--dataset-stats-output", default=str(DEFAULT_DATASET_STATS_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_block_metadata(
        cleaned_path=Path(args.cleaned_path),
        block_metadata_path=Path(args.block_metadata_output),
        dataset_stats_path=Path(args.dataset_stats_output),
    )
