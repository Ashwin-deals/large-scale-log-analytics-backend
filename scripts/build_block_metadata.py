import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from feature_engineering.feature_extractor import HDFSFeatureExtractor

DEFAULT_CLEANED_PATH = Path("data/processed/hdfs_cleaned.csv")
DEFAULT_BLOCK_METADATA_PATH = Path("data/processed/block_metadata.csv")
DEFAULT_DATASET_STATS_PATH = Path("data/processed/dataset_stats.json")

USE_COLUMNS = [
    "datetime",
    "log_level",
    "component",
    "event_type",
    "block_id",
    "source_ip",
    "destination_ip",
    "block_size",
]
STRING_COLUMNS = ["component", "event_type", "source_ip", "destination_ip", "block_id"]
DOMINANT_VALUE_COLUMNS = ["component", "event_type", "source_ip", "destination_ip"]


def build_block_metadata(
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    block_metadata_path: Path = DEFAULT_BLOCK_METADATA_PATH,
    dataset_stats_path: Path = DEFAULT_DATASET_STATS_PATH,
) -> None:
    frame = pd.read_csv(cleaned_path, usecols=USE_COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["block_size"] = pd.to_numeric(frame["block_size"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["datetime"]).copy()
    frame = frame.sort_values("datetime").reset_index(drop=True)

    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip().fillna("UNKNOWN")
        frame[column] = frame[column].replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})

    grouped = frame.groupby("block_id", sort=False)
    aggregated = grouped.agg(
        first_seen=("datetime", "first"),
        block_size=("block_size", "max"),
    ).reset_index()

    for column in DOMINANT_VALUE_COLUMNS:
        dominant = HDFSFeatureExtractor._dominant_value_per_group(frame, "block_id", column)
        aggregated[column] = aggregated["block_id"].map(dominant)

    block_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(block_metadata_path, index=False)

    events_per_day = (
        frame["datetime"].dt.date.astype("string").value_counts().sort_index()
    )
    stats = {
        "total_events": int(len(frame)),
        "total_blocks": int(len(aggregated)),
        "log_level_distribution": frame["log_level"].value_counts().to_dict(),
        "event_type_distribution": frame["event_type"].value_counts().to_dict(),
        "component_distribution": frame["component"].value_counts().to_dict(),
        "date_range": {
            "min": frame["datetime"].min().isoformat(),
            "max": frame["datetime"].max().isoformat(),
        },
        "events_per_day": events_per_day.to_dict(),
    }
    dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)


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
