import argparse
import sys
import tempfile
from itertools import islice
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from parser.hdfs_parser import HDFSParser


def build_hdfs_features(
    raw_log_path: Path,
    structured_output_path: Path,
    cleaned_output_path: Path,
    feature_output_path: Path,
    limit: int | None = None,
) -> None:
    parser_input_path = raw_log_path
    temp_file = None

    if limit is not None:
        temp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        with raw_log_path.open("r", encoding="utf-8", errors="replace") as raw_file:
            temp_file.writelines(islice(raw_file, limit))
        temp_file.close()
        parser_input_path = Path(temp_file.name)

    try:
        structured_output_path.parent.mkdir(parents=True, exist_ok=True)
        structured_logs = HDFSParser().parse(str(parser_input_path))
        structured_logs.to_csv(structured_output_path, index=False)

        cleaned_logs = HDFSDataCleaner().clean(structured_logs, cleaned_output_path)
        HDFSFeatureExtractor().extract(cleaned_logs, feature_output_path)
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured, cleaned, and feature CSVs for HDFS logs."
    )
    parser.add_argument("--raw-log", default="data/raw/hdfs/HDFS.log")
    parser.add_argument("--structured-output", default="data/processed/hdfs_structured.csv")
    parser.add_argument("--cleaned-output", default="data/processed/hdfs_cleaned.csv")
    parser.add_argument("--feature-output", default="data/features/features.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of raw log lines to process for smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_hdfs_features(
        raw_log_path=Path(args.raw_log),
        structured_output_path=Path(args.structured_output),
        cleaned_output_path=Path(args.cleaned_output),
        feature_output_path=Path(args.feature_output),
        limit=args.limit,
    )
