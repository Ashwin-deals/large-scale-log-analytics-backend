"""Standalone manual-verification script for the deployed model.

Independent of scripts/generate_verification_sample.py: this script takes
ONLY a raw log file as input, runs it through the exact same production
pipeline (parser -> cleaner -> feature extractor -> currently deployed
model, loaded from disk, never retrained), and writes fresh predictions.

The ground-truth answer key is never read until AFTER predictions are
computed and written to disk -- it plays no part in parsing, cleaning,
feature extraction, or prediction. It is only used at the very end, to
join against the already-final predictions and print a comparison.

Usage:
    python3 scripts/verify_manually.py
    python3 scripts/verify_manually.py --raw-log data/testing/review_dataset_raw.log \\
        --answer-key data/testing/review_dataset_answer_key.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from parser.hdfs_parser import HDFSParser
from pipeline_api import resolve_current_deployment

DEFAULT_RAW_LOG_PATH = Path("data/testing/review_dataset_raw.log")
DEFAULT_ANSWER_KEY_PATH = Path("data/testing/review_dataset_answer_key.csv")

OUT_DIR = Path("data/testing")


def output_paths(raw_log_path: Path) -> dict[str, Path]:
    """Name every output after the input log.

    Fixed output names meant verifying a second sample silently overwrote the
    first one's predictions and metrics, leaving two datasets sharing one set
    of numbers — the kind of mix-up that is invisible until someone quotes the
    wrong figure.
    """
    stem = raw_log_path.stem
    return {
        "cleaned": OUT_DIR / f"{stem}_cleaned.csv",
        "features": OUT_DIR / f"{stem}_features.csv",
        "predictions": OUT_DIR / f"{stem}_predictions.csv",
        "metrics": Path("data/evaluation") / f"{stem}_metrics.json",
    }


def run_pipeline(raw_log_path: Path, out: dict[str, Path]) -> pd.DataFrame:
    """Parser -> cleaner -> feature extractor -> deployed model.

    Nothing here reads or references the answer key. The model is loaded
    from whatever data/models/current_version.json points at -- not
    retrained, not refit, just loaded as-is off disk.
    """
    print(f"[1/4] Parsing {raw_log_path} ...")
    frame = HDFSParser().parse(str(raw_log_path))
    recognized = frame[frame["date"].notna()].reset_index(drop=True)
    if len(recognized) == 0:
        raise SystemExit("No lines in the raw log matched the HDFS log format.")
    print(f"      {len(recognized):,} / {len(frame):,} lines recognized as HDFS log lines")

    print("[2/4] Cleaning ...")
    cleaned = HDFSDataCleaner().clean(recognized, out["cleaned"])

    print("[3/4] Extracting features ...")
    HDFSFeatureExtractor().extract(cleaned, out["features"])
    features = pd.read_csv(out["features"])
    print(f"      {len(features):,} blocks")

    current = resolve_current_deployment()
    model_path = current["model_path"]
    print(f"[4/4] Loading deployed model: v{current.get('version')} ({model_path}) -- not retrained")
    model = joblib.load(model_path)
    feature_columns = list(model.feature_names_in_)

    # features has no "Label" column at all, so predict() has nothing to
    # compare against -- true_label cannot leak into this step even by
    # accident. Only block_id, predicted_label, anomaly_score come out.
    predictions = predict(model, features, output_path=out["predictions"], feature_columns=feature_columns)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG_PATH)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY_PATH)
    args = parser.parse_args()

    if not args.raw_log.exists():
        raise SystemExit(f"{args.raw_log} not found.")

    out = output_paths(args.raw_log)
    predictions = run_pipeline(args.raw_log, out)
    print(f"\nWrote {out['predictions']} ({len(predictions):,} rows: block_id, predicted_label, anomaly_score)")
    print("No ground truth was read at any point up to this line.\n")

    # --- Only now does the answer key enter the picture -------------------
    if not args.answer_key.exists():
        print(f"Answer key {args.answer_key} not found -- stopping before comparison.")
        return

    answer_key = pd.read_csv(args.answer_key)
    joined = predictions.merge(
        answer_key.rename(columns={"true_label": "true_label"}), on="block_id", how="inner"
    )
    if len(joined) != len(predictions):
        print(
            f"Warning: {len(predictions) - len(joined)} predicted block(s) had no matching "
            f"answer-key row and were dropped before scoring."
        )

    metrics = evaluate(joined, metrics_path=out["metrics"])
    metrics["error_rate"] = 1 - metrics["accuracy"]
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["matrix"]
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0

    print("=== Fresh predictions vs. answer key ===")
    print_metrics_table(metrics)
    print(f"\nError rate:            {metrics['error_rate']:.4f}")
    print(f"False positive rate:   {metrics['false_positive_rate']:.4f}")
    print(f"False negative rate:   {metrics['false_negative_rate']:.4f}")
    print(f"\nWrote {out['metrics']}")


if __name__ == "__main__":
    main()
