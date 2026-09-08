"""Sample a slice of the raw HDFS log to verify the deployed model.

Everything the deployed model has been scored on so far (data/evaluation/*.json)
comes from features.csv, i.e. logs the pipeline already parsed. This script
instead starts from HDFS.log itself and walks the exact same path a real
Log Sources upload takes:

    raw log lines -> HDFSParser -> HDFSDataCleaner -> HDFSFeatureExtractor
    -> detection.predict.predict() against whichever model
    data/models/current_version.json says is deployed (no retraining).

Steps:
  1. Stratified-sample N block_ids from anomaly_label.csv (stratified so the
     rare Anomaly class, ~2.9% of blocks, isn't sampled away).
  2. Stream HDFS.log once and copy every raw line belonging to those blocks
     into a single unlabeled .log file — data/testing/verification_sample.log.
     This is a plain HDFS-format log file: it can be uploaded by hand through
     the dashboard's Log Sources page, same as any other log.
  3. Run that file through the real pipeline and score it against the
     currently deployed model.
  4. Join predictions back to the true labels (kept in a *separate* file —
     the "upload" itself never carries them) and report accuracy/precision/
     recall/F1/error rate plus a confusion matrix.

Nothing here is committed to git: data/testing/ is already gitignored.

Usage:
    python3 scripts/generate_verification_sample.py
    python3 scripts/generate_verification_sample.py --n 8000 --seed 7
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from parser.hdfs_parser import BLOCK_ID_PATTERN, HDFSParser
from pipeline_api import resolve_current_deployment

RAW_LOG_PATH = Path("data/raw/hdfs/HDFS.log")
LABELS_PATH = Path("data/raw/hdfs/anomaly_label.csv")

OUT_DIR = Path("data/testing")
SAMPLE_RAW_LOG_PATH = OUT_DIR / "verification_sample.log"                  # unlabeled — upload this by hand
SAMPLE_TRUE_LABELS_PATH = OUT_DIR / "verification_sample_true_labels.csv"  # ground truth, kept separate
SAMPLE_CLEANED_PATH = OUT_DIR / "verification_sample_cleaned.csv"
SAMPLE_FEATURES_PATH = OUT_DIR / "verification_sample_features.csv"
SAMPLE_PREDICTIONS_PATH = OUT_DIR / "verification_predictions.csv"
METRICS_PATH = Path("data/evaluation/verification_sample_metrics.json")

DEFAULT_N = 5000
RANDOM_STATE = 42


def sample_block_ids(n: int, seed: int) -> tuple[set, pd.DataFrame]:
    labels = pd.read_csv(LABELS_PATH)
    if n >= len(labels):
        return set(labels["BlockId"]), labels
    sampled, _ = train_test_split(labels, train_size=n, random_state=seed, stratify=labels["Label"])
    return set(sampled["BlockId"]), sampled


def extract_raw_lines(block_ids: set) -> int:
    """Streams HDFS.log once, copying only lines that mention a sampled
    block_id. A plain regex search + set lookup per line is much cheaper
    than a full HDFSParser.parse() pass over all 11.2M lines, and the parser
    itself only ever runs on the small extracted subset below."""
    matched = 0
    with RAW_LOG_PATH.open("r", encoding="utf-8", errors="replace") as src, \
         SAMPLE_RAW_LOG_PATH.open("w", encoding="utf-8") as dst:
        for line in src:
            match = BLOCK_ID_PATTERN.search(line)
            if match and match.group(0) in block_ids:
                dst.write(line if line.endswith("\n") else line + "\n")
                matched += 1
    return matched


def run_pipeline() -> pd.DataFrame:
    """Same clean -> extract-features -> predict stages sources_api.py runs
    for a real manual upload, against the currently deployed model."""
    frame = HDFSParser().parse(str(SAMPLE_RAW_LOG_PATH))
    recognized = frame[frame["date"].notna()].reset_index(drop=True)
    if len(recognized) == 0:
        raise SystemExit("No lines in the sample matched the HDFS log format.")

    cleaned = HDFSDataCleaner().clean(recognized, SAMPLE_CLEANED_PATH)
    HDFSFeatureExtractor().extract(cleaned, SAMPLE_FEATURES_PATH)
    features = pd.read_csv(SAMPLE_FEATURES_PATH)

    current = resolve_current_deployment()
    model = joblib.load(current["model_path"])
    feature_columns = list(model.feature_names_in_)

    truth = pd.read_csv(SAMPLE_TRUE_LABELS_PATH).rename(columns={"BlockId": "block_id"})
    scored = features.merge(truth, on="block_id", how="inner")
    if len(scored) != len(features):
        print(
            f"Warning: {len(features) - len(scored)} block(s) in the sample had no "
            f"ground-truth label and were dropped before scoring."
        )

    predictions = predict(model, scored, output_path=SAMPLE_PREDICTIONS_PATH, feature_columns=feature_columns)
    print(f"\nScored against deployed model v{current.get('version')} "
          f"({current.get('model_path')})")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Number of blocks to sample (default {DEFAULT_N}).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Sampling seed (default 42).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    block_ids, sampled_labels = sample_block_ids(args.n, args.seed)
    anomaly_count = int((sampled_labels["Label"] == "Anomaly").sum())
    print(f"Sampled {len(block_ids):,} blocks ({anomaly_count:,} anomalies, "
          f"{100 * anomaly_count / len(block_ids):.2f}%) with seed={args.seed}")

    sampled_labels.to_csv(SAMPLE_TRUE_LABELS_PATH, index=False)
    print(f"Wrote {SAMPLE_TRUE_LABELS_PATH} (ground truth, kept separate from the upload file)")

    print(f"\nScanning {RAW_LOG_PATH} for matching lines...")
    matched_lines = extract_raw_lines(block_ids)
    print(f"Wrote {SAMPLE_RAW_LOG_PATH} ({matched_lines:,} raw log lines, unlabeled)")

    predictions = run_pipeline()
    metrics = evaluate(predictions, metrics_path=METRICS_PATH)

    # Extra framing beyond evaluate()'s accuracy/precision/recall/F1.
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["matrix"]
    metrics["error_rate"] = 1 - metrics["accuracy"]
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0
    metrics["sample_n_requested"] = args.n
    metrics["sample_seed"] = args.seed
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n=== Verification sample results (deployed model, never seen this exact sample) ===")
    print_metrics_table(metrics)
    print(f"\nError rate:            {metrics['error_rate']:.4f}")
    print(f"False positive rate:    {metrics['false_positive_rate']:.4f}  (Normal blocks flagged as Anomaly)")
    print(f"False negative rate:    {metrics['false_negative_rate']:.4f}  (Anomaly blocks missed)")

    print(f"\nWrote {SAMPLE_PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_PATH}")
    print(f"\nRaw log for manual upload: {SAMPLE_RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
