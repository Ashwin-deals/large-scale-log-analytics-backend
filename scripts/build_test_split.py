"""Hold out a labelled test split and measure generalization.

Why this exists: the deployed model is fit on every labelled block and then
scored on those same blocks (see data/evaluation/*_metrics.json — support.total
equals the full 575,061-block dataset). That number is a *training* score, so it
cannot say how the model behaves on data it has not seen.

This script carves the feature set into train/test, writes the test split out as
two files — features without labels, and the labels separately — then trains on
train only and reports both scores so the gap is visible.

Usage:
    python3 scripts/build_test_split.py                  # 30% test split
    python3 scripts/build_test_split.py --test-size 0.2
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
from detection.isolation_forest import TRAINING_FEATURE_COLUMNS, save_model
from detection.predict import predict
from detection.train import load_labeled_features
from optimization.ga_isolation_forest import Chromosome, build_model

GA_CONFIG_PATH = Path("data/optimization/ga_best_config.json")
OUT_DIR = Path("data/testing")
TEST_FEATURES_PATH = OUT_DIR / "test_features.csv"      # unlabelled — model input
TEST_LABELS_PATH = OUT_DIR / "test_labels.csv"          # ground truth, held separately
TRAIN_FEATURES_PATH = OUT_DIR / "train_features.csv"
COMPARISON_PATH = Path("data/evaluation/train_vs_test_comparison.json")
RANDOM_STATE = 42


def load_ga_chromosome():
    """Rebuild the deployed model's GA configuration, so train/test scores are
    comparable with the number already reported for it."""
    config = json.loads(GA_CONFIG_PATH.read_text())["chromosome"]
    # The mask is positional against TRAINING_FEATURE_COLUMNS, so rebuild it in
    # that order rather than from the length of selected_features — a shorter
    # list would silently select the wrong columns.
    mask = [int(config["feature_mask"][column]) for column in TRAINING_FEATURE_COLUMNS]
    chromosome = Chromosome(
        feature_mask=mask,
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        max_features=config["max_features"],
        contamination=config["contamination"],
    )
    if chromosome.selected_features != config["selected_features"]:
        raise SystemExit(
            f"Feature mask mismatch: rebuilt {chromosome.selected_features} "
            f"but config says {config['selected_features']}"
        )
    return chromosome, chromosome.selected_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction held out (default 0.3).")
    args = parser.parse_args()

    merged = load_labeled_features()
    chromosome, feature_columns = load_ga_chromosome()

    print(f"Loaded {len(merged):,} labelled blocks "
          f"({(merged['Label'] == 'Anomaly').sum():,} anomalies, "
          f"{100 * (merged['Label'] == 'Anomaly').mean():.2f}%)")
    print(f"Model features (GA-selected): {feature_columns}")

    # Stratified so the rare Anomaly class keeps the same proportion in both
    # halves — an unstratified split can leave the test set almost anomaly-free
    # and make F1 meaningless.
    train_df, test_df = train_test_split(
        merged,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=merged["Label"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The test set as the model would receive it: identifier + features, no label.
    test_df[["block_id"] + feature_columns].to_csv(TEST_FEATURES_PATH, index=False)
    test_df[["block_id", "Label"]].to_csv(TEST_LABELS_PATH, index=False)
    train_df[["block_id"] + feature_columns].to_csv(TRAIN_FEATURES_PATH, index=False)

    print(f"\nWrote {TEST_FEATURES_PATH}  ({len(test_df):,} rows, unlabelled)")
    print(f"Wrote {TEST_LABELS_PATH}    ({len(test_df):,} labels)")
    print(f"Wrote {TRAIN_FEATURES_PATH} ({len(train_df):,} rows)")

    # Fit on the training half only. Isolation Forest is unsupervised, so the
    # labels are never passed to fit(); they are only used to score afterwards.
    model = build_model(chromosome)
    model.fit(train_df[feature_columns])
    save_model(model, OUT_DIR / "isolation_forest_trainsplit.pkl")

    train_pred = predict(model, train_df, output_path=OUT_DIR / "train_predictions.csv", feature_columns=feature_columns)
    train_metrics = evaluate(train_pred, metrics_path=Path("data/evaluation/trainsplit_train_metrics.json"))

    # Score the held-out half by joining the unlabelled features back to the
    # labels file, exactly as an external evaluator would.
    unlabelled = pd.read_csv(TEST_FEATURES_PATH)
    truth = pd.read_csv(TEST_LABELS_PATH)
    scored = unlabelled.merge(truth, on="block_id", how="inner")
    if len(scored) != len(unlabelled):
        raise SystemExit(f"Label join lost rows: {len(unlabelled)} features vs {len(scored)} joined")

    test_pred = predict(model, scored, output_path=OUT_DIR / "test_predictions.csv", feature_columns=feature_columns)
    test_metrics = evaluate(test_pred, metrics_path=Path("data/evaluation/trainsplit_test_metrics.json"))

    print("\n=== TRAIN split (data the model was fit on) ===")
    print_metrics_table(train_metrics)
    print("\n=== TEST split (held out — never seen during fit) ===")
    print_metrics_table(test_metrics)

    deployed = json.loads(GA_CONFIG_PATH.read_text())["full_dataset_metrics"]
    comparison = {
        "note": (
            "deployed_reported is the number already published for the deployed model. "
            "It is a training score: that model was fit on all labelled blocks and scored "
            "on the same blocks, so it is not comparable to a held-out result."
        ),
        "test_size": args.test_size,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_used": feature_columns,
        "train_split": {k: train_metrics[k] for k in ("accuracy", "precision", "recall", "f1")},
        "test_split": {k: test_metrics[k] for k in ("accuracy", "precision", "recall", "f1")},
        "deployed_reported_on_full_training_data": {
            k: deployed[k] for k in ("accuracy", "precision", "recall", "f1")
        },
        "generalization_gap_f1": train_metrics["f1"] - test_metrics["f1"],
    }
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))

    print("\n=== COMPARISON (F1 on the Anomaly class) ===")
    print(f"  train split (seen)      {train_metrics['f1']:.4f}")
    print(f"  test split  (held out)  {test_metrics['f1']:.4f}")
    print(f"  gap (train - test)      {comparison['generalization_gap_f1']:+.4f}")
    print(f"  deployed model, as reported on its own training data: {deployed['f1']:.4f}")
    print(f"\nWrote {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
