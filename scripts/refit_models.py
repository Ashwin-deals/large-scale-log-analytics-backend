"""Refit the deployed models against the current feature encoding.

Models saved before stable category encoding carry no encoder fingerprint, and
their learned thresholds refer to codes assigned by whichever slice of the log
was used to build features at the time. Applied to features built from any
other slice they produce confident nonsense — this is what made the dashboard
report ~49% of blocks as anomalous.

This refits V1 (baseline) and V2 (the GA-optimised configuration) on the
current features, regenerates their metrics, and stamps both with the
vocabulary fingerprint so the mismatch can never go unnoticed again. The
lineage and hyperparameters are unchanged; only the fit is redone.

Run it after any features rebuild that changes the category vocabulary.

Usage:
    python3 scripts/refit_models.py
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import evaluate, print_metrics_table
from detection.isolation_forest import (
    TRAINING_FEATURE_COLUMNS,
    build_isolation_forest,
    save_model,
)
from detection.predict import predict
from detection.train import load_labeled_features
from feature_engineering.category_encoder import CategoryEncoder
from optimization.ga_isolation_forest import Chromosome, build_model

GA_CONFIG_PATH = Path("data/optimization/ga_best_config.json")
CURRENT_VERSION_PATH = Path("data/models/current_version.json")

V1_MODEL_PATH = Path("data/models/isolation_forest_v1.pkl")
V1_METRICS_PATH = Path("data/evaluation/baseline_isolation_forest_metrics.json")
V2_MODEL_PATH = Path("data/models/isolation_forest_v2.pkl")
V2_METRICS_PATH = Path("data/evaluation/isolation_forest_v2_metrics.json")


def ga_chromosome():
    config = json.loads(GA_CONFIG_PATH.read_text())["chromosome"]
    mask = [int(config["feature_mask"][column]) for column in TRAINING_FEATURE_COLUMNS]
    chromosome = Chromosome(
        feature_mask=mask,
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        max_features=config["max_features"],
        contamination=config["contamination"],
    )
    if chromosome.selected_features != config["selected_features"]:
        raise SystemExit("GA feature mask does not match selected_features in the config.")
    return chromosome


def main():
    merged = load_labeled_features()
    fingerprint = CategoryEncoder.load().fingerprint()
    print(f"Refitting on {len(merged):,} blocks — vocabulary {fingerprint}\n")

    # --- V1: baseline, all features, default hyperparameters -----------------
    v1 = build_isolation_forest()
    v1.fit(merged[TRAINING_FEATURE_COLUMNS])
    save_model(v1, V1_MODEL_PATH)
    v1_pred = predict(
        v1, merged,
        output_path="data/predictions/isolation_forest_v1_predictions.csv",
        feature_columns=TRAINING_FEATURE_COLUMNS,
    )
    v1_metrics = evaluate(v1_pred, metrics_path=V1_METRICS_PATH)
    print("=== V1 (baseline) ===")
    print_metrics_table(v1_metrics)

    # --- V2: the GA-selected features and hyperparameters --------------------
    chromosome = ga_chromosome()
    features = chromosome.selected_features
    v2 = build_model(chromosome)
    v2.fit(merged[features])
    save_model(v2, V2_MODEL_PATH)
    v2_pred = predict(
        v2, merged,
        output_path="data/predictions/isolation_forest_v2_predictions.csv",
        feature_columns=features,
    )
    v2_metrics = evaluate(v2_pred, metrics_path=V2_METRICS_PATH)
    print("\n=== V2 (GA-optimised) ===")
    print_metrics_table(v2_metrics)

    # Keep the deployment pointer's recorded metric in step with the refit, or
    # the promotion check would compare candidates against a stale number.
    current = json.loads(CURRENT_VERSION_PATH.read_text())
    current["metric_value"] = v2_metrics[current.get("metric", "f1")]
    CURRENT_VERSION_PATH.write_text(json.dumps(current, indent=2))

    print(f"\nBoth models stamped with vocabulary {fingerprint}.")
    print(f"Updated {CURRENT_VERSION_PATH} metric_value -> {current['metric_value']:.4f}")
    print("\nNOTE: these scores describe the currently-built feature slice. Rebuild "
          "features over more of the log and re-run this script to refresh them.")


if __name__ == "__main__":
    main()
