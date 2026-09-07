"""
Sprint 5: GA-optimize the Isolation Forest baseline (see optimization/ga_isolation_forest.py
for the gene encoding and fitness function). Trains a candidate model with the best
chromosome found and evaluates it the same way as the V1 baseline for direct comparison.
"""

import json
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import DEFAULT_METRICS_PATH as V1_METRICS_PATH
from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from detection.train import load_labeled_features
from optimization.ga_isolation_forest import (
    GENE_NAMES,
    TRAINING_FEATURE_COLUMNS,
    build_model,
    run_ga,
    stratified_subsample,
)

FITNESS_SAMPLE_SIZE = 40_000
POPULATION_SIZE = 16
NUM_GENERATIONS = 12
NUM_PARENTS_MATING = 8

V2_MODEL_PATH = Path("data/models/isolation_forest_v2_candidate.pkl")
V2_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_v2_candidate_predictions.csv")
V2_METRICS_PATH = Path("data/evaluation/isolation_forest_v2_candidate_metrics.json")
GA_RUN_HISTORY_PATH = Path("data/optimization/ga_run_history.json")
GA_BEST_CONFIG_PATH = Path("data/optimization/ga_best_config.json")


def print_comparison_table(v1_metrics: dict, v2_metrics: dict) -> None:
    print(f"\n{'Metric':<12}{'V1 baseline':>14}{'V2 GA candidate':>18}{'Delta':>10}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        v1_value = v1_metrics[key]
        v2_value = v2_metrics[key]
        print(f"{key:<12}{v1_value:>14.4f}{v2_value:>18.4f}{v2_value - v1_value:>+10.4f}")


def main() -> None:
    merged = load_labeled_features()
    print(f"Loaded {len(merged):,} labeled blocks (verified join reused, not re-checked).")

    fitness_data = stratified_subsample(merged, FITNESS_SAMPLE_SIZE)
    print(
        f"Fitness evaluations use a fixed stratified subsample of "
        f"{len(fitness_data):,} rows (same subsample for every candidate this run)."
    )
    print(
        f"GA config: population_size={POPULATION_SIZE}, "
        f"num_generations={NUM_GENERATIONS}, num_parents_mating={NUM_PARENTS_MATING}"
    )

    print("\nRunning GA...")
    ga_instance, result = run_ga(
        X=fitness_data[TRAINING_FEATURE_COLUMNS],
        y=fitness_data["Label"],
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
    )
    print(
        f"GA finished in {result.wall_clock_seconds:.1f}s over "
        f"{result.generations_completed} generations. "
        f"Best fitness (F1 on Anomaly, fitness subsample): {result.best_fitness:.4f}"
    )

    best = result.best_chromosome
    print(f"Best chromosome: {best.as_readable_dict()}")

    print("\nRetraining final candidate on the full dataset with the best chromosome...")
    selected_features = best.selected_features
    model = build_model(best)
    model.fit(merged[selected_features])

    V2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, V2_MODEL_PATH)
    print(f"Saved candidate model to {V2_MODEL_PATH}")

    predictions = predict(
        model,
        merged,
        output_path=V2_PREDICTIONS_PATH,
        feature_columns=selected_features,
    )
    v2_metrics = evaluate(predictions, metrics_path=V2_METRICS_PATH)

    GA_RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GA_RUN_HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(
            {
                "gene_names": GENE_NAMES,
                "population_size": result.population_size,
                "num_generations": result.num_generations,
                "generations_completed": result.generations_completed,
                "fitness_sample_size": result.fitness_sample_size,
                "wall_clock_seconds": result.wall_clock_seconds,
                "generation_seconds": result.generation_seconds,
                "best_fitness_per_generation": result.best_fitness_per_generation,
                "best_fitness_on_fitness_subsample": result.best_fitness,
            },
            history_file,
            indent=2,
        )
    print(f"Saved GA run history to {GA_RUN_HISTORY_PATH}")

    with GA_BEST_CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "chromosome": best.as_readable_dict(),
                "threshold_gene": "contamination",
                "threshold_note": (
                    "IsolationForest.predict() derives its Normal/Anomaly decision "
                    "directly from `contamination` (via the fitted offset_), so "
                    "contamination is searched as the threshold gene instead of a "
                    "separate manual score cutoff."
                ),
                "fitness_metric": "f1_score(pos_label='Anomaly')",
                "fitness_metric_note": (
                    "Chosen over accuracy because of the 2.9% anomaly rate in this "
                    "dataset (see V1 baseline): accuracy is trivially maximized by "
                    "predicting Normal for everything. F1 on the Anomaly class "
                    "balances precision/recall for the class of interest and matches "
                    "the metric used to evaluate V1, so GA optimizes exactly what is "
                    "reported."
                ),
                "fitness_sample_size": result.fitness_sample_size,
                "best_fitness_on_fitness_subsample": result.best_fitness,
                "full_dataset_metrics": v2_metrics,
            },
            config_file,
            indent=2,
        )
    print(f"Saved winning config to {GA_BEST_CONFIG_PATH}")

    print("\n=== V2 GA candidate metrics (full dataset) ===")
    print_metrics_table(v2_metrics)

    with V1_METRICS_PATH.open("r", encoding="utf-8") as v1_file:
        v1_metrics = json.load(v1_file)
    print_comparison_table(v1_metrics, v2_metrics)


if __name__ == "__main__":
    main()
