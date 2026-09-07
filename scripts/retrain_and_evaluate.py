"""
Sprint 6: the periodic retrain-evaluate-promote loop. Run this manually for
now; scheduling comes later.

Loads the latest labeled features, runs a fresh GA optimization pass to
produce a candidate model, evaluates it, and hands the decision to
optimization.model_evolution.evaluate_and_promote() against whatever is
currently deployed (data/models/current_version.json, falling back to the
V1 baseline if this is the very first run).

Every call to this script overwrites the working "candidate" slot
(data/models/isolation_forest_candidate.pkl and friends) with the newest
attempt; evaluate_and_promote() is responsible for either promoting it or
archiving a copy under data/models/archive/ so rejected candidates aren't
silently lost.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from detection.train import load_labeled_features
from optimization.ga_isolation_forest import TRAINING_FEATURE_COLUMNS, build_model, run_ga, stratified_subsample
from optimization.model_evolution import DEFAULT_CURRENT_VERSION_PATH, evaluate_and_promote

DATA_SLICE_CHOICES = ("all", "early", "late")

FITNESS_SAMPLE_SIZE = 40_000
POPULATION_SIZE = 16
NUM_GENERATIONS = 12
NUM_PARENTS_MATING = 8

CANDIDATE_MODEL_PATH = Path("data/models/isolation_forest_candidate.pkl")
CANDIDATE_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_candidate_predictions.csv")
CANDIDATE_METRICS_PATH = Path("data/evaluation/isolation_forest_candidate_metrics.json")

DEFAULT_CURRENT_MODEL_PATH = Path("data/models/isolation_forest_v1.pkl")
DEFAULT_CURRENT_METRICS_PATH = Path("data/evaluation/baseline_isolation_forest_metrics.json")


def resolve_current_deployment() -> tuple[Path, Path]:
    """
    current_version.json (written the first time evaluate_and_promote runs)
    tracks whatever is actually deployed. Before that file exists, the V1
    baseline is the only thing that's ever been deployed.
    """
    if not DEFAULT_CURRENT_VERSION_PATH.exists():
        return DEFAULT_CURRENT_MODEL_PATH, DEFAULT_CURRENT_METRICS_PATH

    with DEFAULT_CURRENT_VERSION_PATH.open("r", encoding="utf-8") as version_file:
        current = json.load(version_file)
    return Path(current["model_path"]), Path(current["metrics_path"])


def slice_by_time(merged: pd.DataFrame, data_slice: str) -> pd.DataFrame:
    """
    features.csv rows are ordered by each block's earliest event after the
    cleaned logs are sorted by datetime (feature_engineering/feature_extractor.py
    groups block_id with sort=False, which preserves that first-appearance
    order), and load_labeled_features()'s inner join preserves the features
    frame's row order. So splitting `merged` in half by row position is a
    genuine chronological earlier/later split, not an arbitrary one.
    """
    if data_slice == "all":
        return merged
    midpoint = len(merged) // 2
    if data_slice == "early":
        return merged.iloc[:midpoint].reset_index(drop=True)
    return merged.iloc[midpoint:].reset_index(drop=True)


def train_candidate(train_data, eval_data) -> dict:
    """
    GA fitness selection and the candidate's final fit both run on
    `train_data` (the adaptation cycle's data slice); the fitted candidate is
    then scored on `eval_data` so its metrics stay comparable to whatever
    current/prior versions were evaluated on (the full dataset, by default).
    """
    fitness_sample_size = min(FITNESS_SAMPLE_SIZE, len(train_data))
    fitness_data = stratified_subsample(train_data, fitness_sample_size)
    print(f"Fitness evaluations use a stratified subsample of {len(fitness_data):,} rows.")
    print(
        f"GA config: population_size={POPULATION_SIZE}, "
        f"num_generations={NUM_GENERATIONS}, num_parents_mating={NUM_PARENTS_MATING}"
    )

    print("Running GA...")
    _, result = run_ga(
        X=fitness_data[TRAINING_FEATURE_COLUMNS],
        y=fitness_data["Label"],
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
    )
    print(
        f"GA finished in {result.wall_clock_seconds:.1f}s over "
        f"{result.generations_completed} generations, best fitness={result.best_fitness:.4f}"
    )

    best = result.best_chromosome
    selected_features = best.selected_features
    print(f"Best chromosome: {best.as_readable_dict()}")

    model = build_model(best)
    model.fit(train_data[selected_features])

    CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL_PATH)

    predictions = predict(
        model,
        eval_data,
        output_path=CANDIDATE_PREDICTIONS_PATH,
        feature_columns=selected_features,
    )
    metrics = evaluate(predictions, metrics_path=CANDIDATE_METRICS_PATH)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-slice",
        choices=DATA_SLICE_CHOICES,
        default="all",
        help=(
            "Which chronological half of the labeled blocks to use for GA "
            "fitness selection and the candidate's final fit (default: all). "
            "'early'/'late' simulate a fresh adaptation cycle running on a "
            "distinct data window; evaluation always runs against the full "
            "dataset so metrics stay comparable across versions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    current_model_path, current_metrics_path = resolve_current_deployment()
    print(f"Current deployment: {current_model_path} (metrics: {current_metrics_path})")

    merged = load_labeled_features()
    print(f"Loaded {len(merged):,} labeled blocks (verified join reused, not re-checked).")

    train_data = slice_by_time(merged, args.data_slice)
    if args.data_slice != "all":
        print(
            f"Data slice: '{args.data_slice}' half -- {len(train_data):,} blocks used for "
            f"GA fitness + candidate training; evaluation still uses the full "
            f"{len(merged):,}-block dataset for comparability with prior versions."
        )

    candidate_metrics = train_candidate(train_data, merged)
    print("\n=== Candidate metrics (full dataset) ===")
    print_metrics_table(candidate_metrics)

    decision = evaluate_and_promote(
        current_model_path=current_model_path,
        candidate_model_path=CANDIDATE_MODEL_PATH,
        current_metrics_path=current_metrics_path,
        candidate_metrics_path=CANDIDATE_METRICS_PATH,
        metric="f1",
    )

    print("\n=== Deploy decision ===")
    if decision["promoted"]:
        print(
            f"PROMOTED: v{decision['current_version']} -> v{decision['new_version']} "
            f"({decision['metric']}: {decision['current_metric_value']:.4f} -> "
            f"{decision['candidate_metric_value']:.4f})"
        )
        print(f"New current model: {decision['promoted_model_path']}")
    else:
        print(
            f"NOT PROMOTED: candidate {decision['metric']}="
            f"{decision['candidate_metric_value']:.4f} did not beat current "
            f"{decision['metric']}={decision['current_metric_value']:.4f}"
        )
        print(f"Candidate archived at: {decision['archived_candidate_path']}")


if __name__ == "__main__":
    main()
