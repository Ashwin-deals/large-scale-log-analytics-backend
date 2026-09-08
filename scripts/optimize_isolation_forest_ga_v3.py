"""
GA optimization robustness pass -- produces the V3 candidate.

Builds on scripts/optimize_isolation_forest_ga.py (which produced V2), with
three changes motivated by the V3 chronological-slice failure documented in
data/evaluation/v1_v2_v3_comparison.md (a candidate that scored fine on its
own fitness subsample collapsed once scored on the full dataset):

  1. Fitness is now stratified k-fold CV (K_FOLDS=5) over a single fixed
     subsample, not one fit + score on that subsample. Every candidate's
     mean F1 across folds is used as fitness, same as before; the std
     across folds is also recorded and flagged (not penalized in fitness --
     see optimization.ga_isolation_forest.HIGH_VARIANCE_STD_THRESHOLD) when
     it's high, since that's exactly the kind of instability that let V3
     look fine in search and fail in production.
  2. The search space is wider: added `bootstrap` as a gene, and widened
     max_samples/max_features (see optimization/ga_isolation_forest.py's
     module docstring for the numeric rationale).
  3. population_size and num_generations are both scaled up, now that wall
     clock time isn't a constraint -- one generation is timed first (at the
     real population size) and an estimate for the full run is printed
     before committing to it, same discipline used to originally size
     FITNESS_SAMPLE_SIZE/population/generations for V1/V2.

Same output convention as optimize_isolation_forest_ga.py's V2 run, one
version number up: data/models/isolation_forest_v3_candidate.pkl,
data/predictions/isolation_forest_v3_candidate_predictions.csv,
data/evaluation/isolation_forest_v3_candidate_metrics.json. GA run
history/best config are versioned too (ga_run_history_v3.json,
ga_best_config_v3.json) rather than overwriting the V2 run's files.

Does NOT call evaluate_and_promote() -- this only produces the candidate and
its metrics, exactly like the V2 script did. Promotion is a separate,
deliberate step (scripts/retrain_and_evaluate.py, or calling
optimization.model_evolution.evaluate_and_promote() directly).
"""

import json
import sys
import time
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
    HIGH_VARIANCE_STD_THRESHOLD,
    TRAINING_FEATURE_COLUMNS,
    build_model,
    run_ga,
    stratified_subsample,
)

# --- Fitness evaluation ---------------------------------------------------
# 100k rows, vs V2's 40k: each candidate is now scored across K_FOLDS=5
# folds rather than a single fit+score, so the effective per-fold training
# set (~80% of 100k = 80k rows) stays comfortably larger than V2's entire
# 40k fitness sample, while the held-out fold (~20k rows, ~580 Anomaly
# examples at the dataset's ~2.9% rate) is large enough for a low-noise F1
# estimate rather than a handful of examples swinging the score.
FITNESS_SAMPLE_SIZE = 100_000
K_FOLDS = 5

# --- Search scale ----------------------------------------------------------
# V2 used population_size=16, num_generations=12 (192 candidate
# evaluations, each a single fit on 40k rows) and finished in a few
# minutes. k_folds=5 on a 100k-row sample makes each candidate evaluation
# roughly an order of magnitude more expensive (5 fits instead of 1, ~2.5x
# more rows per fit) -- confirmed empirically: a clean, uncontended
# single-generation timing at population_size=24 took 196.8s, ~8.2s per
# candidate. An initial guess of population_size=64/num_generations=40
# (2560 evaluations) was rejected on that basis -- at ~8.2s/candidate that
# is several hours, which "time isn't a constraint" doesn't require reaching
# for. population_size=32/num_generations=20 (640 evaluations, ~3.3x V2's
# total) was the first real run's target -- still a meaningfully larger and
# much more robust search than V2's, sized to actually finish.
#
# NUM_GENERATIONS reduced 20 -> 10 after that run (interrupted at generation
# 10/20, not a code failure): fitness had already plateaued at 0.6426 from
# generation 5 through generation 10 with no further improvement, and fold
# std stayed low/stable throughout (no high-variance flags) -- so the
# second half of that run wasn't earning anything. 10 generations covers
# the observed plateau point (5) with a safety margin. Everything else
# (population size, k-fold config, gene encoding, search ranges) is
# unchanged from that run. random_seed is fixed (RANDOM_STATE), so this
# restart-from-scratch reproduces generations 1-10 of the interrupted run
# exactly, then stops instead of continuing to 20.
#
# NUM_PARENTS_MATING keeps V2's ~50%-of-population ratio.
# time_one_generation() below re-measures the real cost at these exact
# settings and prints an estimate before the full run actually starts, per
# the same discipline used to size these numbers.
POPULATION_SIZE = 32
NUM_GENERATIONS = 10
NUM_PARENTS_MATING = 16

V3_MODEL_PATH = Path("data/models/isolation_forest_v3_candidate.pkl")
V3_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_v3_candidate_predictions.csv")
V3_METRICS_PATH = Path("data/evaluation/isolation_forest_v3_candidate_metrics.json")
GA_RUN_HISTORY_PATH = Path("data/optimization/ga_run_history_v3.json")
GA_BEST_CONFIG_PATH = Path("data/optimization/ga_best_config_v3.json")

V2_METRICS_PATH = Path("data/evaluation/isolation_forest_v2_metrics.json")


def time_one_generation(X, y) -> float:
    """
    Runs a single generation (num_generations=1) at the real
    population_size/k_folds/fitness_sample_size, so the full run's cost
    (population_size x num_generations x k_folds candidate-fold fits) can be
    estimated and printed before committing to it, rather than guessing.
    """
    print(
        f"Timing one generation at population_size={POPULATION_SIZE}, "
        f"k_folds={K_FOLDS}, fitness_sample_size={len(X):,}..."
    )
    start = time.perf_counter()
    run_ga(
        X=X,
        y=y,
        population_size=POPULATION_SIZE,
        num_generations=1,
        num_parents_mating=NUM_PARENTS_MATING,
        k_folds=K_FOLDS,
    )
    elapsed = time.perf_counter() - start
    print(f"One generation took {elapsed:.1f}s.")
    return elapsed


def print_comparison_table(v1_metrics: dict, v2_metrics: dict | None, v3_metrics: dict) -> None:
    rows = [("V1", v1_metrics)]
    if v2_metrics is not None:
        rows.append(("V2", v2_metrics))
    rows.append(("V3", v3_metrics))

    header = f"\n{'Metric':<12}" + "".join(f"{label:>14}" for label, _ in rows)
    print(header)
    for key in ["accuracy", "precision", "recall", "f1"]:
        line = f"{key:<12}"
        for _, metrics in rows:
            line += f"{metrics[key]:>14.4f}"
        print(line)


def main() -> None:
    merged = load_labeled_features()
    print(f"Loaded {len(merged):,} labeled blocks (verified join reused, not re-checked).")

    fitness_data = stratified_subsample(merged, FITNESS_SAMPLE_SIZE)
    print(
        f"Fitness evaluations use a fixed stratified subsample of {len(fitness_data):,} rows, "
        f"scored with {K_FOLDS}-fold stratified cross-validation per candidate "
        f"(mean F1 across folds = fitness; std across folds is logged and flagged, not penalized)."
    )

    X = fitness_data[TRAINING_FEATURE_COLUMNS]
    y = fitness_data["Label"]

    one_gen_seconds = time_one_generation(X, y)
    estimated_total_seconds = one_gen_seconds * NUM_GENERATIONS
    print(
        f"Estimated total GA time: ~{estimated_total_seconds / 60:.1f} min "
        f"({NUM_GENERATIONS} generations x ~{one_gen_seconds:.1f}s/generation)."
    )

    print(
        f"\nGA config: population_size={POPULATION_SIZE}, num_generations={NUM_GENERATIONS}, "
        f"num_parents_mating={NUM_PARENTS_MATING}, k_folds={K_FOLDS}"
    )
    print("Running full GA search...")
    ga_instance, result = run_ga(
        X=X,
        y=y,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        k_folds=K_FOLDS,
    )
    print(
        f"GA finished in {result.wall_clock_seconds:.1f}s over {result.generations_completed} generations. "
        f"Best fitness (mean F1 across {K_FOLDS} folds): {result.best_fitness:.4f} "
        f"(std={result.best_fitness_std:.4f}{', HIGH VARIANCE' if result.high_variance else ''})"
    )

    best = result.best_chromosome
    print(f"Best chromosome: {best.as_readable_dict()}")

    print("\nRetraining final candidate on the full dataset with the best chromosome...")
    selected_features = best.selected_features
    model = build_model(best)
    model.fit(merged[selected_features])

    V3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, V3_MODEL_PATH)
    print(f"Saved candidate model to {V3_MODEL_PATH}")

    predictions = predict(
        model,
        merged,
        output_path=V3_PREDICTIONS_PATH,
        feature_columns=selected_features,
    )
    v3_metrics = evaluate(predictions, metrics_path=V3_METRICS_PATH)

    GA_RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GA_RUN_HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(
            {
                "gene_names": GENE_NAMES,
                "k_folds": K_FOLDS,
                "population_size": result.population_size,
                "num_generations": result.num_generations,
                "num_parents_mating": result.num_parents_mating,
                "generations_completed": result.generations_completed,
                "fitness_sample_size": result.fitness_sample_size,
                "one_generation_timing_probe_seconds": one_gen_seconds,
                "estimated_total_seconds_before_run": estimated_total_seconds,
                "wall_clock_seconds": result.wall_clock_seconds,
                "generation_seconds": result.generation_seconds,
                "best_fitness_mean_per_generation": result.best_fitness_per_generation,
                "best_fitness_std_per_generation": result.best_fitness_std_per_generation,
                "best_fitness_mean_on_fitness_subsample": result.best_fitness,
                "best_fitness_std_on_fitness_subsample": result.best_fitness_std,
                "best_fold_scores_on_fitness_subsample": result.best_fold_scores,
                "high_variance_std_threshold": HIGH_VARIANCE_STD_THRESHOLD,
                "high_variance_flagged": result.high_variance,
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
                "fitness_metric": f"mean(f1_score(pos_label='Anomaly')) across {K_FOLDS} stratified folds",
                "fitness_metric_note": (
                    "V1/V2 scored each candidate with a single fit+score on one "
                    "stratified subsample. That let a V3 attempt (see "
                    "data/evaluation/v1_v2_v3_comparison.md) look fine on its own "
                    "fitness sample (0.6891) while collapsing on the full dataset -- "
                    "its contamination threshold was calibrated to one data slice and "
                    "badly miscalibrated everywhere else. k-fold CV scores each "
                    "candidate on k held-out slices of the same fitness sample and "
                    "reports both the mean (used as fitness) and the std across "
                    "folds, so an unstable candidate is visible even when its mean "
                    "still looks competitive."
                ),
                "fitness_sample_size": result.fitness_sample_size,
                "k_folds": K_FOLDS,
                "best_fitness_mean_on_fitness_subsample": result.best_fitness,
                "best_fitness_std_on_fitness_subsample": result.best_fitness_std,
                "best_fold_scores_on_fitness_subsample": result.best_fold_scores,
                "high_variance_std_threshold": HIGH_VARIANCE_STD_THRESHOLD,
                "high_variance_flagged": result.high_variance,
                "full_dataset_metrics": v3_metrics,
            },
            config_file,
            indent=2,
        )
    print(f"Saved winning config to {GA_BEST_CONFIG_PATH}")

    print("\n=== V3 GA candidate metrics (full dataset) ===")
    print_metrics_table(v3_metrics)

    with V1_METRICS_PATH.open("r", encoding="utf-8") as v1_file:
        v1_metrics = json.load(v1_file)

    v2_metrics = None
    if V2_METRICS_PATH.exists():
        with V2_METRICS_PATH.open("r", encoding="utf-8") as v2_file:
            v2_metrics = json.load(v2_file)

    print_comparison_table(v1_metrics, v2_metrics, v3_metrics)

    print(
        "\nIn-search estimate vs. actual full-dataset result: "
        f"fold mean F1={result.best_fitness:.4f} (std={result.best_fitness_std:.4f}) "
        f"on the {result.fitness_sample_size:,}-row fitness subsample "
        f"-> full-dataset F1={v3_metrics['f1']:.4f} "
        f"(delta={v3_metrics['f1'] - result.best_fitness:+.4f})"
    )

    print(
        "\nThis candidate was NOT auto-promoted (by design). To evaluate promotion "
        "against the currently deployed model, run scripts/retrain_and_evaluate.py "
        "or call optimization.model_evolution.evaluate_and_promote() directly with "
        f"candidate_model_path={V3_MODEL_PATH} and candidate_metrics_path={V3_METRICS_PATH}."
    )


if __name__ == "__main__":
    main()
