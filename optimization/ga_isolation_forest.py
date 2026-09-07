"""
Genetic Algorithm optimization of the Isolation Forest baseline (Sprint 5).

Gene encoding
-------------
A chromosome is a flat list of 11 genes, in this fixed order:

  [0..6]  binary feature mask, one gene per TRAINING_FEATURE_COLUMNS entry
          (block_id and is_error are never candidate genes: block_id isn't
          a feature, is_error is zero-variance in this dataset).
  [7]     n_estimators   -- discrete choice from N_ESTIMATORS_CHOICES
  [8]     max_samples    -- float in MAX_SAMPLES_RANGE, fraction of rows per tree
  [9]     max_features   -- float in MAX_FEATURES_RANGE, fraction of features per tree
  [10]    contamination  -- float in CONTAMINATION_RANGE

Threshold gene: contamination (not a separate score cutoff)
-------------------------------------------------------------
IsolationForest.predict() already derives its -1/+1 decision purely from
`contamination` (it sets the internal `offset_` used against
decision_function()). Searching contamination directly optimizes exactly
the operating point sklearn uses at inference time, with no second,
redundant thresholding mechanism to keep in sync. A block is anomalous
(predicted "Anomaly") when Isolation Forest's contamination-derived
decision is -1, exactly as in the V1 baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import pygad
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS

RANDOM_STATE = 42
POSITIVE_LABEL = "Anomaly"

N_ESTIMATORS_CHOICES = [50, 75, 100, 125, 150, 175, 200]
MAX_SAMPLES_RANGE = {"low": 0.25, "high": 1.0}
MAX_FEATURES_RANGE = {"low": 0.5, "high": 1.0}
CONTAMINATION_RANGE = {"low": 0.01, "high": 0.30}

N_FEATURE_GENES = len(TRAINING_FEATURE_COLUMNS)
GENE_NAMES = (
    [f"feature__{column}" for column in TRAINING_FEATURE_COLUMNS]
    + ["n_estimators", "max_samples", "max_features", "contamination"]
)
GENE_SPACE = (
    [[0, 1]] * N_FEATURE_GENES
    + [N_ESTIMATORS_CHOICES, MAX_SAMPLES_RANGE, MAX_FEATURES_RANGE, CONTAMINATION_RANGE]
)
GENE_TYPE = [int] * N_FEATURE_GENES + [int, float, float, float]


@dataclass
class Chromosome:
    feature_mask: list[int]
    n_estimators: int
    max_samples: float
    max_features: float
    contamination: float

    @property
    def selected_features(self) -> list[str]:
        return [
            column
            for column, on in zip(TRAINING_FEATURE_COLUMNS, self.feature_mask)
            if on
        ]

    def as_readable_dict(self) -> dict:
        return {
            "selected_features": self.selected_features,
            "feature_mask": dict(zip(TRAINING_FEATURE_COLUMNS, self.feature_mask)),
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "contamination": self.contamination,
        }


def decode_chromosome(genes) -> Chromosome:
    genes = list(genes)
    feature_mask = [int(round(gene)) for gene in genes[:N_FEATURE_GENES]]
    n_estimators = int(round(genes[N_FEATURE_GENES]))
    max_samples = float(np.clip(genes[N_FEATURE_GENES + 1], 0.05, 1.0))
    max_features = float(np.clip(genes[N_FEATURE_GENES + 2], 0.05, 1.0))
    contamination = float(np.clip(genes[N_FEATURE_GENES + 3], 0.001, 0.5))
    return Chromosome(
        feature_mask=feature_mask,
        n_estimators=max(1, n_estimators),
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
    )


def build_model(chromosome: Chromosome) -> IsolationForest:
    return IsolationForest(
        n_estimators=chromosome.n_estimators,
        max_samples=chromosome.max_samples,
        max_features=chromosome.max_features,
        contamination=chromosome.contamination,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def score_chromosome(chromosome: Chromosome, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Trains an Isolation Forest with the chromosome's feature subset and
    hyperparameters (unsupervised: y is only used to score the result, never
    passed to fit()), then scores it with F1 on the Anomaly class.

    F1(Anomaly), not accuracy, given the 2.9% anomaly rate observed in the
    V1 baseline: a model that predicts "Normal" for everything scores ~97%
    accuracy while catching zero anomalies. F1 on the positive class
    balances precision and recall for the class that actually matters, and
    matches the metric already used to evaluate V1, so the GA optimizes
    exactly what gets reported and compared.
    """
    if sum(chromosome.feature_mask) == 0:
        return 0.0

    selected = chromosome.selected_features
    model = build_model(chromosome)
    model.fit(X[selected])
    raw_predictions = model.predict(X[selected])
    predicted_label = np.where(raw_predictions == -1, "Anomaly", "Normal")
    return float(f1_score(y, predicted_label, pos_label=POSITIVE_LABEL, zero_division=0))


def stratified_subsample(
    merged: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draws a fixed, label-stratified subsample used for every fitness
    evaluation in a GA run, so each candidate is scored on the same data
    (a stable fitness landscape) and fitness evaluation stays fast even
    though the full dataset has 575k+ rows.
    """
    if sample_size >= len(merged):
        return merged.reset_index(drop=True)
    subsample, _ = train_test_split(
        merged,
        train_size=sample_size,
        stratify=merged["Label"],
        random_state=random_state,
    )
    return subsample.reset_index(drop=True)


def make_fitness_func(X: pd.DataFrame, y: pd.Series) -> Callable:
    def fitness_func(ga_instance, solution, solution_idx):
        chromosome = decode_chromosome(solution)
        return score_chromosome(chromosome, X, y)

    return fitness_func


@dataclass
class GARunResult:
    best_chromosome: Chromosome
    best_fitness: float
    generations_completed: int
    best_fitness_per_generation: list[float] = field(default_factory=list)
    generation_seconds: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    population_size: int = 0
    num_generations: int = 0
    fitness_sample_size: int = 0


def run_ga(
    X: pd.DataFrame,
    y: pd.Series,
    population_size: int,
    num_generations: int,
    num_parents_mating: int,
    random_seed: int = RANDOM_STATE,
) -> tuple[pygad.GA, GARunResult]:
    fitness_func = make_fitness_func(X, y)

    best_fitness_per_generation: list[float] = []
    generation_seconds: list[float] = []
    generation_start = {"t": time.perf_counter()}

    def on_generation(ga_instance):
        now = time.perf_counter()
        generation_seconds.append(now - generation_start["t"])
        generation_start["t"] = now
        _, best_fitness, _ = ga_instance.best_solution()
        best_fitness_per_generation.append(float(best_fitness))
        print(
            f"  generation {ga_instance.generations_completed:>2}/{num_generations} "
            f"best_fitness={best_fitness:.4f} "
            f"({generation_seconds[-1]:.1f}s)"
        )

    start = time.perf_counter()
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=population_size,
        num_genes=len(GENE_SPACE),
        gene_space=GENE_SPACE,
        gene_type=GENE_TYPE,
        fitness_func=fitness_func,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        keep_elitism=1,
        on_generation=on_generation,
        random_seed=random_seed,
        suppress_warnings=True,
    )
    ga_instance.run()
    elapsed = time.perf_counter() - start

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_chromosome = decode_chromosome(best_solution)

    result = GARunResult(
        best_chromosome=best_chromosome,
        best_fitness=float(best_fitness),
        generations_completed=int(ga_instance.generations_completed),
        best_fitness_per_generation=best_fitness_per_generation,
        generation_seconds=generation_seconds,
        wall_clock_seconds=elapsed,
        population_size=population_size,
        num_generations=num_generations,
        fitness_sample_size=len(X),
    )
    return ga_instance, result
