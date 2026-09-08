"""
Genetic Algorithm optimization of the Isolation Forest baseline (Sprint 5;
robustness pass in Sprint 7 -- see the module-level notes below each marked
"(Sprint 7)").

Gene encoding
-------------
A chromosome is a flat list of 12 genes, in this fixed order:

  [0..6]  binary feature mask, one gene per TRAINING_FEATURE_COLUMNS entry
          (block_id and is_error are never candidate genes: block_id isn't
          a feature, is_error is zero-variance in this dataset).
  [7]     n_estimators   -- discrete choice from N_ESTIMATORS_CHOICES
  [8]     max_samples    -- float in MAX_SAMPLES_RANGE, fraction of rows per tree
  [9]     max_features   -- float in MAX_FEATURES_RANGE, fraction of features per tree
  [10]    contamination  -- float in CONTAMINATION_RANGE
  [11]    bootstrap      -- binary, (Sprint 7) whether trees sample rows with
                             replacement (sklearn default is False/no-replacement)

Threshold gene: contamination (not a separate score cutoff)
-------------------------------------------------------------
IsolationForest.predict() already derives its -1/+1 decision purely from
`contamination` (it sets the internal `offset_` used against
decision_function()). Searching contamination directly optimizes exactly
the operating point sklearn uses at inference time, with no second,
redundant thresholding mechanism to keep in sync. A block is anomalous
(predicted "Anomaly") when Isolation Forest's contamination-derived
decision is -1, exactly as in the V1 baseline.

(Sprint 7) Widened max_samples / max_features
-----------------------------------------------
Both ranges were widened at their low end now that training time isn't a
constraint (see scripts/optimize_isolation_forest_ga_v3.py):
  - max_samples: 0.25 -> 0.1. The original Isolation Forest paper (Liu et
    al.) gets strong isolation with *very* small per-tree subsamples
    (sklearn's own default, max_samples="auto", is min(256, n_samples) --
    a tiny fraction on a dataset this size); 0.25 as a floor excluded that
    whole small-sample regime the algorithm is actually designed around.
  - max_features: 0.5 -> 0.3. With only 7 candidate columns, 0.3 still
    keeps at least ~2 features per split for a chromosome using all of
    them, while giving the GA room to explore more per-tree feature
    diversity for chromosomes with fewer features selected.
Both keep high=1.0 unchanged (no ceiling change requested, and 1.0 is
already sklearn's own default upper bound).

(Sprint 7) K-fold fitness evaluation
---------------------------------------
score_chromosome() (single fit+score on one fixed stratified subsample) is
still here and still what scripts/optimize_isolation_forest_ga.py and the
live /api/models/retrain endpoint use by default (k_folds=1 in
make_fitness_func/run_ga) -- unchanged speed/behavior for those existing
callers. k_fold_score_chromosome() is the new, more robust alternative:
stratified K-fold CV over the same fitness subsample, returning both the
mean F1 (used as fitness) and the std across folds. The V3 incident
documented in data/evaluation/v1_v2_v3_comparison.md is exactly the failure
mode this catches -- a chromosome that looked fine on a single fitness
subsample (0.6891) collapsed on the full dataset once its contamination
threshold, calibrated to one data slice, was applied elsewhere. A high
fold-to-fold std is a visible warning sign of that kind of instability even
when the mean alone still looks competitive; see HIGH_VARIANCE_STD_THRESHOLD.
Opt in with `k_folds` on make_fitness_func()/run_ga() (see
scripts/optimize_isolation_forest_ga_v3.py, which sets k_folds=5).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import pygad
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS

RANDOM_STATE = 42
POSITIVE_LABEL = "Anomaly"

N_ESTIMATORS_CHOICES = [50, 75, 100, 125, 150, 175, 200]
MAX_SAMPLES_RANGE = {"low": 0.1, "high": 1.0}
MAX_FEATURES_RANGE = {"low": 0.3, "high": 1.0}
CONTAMINATION_RANGE = {"low": 0.01, "high": 0.30}
BOOTSTRAP_GENE_SPACE = [0, 1]

# (Sprint 7) A fold-to-fold F1 std above this is flagged (printed, and
# recorded as `high_variance` on GARunResult / in the saved run history) as
# an unstable candidate -- not penalized in fitness itself, per the task:
# fitness still selects on mean F1 alone, this is purely a visibility flag
# a human reviewing the run can act on before deciding to promote. 0.05 is
# a judgment call, not a derived constant: F1 on this dataset's candidates
# has historically landed in the 0.15-0.70 range (see
# v1_v2_v3_comparison.md), so a std of 0.05 already represents a
# meaningfully inconsistent candidate relative to that range.
HIGH_VARIANCE_STD_THRESHOLD = 0.05

DEFAULT_K_FOLDS = 1  # 1 = score_chromosome's original single fit+score, unchanged for existing callers

N_FEATURE_GENES = len(TRAINING_FEATURE_COLUMNS)
GENE_NAMES = (
    [f"feature__{column}" for column in TRAINING_FEATURE_COLUMNS]
    + ["n_estimators", "max_samples", "max_features", "contamination", "bootstrap"]
)
GENE_SPACE = (
    [[0, 1]] * N_FEATURE_GENES
    + [N_ESTIMATORS_CHOICES, MAX_SAMPLES_RANGE, MAX_FEATURES_RANGE, CONTAMINATION_RANGE, BOOTSTRAP_GENE_SPACE]
)
GENE_TYPE = [int] * N_FEATURE_GENES + [int, float, float, float, int]


@dataclass
class Chromosome:
    feature_mask: list[int]
    n_estimators: int
    max_samples: float
    max_features: float
    contamination: float
    bootstrap: bool = False

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
            "bootstrap": self.bootstrap,
        }


def decode_chromosome(genes) -> Chromosome:
    genes = list(genes)
    feature_mask = [int(round(gene)) for gene in genes[:N_FEATURE_GENES]]
    n_estimators = int(round(genes[N_FEATURE_GENES]))
    max_samples = float(np.clip(genes[N_FEATURE_GENES + 1], 0.05, 1.0))
    max_features = float(np.clip(genes[N_FEATURE_GENES + 2], 0.05, 1.0))
    contamination = float(np.clip(genes[N_FEATURE_GENES + 3], 0.001, 0.5))
    bootstrap = bool(int(round(genes[N_FEATURE_GENES + 4]))) if len(genes) > N_FEATURE_GENES + 4 else False
    return Chromosome(
        feature_mask=feature_mask,
        n_estimators=max(1, n_estimators),
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        bootstrap=bootstrap,
    )


def build_model(chromosome: Chromosome, random_state: int = RANDOM_STATE) -> IsolationForest:
    """Build an Isolation Forest for this chromosome.

    random_state is a parameter rather than a constant so a retrain can explore
    a different corner of the search space. Pinned to RANDOM_STATE it reproduces
    the same model from the same data every time, which is what made every
    retrain return a bit-identical candidate that could never beat the current
    model (see the repeated "0.6788 did not beat 0.6788" history entries).
    """
    return IsolationForest(
        n_estimators=chromosome.n_estimators,
        max_samples=chromosome.max_samples,
        max_features=chromosome.max_features,
        contamination=chromosome.contamination,
        bootstrap=chromosome.bootstrap,
        n_jobs=-1,
        random_state=random_state,
    )


def _fit_and_score(
    chromosome: Chromosome,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    random_state: int = RANDOM_STATE,
) -> float:
    """
    Fits on X_train, scores with F1(Anomaly) on X_eval/y_eval. X_train and
    X_eval are the same frame for score_chromosome's in-sample check;
    distinct train/held-out folds for k_fold_score_chromosome.

    F1(Anomaly), not accuracy, given the 2.9% anomaly rate observed in the
    V1 baseline: a model that predicts "Normal" for everything scores ~97%
    accuracy while catching zero anomalies. F1 on the positive class
    balances precision and recall for the class that actually matters, and
    matches the metric already used to evaluate V1, so the GA optimizes
    exactly what gets reported and compared.
    """
    selected = chromosome.selected_features
    model = build_model(chromosome, random_state=random_state)
    model.fit(X_train[selected])
    raw_predictions = model.predict(X_eval[selected])
    predicted_label = np.where(raw_predictions == -1, "Anomaly", "Normal")
    return float(f1_score(y_eval, predicted_label, pos_label=POSITIVE_LABEL, zero_division=0))


def score_chromosome(chromosome: Chromosome, X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE) -> float:
    """
    Original single fit+score: trains on X, scores (in-sample) on the same
    X/y. Still what the live retrain endpoint and the repeatable adaptation
    cycle use (k_folds=1 in make_fitness_func/run_ga) -- see
    k_fold_score_chromosome for the more robust, opt-in alternative.
    """
    if sum(chromosome.feature_mask) == 0:
        return 0.0
    return _fit_and_score(chromosome, X, X, y, random_state=random_state)


@dataclass
class FoldScore:
    mean: float
    std: float
    fold_scores: list[float] = field(default_factory=list)


def k_fold_score_chromosome(
    chromosome: Chromosome,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    random_state: int = RANDOM_STATE,
) -> FoldScore:
    """
    Stratified k-fold CV over (X, y): for each fold, fits on the other k-1
    folds (unsupervised -- y is only used to define the stratified split and
    to score the held-out fold) and scores F1(Anomaly) on the held-out fold.
    Returns the mean of those k scores (used as fitness) and their std,
    which is what surfaces a chromosome that's strong on some slices of the
    fitness subsample and weak on others -- the V3 failure mode (see the
    module docstring) that a single fit+score average can't reveal.
    """
    if sum(chromosome.feature_mask) == 0:
        return FoldScore(mean=0.0, std=0.0, fold_scores=[0.0] * k)

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_scores = [
        _fit_and_score(chromosome, X.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx], random_state=random_state)
        for train_idx, test_idx in splitter.split(X, y)
    ]
    return FoldScore(
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        fold_scores=fold_scores,
    )


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


def _chromosome_cache_key(chromosome: Chromosome) -> tuple:
    """Hashable identity for a decoded chromosome, used to look a FoldScore
    back up in make_fitness_func's score_cache without recomputing it."""
    return (
        tuple(chromosome.feature_mask),
        chromosome.n_estimators,
        round(chromosome.max_samples, 6),
        round(chromosome.max_features, 6),
        round(chromosome.contamination, 6),
        chromosome.bootstrap,
    )


def make_fitness_func(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: int = DEFAULT_K_FOLDS,
    random_state: int = RANDOM_STATE,
    score_cache: dict | None = None,
) -> Callable:
    """
    k_folds=1 (default) reproduces score_chromosome's original single
    fit+score -- unchanged behavior/speed for the live retrain endpoint and
    the repeatable adaptation cycle, neither of which pass k_folds. k_folds>1
    runs real stratified k-fold CV (k_fold_score_chromosome) -- what
    scripts/optimize_isolation_forest_ga_v3.py opts into.

    score_cache, when given, is filled with every evaluated chromosome's
    full FoldScore (mean/std/per-fold), keyed by _chromosome_cache_key, so
    run_ga's on_generation callback can log the current best candidate's std
    without re-scoring it -- pygad's fitness_func can only return a single
    number, so the std has to be recovered from here rather than from
    ga_instance.best_solution() directly.
    """
    if score_cache is None:
        score_cache = {}

    def fitness_func(ga_instance, solution, solution_idx):
        chromosome = decode_chromosome(solution)
        key = _chromosome_cache_key(chromosome)
        # keep_elitism carries the same top individual forward unchanged
        # across generations; fitness is deterministic given (chromosome, X,
        # y, k_folds, random_state), so re-scoring an exact repeat -- most
        # commonly that elite -- is pure waste, and more so at k_folds>1
        # where each score is k separate fits.
        cached = score_cache.get(key)
        if cached is not None:
            return cached.mean

        if k_folds <= 1:
            fitness = score_chromosome(chromosome, X, y, random_state=random_state)
            score = FoldScore(mean=fitness, std=0.0, fold_scores=[fitness])
        else:
            score = k_fold_score_chromosome(chromosome, X, y, k=k_folds, random_state=random_state)
        score_cache[key] = score
        return score.mean

    return fitness_func


@dataclass
class GARunResult:
    best_chromosome: Chromosome
    best_fitness: float
    generations_completed: int
    best_fitness_std: float = 0.0
    best_fold_scores: list[float] = field(default_factory=list)
    best_fitness_per_generation: list[float] = field(default_factory=list)
    best_fitness_std_per_generation: list[float] = field(default_factory=list)
    generation_seconds: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    population_size: int = 0
    num_generations: int = 0
    num_parents_mating: int = 0
    fitness_sample_size: int = 0
    k_folds: int = 1
    high_variance: bool = False


def run_ga(
    X: pd.DataFrame,
    y: pd.Series,
    population_size: int,
    num_generations: int,
    num_parents_mating: int,
    k_folds: int = DEFAULT_K_FOLDS,
    random_seed: int = RANDOM_STATE,
) -> tuple[pygad.GA, GARunResult]:
    score_cache: dict[tuple, FoldScore] = {}
    fitness_func = make_fitness_func(X, y, k_folds=k_folds, random_state=random_seed, score_cache=score_cache)

    best_fitness_per_generation: list[float] = []
    best_fitness_std_per_generation: list[float] = []
    generation_seconds: list[float] = []
    generation_start = {"t": time.perf_counter()}

    def on_generation(ga_instance):
        now = time.perf_counter()
        generation_seconds.append(now - generation_start["t"])
        generation_start["t"] = now

        best_solution, best_fitness, _ = ga_instance.best_solution()
        best_fitness_per_generation.append(float(best_fitness))

        best_score = score_cache.get(_chromosome_cache_key(decode_chromosome(best_solution)))
        best_std = best_score.std if best_score is not None else 0.0
        best_fitness_std_per_generation.append(best_std)

        flag = "  <- HIGH FOLD VARIANCE" if best_std > HIGH_VARIANCE_STD_THRESHOLD else ""
        print(
            f"  generation {ga_instance.generations_completed:>2}/{num_generations} "
            f"best_fitness_mean={best_fitness:.4f} best_fitness_std={best_std:.4f}{flag} "
            f"({generation_seconds[-1]:.1f}s)"
        )

    start = time.perf_counter()
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=population_size,
        num_genes=len(GENE_SPACE),
        # pygad mutates gene_space/gene_type in place (e.g. normalizes a bare
        # `float` entry into `[float, None]`), which corrupts these
        # module-level constants for any later run_ga() call in the same
        # process -- confirmed empirically: a second call reusing the
        # already-mutated GENE_TYPE fails pygad's own re-validation ("the
        # precision for float gene data types must be an integer but ...
        # has a precision of None"). Pass copies so pygad only ever mutates
        # a throwaway.
        gene_space=copy.deepcopy(GENE_SPACE),
        gene_type=copy.deepcopy(GENE_TYPE),
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
    best_score = score_cache.get(_chromosome_cache_key(best_chromosome))
    best_std = best_score.std if best_score is not None else 0.0
    best_fold_scores = best_score.fold_scores if best_score is not None else [float(best_fitness)]

    result = GARunResult(
        best_chromosome=best_chromosome,
        best_fitness=float(best_fitness),
        generations_completed=int(ga_instance.generations_completed),
        best_fitness_std=best_std,
        best_fold_scores=best_fold_scores,
        best_fitness_per_generation=best_fitness_per_generation,
        best_fitness_std_per_generation=best_fitness_std_per_generation,
        generation_seconds=generation_seconds,
        wall_clock_seconds=elapsed,
        population_size=population_size,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_sample_size=len(X),
        k_folds=k_folds,
        high_variance=best_std > HIGH_VARIANCE_STD_THRESHOLD,
    )
    return ga_instance, result
