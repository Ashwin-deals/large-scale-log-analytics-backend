"""PyGAD search for the storage policy, plus the tune/holdout protocol and the
report that compares the tuned policy against the baselines.

Gene encoding
-------------
A chromosome is a flat list of 6 floats, in this fixed order:

  [0..2]  weight_anomaly_score, weight_event_frequency, weight_block_size
          -- each in [0, 1]; rescaled to sum to 1 when decoded
  [3]     t_hot          -- in [T_HOT_RANGE]
  [4]     t_warm_ratio   -- t_warm = ratio * t_hot
  [5]     summary_fraction (f)

t_warm is searched as a fraction of t_hot rather than as a free gene so every
chromosome satisfies t_warm < t_hot by construction. Searching both freely
wastes about half the space on invalid orderings, and repairing them (swap, or
clip) makes many genotypes decode to the same policy.

Same conventions as optimization/ga_isolation_forest.py: gene_space/gene_type
are deep-copied before they reach pygad (pygad mutates them in place), and the
random seed is threaded through as a parameter.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pygad
from sklearn.model_selection import train_test_split

from . import baselines
from .policy import (
    ERROR_WEIGHT,
    PENALTY_WEIGHT,
    RETENTION_TARGET,
    WARM_CREDIT,
    BlockTable,
    Normalizer,
    StorageOutcome,
    StoragePolicy,
    WARM,
    assign_tiers,
    evaluate_policy,
    importance,
    penalized_fitness,
)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.5

T_HOT_RANGE = {"low": 0.01, "high": 1.0}
T_WARM_RATIO_RANGE = {"low": 0.0, "high": 1.0}
SUMMARY_FRACTION_RANGE = {"low": 0.01, "high": 0.5}
MAX_T_WARM_RATIO = 1.0 - 1e-6

GENE_NAMES = [
    "weight_anomaly_score", "weight_event_frequency", "weight_block_size",
    "t_hot", "t_warm_ratio", "summary_fraction",
]
GENE_SPACE = [{"low": 0.0, "high": 1.0}] * 3 + [T_HOT_RANGE, T_WARM_RATIO_RANGE, SUMMARY_FRACTION_RANGE]
GENE_TYPE = [float] * len(GENE_NAMES)


def decode_policy(genes) -> StoragePolicy:
    w_anomaly, w_frequency, w_size, t_hot, t_warm_ratio, summary_fraction = (float(g) for g in genes)
    if w_anomaly + w_frequency + w_size <= 0:
        # All-zero weights have no meaning; fall back to equal weighting
        # rather than letting the fitness function divide by zero.
        w_anomaly = w_frequency = w_size = 1.0
    t_hot = float(np.clip(t_hot, T_HOT_RANGE["low"], T_HOT_RANGE["high"]))
    return StoragePolicy(
        weight_anomaly_score=w_anomaly,
        weight_event_frequency=w_frequency,
        weight_block_size=w_size,
        t_hot=t_hot,
        t_warm=float(np.clip(t_warm_ratio, 0.0, MAX_T_WARM_RATIO)) * t_hot,
        summary_fraction=float(np.clip(summary_fraction, SUMMARY_FRACTION_RANGE["low"], SUMMARY_FRACTION_RANGE["high"])),
    )


def stratified_split(labels, test_size: float = DEFAULT_TEST_SIZE, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Row positions for the tuning half and the held-out half, stratified on
    the ground-truth label so both halves carry the same anomaly rate."""
    positions = np.arange(len(labels))
    tune, holdout = train_test_split(positions, test_size=test_size, stratify=np.asarray(labels), random_state=seed)
    return np.sort(tune), np.sort(holdout)


@dataclass
class GARunResult:
    best_policy: StoragePolicy
    best_fitness: float
    generations_completed: int
    best_fitness_per_generation: list[float] = field(default_factory=list)
    best_saved_pct_per_generation: list[float] = field(default_factory=list)
    best_weighted_retention_per_generation: list[float] = field(default_factory=list)
    best_warm_blocks_per_generation: list[int] = field(default_factory=list)
    generation_seconds: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    population_size: int = 0
    num_generations: int = 0
    num_parents_mating: int = 0

    def history_dict(self) -> dict:
        return {
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "num_parents_mating": self.num_parents_mating,
            "generations_completed": self.generations_completed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "best_fitness_per_generation": self.best_fitness_per_generation,
            "best_storage_saved_pct_per_generation": self.best_saved_pct_per_generation,
            "best_weighted_retention_per_generation": self.best_weighted_retention_per_generation,
            "best_warm_blocks_per_generation": self.best_warm_blocks_per_generation,
            "generation_seconds": self.generation_seconds,
        }


def make_fitness_func(table: BlockTable, retention_target: float, penalty_weight: float, warm_credit: float = WARM_CREDIT):
    def fitness_func(ga_instance, solution, solution_idx):
        outcome = evaluate_policy(table, decode_policy(solution), warm_credit)
        return penalized_fitness(outcome.storage_saved_pct, outcome.weighted_retention, retention_target, penalty_weight)

    return fitness_func


def run_ga(
    table: BlockTable,
    population_size: int = 80,
    num_generations: int = 80,
    num_parents_mating: int = 30,
    retention_target: float = RETENTION_TARGET,
    penalty_weight: float = PENALTY_WEIGHT,
    random_seed: int = RANDOM_STATE,
    verbose: bool = True,
    warm_credit: float = WARM_CREDIT,
) -> tuple[pygad.GA, GARunResult]:
    history = GARunResult(
        best_policy=None, best_fitness=0.0, generations_completed=0,
        population_size=population_size, num_generations=num_generations, num_parents_mating=num_parents_mating,
    )
    generation_start = {"t": time.perf_counter()}

    def on_generation(ga_instance):
        now = time.perf_counter()
        history.generation_seconds.append(now - generation_start["t"])
        generation_start["t"] = now

        solution, fitness, _ = ga_instance.best_solution()
        outcome = evaluate_policy(table, decode_policy(solution), warm_credit)
        history.best_fitness_per_generation.append(float(fitness))
        history.best_saved_pct_per_generation.append(outcome.storage_saved_pct)
        history.best_weighted_retention_per_generation.append(outcome.weighted_retention)
        history.best_warm_blocks_per_generation.append(outcome.tier_counts["WARM"])
        if verbose and (ga_instance.generations_completed % 10 == 0 or ga_instance.generations_completed == 1):
            print(
                f"  generation {ga_instance.generations_completed:>3}/{num_generations} "
                f"fitness={fitness:.3f} saved={outcome.storage_saved_pct:.2f}% "
                f"weighted_retention={outcome.weighted_retention:.4f} warm_blocks={outcome.tier_counts['WARM']} "
                f"({history.generation_seconds[-1]:.2f}s)"
            )

    start = time.perf_counter()
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=population_size,
        num_genes=len(GENE_NAMES),
        gene_space=copy.deepcopy(GENE_SPACE),
        gene_type=copy.deepcopy(GENE_TYPE),
        fitness_func=make_fitness_func(table, retention_target, penalty_weight, warm_credit),
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=34,  # 2 of 6 genes
        keep_elitism=2,
        on_generation=on_generation,
        random_seed=random_seed,
        suppress_warnings=True,
    )
    ga_instance.run()
    history.wall_clock_seconds = time.perf_counter() - start

    solution, fitness, _ = ga_instance.best_solution()
    history.best_policy = decode_policy(solution)
    history.best_fitness = float(fitness)
    history.generations_completed = int(ga_instance.generations_completed)
    return ga_instance, history


def _outcome_entry(outcome: StorageOutcome, target: float, penalty_weight: float, **extra) -> dict:
    return {
        **outcome.to_dict(),
        "meets_retention_target": outcome.meets_target(target),
        "fitness": outcome.fitness(target, penalty_weight),
        **extra,
    }


def compare_to_baselines(
    tune_table: BlockTable,
    holdout_table: BlockTable,
    ga_outcome: StorageOutcome,
    retention_target: float,
    penalty_weight: float,
    n_random_trials: int = 20,
    seed: int = RANDOM_STATE,
) -> dict:
    """Every baseline measured on the holdout half, scored with the same
    fitness the GA optimized, so "beats" has one definition: higher penalized
    storage saving on data the GA never saw."""
    def entry(outcome, **extra):
        return _outcome_entry(outcome, retention_target, penalty_weight, **extra)

    ga_fitness = ga_outcome.fitness(retention_target, penalty_weight)
    results: dict[str, dict] = {
        "keep_all": entry(baselines.keep_all(holdout_table)),
        "keep_flagged_only": entry(baselines.keep_flagged(holdout_table)),
    }

    # k is picked on the tuning half (smallest k that reaches the target there)
    # and applied to the holdout -- the strongest form of this baseline that
    # doesn't peek at held-out labels.
    k_pct = baselines.smallest_top_k_for_target(tune_table, retention_target)
    results["top_k_percent_tuned"] = entry(baselines.top_k_percent(holdout_table, k_pct), k_percent=k_pct)
    results["top_k_percent_same_savings"] = entry(
        baselines.top_k_matching_lines(holdout_table, ga_outcome.kept_lines),
    )

    random_runs = baselines.random_matching_lines(holdout_table, ga_outcome.kept_lines, n_random_trials, seed)
    random_entry = entry(random_runs[0])
    for key in ("storage_saved_pct", "weighted_retention", "full_retention", "coverage", "flagged_retention"):
        values = [getattr(run, key) for run in random_runs]
        random_entry[key] = float(np.mean(values))
        random_entry[f"{key}_std"] = float(np.std(values))
    random_entry["meets_retention_target"] = random_entry["weighted_retention"] >= retention_target
    random_entry["fitness"] = penalized_fitness(
        random_entry["storage_saved_pct"], random_entry["weighted_retention"], retention_target, penalty_weight
    )
    random_entry["n_trials"] = n_random_trials
    results["random_sampling_same_savings"] = random_entry

    sweep = {}
    for pct in (1, 2, 5, 10, 20, 30, 40, 50):
        sweep[f"{pct}%"] = baselines.top_k_percent(holdout_table, pct).to_dict()
    results["top_k_percent_sweep"] = sweep

    for name, result in results.items():
        if name == "top_k_percent_sweep":
            continue
        result["ga_better"] = ga_fitness > result["fitness"]
    return results


def warm_tier_diagnostics(table: BlockTable, policy: StoragePolicy, warm_credit: float) -> dict:
    """What the WARM tier actually costs and earns under a policy -- the numbers
    behind "is WARM a real tier or has it collapsed?"."""
    outcome = evaluate_policy(table, policy, warm_credit)
    warm_blocks = outcome.tier_counts["WARM"]
    warm_ef = table.event_frequency[assign_tiers(importance(table, policy), policy) == WARM]
    return {
        "warm_blocks": warm_blocks,
        "warm_block_share": warm_blocks / len(table) if len(table) else 0.0,
        "warm_anomalies": outcome.tier_anomalies["WARM"],
        "mean_lines_per_warm_block": outcome.tier_lines["WARM"] / warm_blocks if warm_blocks else None,
        "mean_event_frequency_of_warm_blocks": float(warm_ef.mean()) if warm_blocks else None,
        "share_of_warm_blocks_kept_at_one_line": (
            float((np.maximum(1, np.rint(policy.summary_fraction * warm_ef)) == 1).mean()) if warm_blocks else None
        ),
        "summary_fraction": policy.summary_fraction,
        "cold_line_cost": 1,
    }


@dataclass
class ScoringVariant:
    """One full tune -> holdout run: a score source, the GA's policy on it, and
    the tables both were evaluated on."""

    model_info: dict
    normalizer: Normalizer
    tune_table: BlockTable
    holdout_table: BlockTable
    ga_result: GARunResult

    @property
    def policy(self) -> StoragePolicy:
        return self.ga_result.best_policy


def variant_section(variant: ScoringVariant, retention_target: float, penalty_weight: float, warm_credit: float,
                    n_random_trials: int, seed: int) -> dict:
    policy = variant.policy
    tune_outcome = evaluate_policy(variant.tune_table, policy, warm_credit)
    holdout_outcome = evaluate_policy(variant.holdout_table, policy, warm_credit)
    comparison = compare_to_baselines(
        variant.tune_table, variant.holdout_table, holdout_outcome, retention_target, penalty_weight, n_random_trials, seed
    )
    compared = {name: r for name, r in comparison.items() if name != "top_k_percent_sweep"}
    return {
        "model": variant.model_info,
        "policy": policy.to_dict(),
        "tuning": _outcome_entry(tune_outcome, retention_target, penalty_weight),
        "holdout": _outcome_entry(holdout_outcome, retention_target, penalty_weight),
        "warm_tier": warm_tier_diagnostics(variant.holdout_table, policy, warm_credit),
        "baselines": comparison,
        "ga_beats_baselines": all(r["ga_better"] for r in compared.values()),
        "ga_history": variant.ga_result.history_dict(),
    }


COMPARISON_METRICS = ("storage_saved_pct", "weighted_retention", "full_retention", "coverage", "fitness", "meets_retention_target")


def holdout_comparison(sections: dict[str, dict]) -> list[dict]:
    """The holdout table with one column per scoring variant, row per policy."""
    first = next(iter(sections.values()))
    names = ["ga_policy"] + [name for name in first["baselines"] if name != "top_k_percent_sweep"]
    rows = []
    for name in names:
        row = {"name": name}
        for variant, section in sections.items():
            entry = section["holdout"] if name == "ga_policy" else section["baselines"][name]
            row[variant] = {metric: entry[metric] for metric in COMPARISON_METRICS}
        rows.append(row)
    return rows


def build_report(
    primary: ScoringVariant,
    comparison: ScoringVariant | None,
    retention_target: float,
    penalty_weight: float,
    test_size: float,
    seed: int,
    warm_credit: float = WARM_CREDIT,
    n_random_trials: int = 20,
) -> tuple[dict, dict]:
    """Returns (policy_best.json payload, storage_report.json payload).

    `primary` is the clean run (model refitted on the tune half only), whose
    policy is the one saved and served. `comparison` is the same protocol on
    the deployed model's scores, kept for the side-by-side column."""
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    section = lambda v: variant_section(v, retention_target, penalty_weight, warm_credit, n_random_trials, seed)
    primary_section = section(primary)

    policy_payload = {
        "generated_at": generated_at,
        "model": primary.model_info,
        "policy": primary.policy.to_dict(),
        "normalizer": primary.normalizer.to_dict(),
        "retention_target": retention_target,
        "penalty_weight": penalty_weight,
        "warm_credit": warm_credit,
        "split": {"test_size": test_size, "seed": seed},
    }
    report_payload = {
        "generated_at": generated_at,
        "config": {
            "retention_target": retention_target,
            "penalty_weight": penalty_weight,
            "warm_credit": warm_credit,
            "error_weight": ERROR_WEIGHT,
            "test_size": test_size,
            "seed": seed,
            "anomaly_definition": "ground-truth label (anomaly_label.csv); flagged_retention reports the model's own Anomaly calls",
            "retention_definitions": {
                "weighted_retention": "(HOT + warm_credit * WARM anomalies) / all anomalies -- used by fitness and the target",
                "full_retention": "HOT anomalies / all anomalies",
                "coverage": "(HOT + WARM anomalies) / all anomalies",
            },
        },
        "split": {
            "tune": {"blocks": len(primary.tune_table), "anomalies": int(primary.tune_table.is_anomaly.sum()), "lines": int(primary.tune_table.event_frequency.sum())},
            "holdout": {"blocks": len(primary.holdout_table), "anomalies": int(primary.holdout_table.is_anomaly.sum()), "lines": int(primary.holdout_table.event_frequency.sum())},
        },
        **primary_section,
    }
    if comparison is not None:
        comparison_section = section(comparison)
        report_payload["deployed_model_scoring"] = comparison_section
        report_payload["holdout_comparison"] = holdout_comparison({"clean": primary_section, "deployed": comparison_section})
    return policy_payload, report_payload
