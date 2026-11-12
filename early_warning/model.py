"""Isolation Forest per prefix length k, GA-tuned alert threshold, held-out evaluation.

Protocol (the same blocks for every k, so the k's are comparable)
-----------------------------------------------------------------
  1. Stratified split of the labeled blocks into train / test (default 70/30).
  2. Fit an Isolation Forest on the train blocks' prefix features only
     (unsupervised; labels never reach fit). Columns that are constant on the
     train blocks are dropped first (error_count is all zero on HDFS -- the
     log only has INFO and WARN).
  3. GA-tune the alert threshold on the train blocks' scores and labels.
  4. Score the untouched test blocks once and report.

Threshold gene
--------------
One gene, `alert_quantile`: the threshold is that quantile of the train
blocks' anomaly scores, so the gene means the same thing (the share of train
blocks that would *not* alert) whatever the score scale of a given k's
model. Fitness is F1 on the Anomaly class, the metric optimization/ uses. With
a single gene there is nothing to cross over, so the GA runs on mutation
alone (crossover_type=None). gene_space/gene_type are deep-copied before they
reach pygad, the same as optimization/ga_isolation_forest.py.

Lead time
---------
When an alert fires on a block after its k-th event, the block has
total_events events in all, so the fraction still unseen is
(total_events - min(k, total_events)) / total_events. A block with no more
than k events is only complete when it ends, so its lead time is 0.
Reported over true-positive alerts (how early a correct alert is) and as
`expected_lead_per_alert` = sum of TP leads / all alerts, i.e. precision x
mean TP lead, where a false alarm earns no lead.

The same lead is also reported in seconds: block_end_seconds minus
prefix_end_seconds, the wall-clock (log-time) gap between the alert firing at
the k-th event and the block's last event, as the mean and median over true
alerts. It is 0 for a block with no more than k events. Timestamps have
1-second resolution, so a sub-second gap reads as 0.

Still-running blocks
--------------------
Blocks with no more than k events finish before (or at) the k-th event, so
alerting on them is detection after the fact. With many short anomalous HDFS
blocks, they can carry most of the precision at small k. `test_in_progress`
re-reports the same alerts on only the blocks with more than k events (those
still running when the alert fires), which is the early-warning performance proper.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pygad
from sklearn.model_selection import train_test_split

from detection.isolation_forest import build_isolation_forest

from .prefix_features import BLOCK_END_COLUMN, PREFIX_END_COLUMN, TOTAL_EVENTS_COLUMN

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.3
POSITIVE_LABEL = "Anomaly"

ALERT_QUANTILE_RANGE = {"low": 0.5, "high": 0.9995}
GENE_NAMES = ["alert_quantile"]
GENE_SPACE = [ALERT_QUANTILE_RANGE]
GENE_TYPE = [float]


def lead_fraction(total_events, k: int) -> np.ndarray:
    total = np.asarray(total_events, dtype=float)
    seen = np.minimum(total, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (total - seen) / total, 0.0)


def lead_seconds(frame: pd.DataFrame) -> np.ndarray | None:
    """Seconds from the alert (the k-th event) to the block's last event, or
    None when the frame predates the time bookkeeping columns."""
    if PREFIX_END_COLUMN not in frame.columns or BLOCK_END_COLUMN not in frame.columns:
        return None
    gap = frame[BLOCK_END_COLUMN].to_numpy(dtype=float) - frame[PREFIX_END_COLUMN].to_numpy(dtype=float)
    return np.where(np.isnan(gap), np.nan, np.maximum(gap, 0.0))


def stratified_split(labels, test_size: float = DEFAULT_TEST_SIZE, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(labels))
    train, test = train_test_split(positions, test_size=test_size, stratify=np.asarray(labels), random_state=seed)
    return np.sort(train), np.sort(test)


def f1_at(scores: np.ndarray, is_anomaly: np.ndarray, threshold: float) -> float:
    alert = scores >= threshold
    tp = int((alert & is_anomaly).sum())
    fp = int((alert & ~is_anomaly).sum())
    fn = int((~alert & is_anomaly).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _seconds_summary(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if not len(values):
        return {"mean_seconds_over_true_alerts": None, "median_seconds_over_true_alerts": None}
    return {"mean_seconds_over_true_alerts": float(values.mean()), "median_seconds_over_true_alerts": float(np.median(values))}


def alert_metrics(alert: np.ndarray, is_anomaly: np.ndarray, lead: np.ndarray, lead_secs: np.ndarray | None = None) -> dict:
    alert, is_anomaly = np.asarray(alert, bool), np.asarray(is_anomaly, bool)
    tp = int((alert & is_anomaly).sum())
    fp = int((alert & ~is_anomaly).sum())
    fn = int((~alert & is_anomaly).sum())
    tn = int((~alert & ~is_anomaly).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tp_lead = lead[alert & is_anomaly]
    n_alerts = tp + fp
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(alert) if len(alert) else 0.0,
        "alerts": n_alerts,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "lead_time": {
            "mean_over_true_alerts": float(tp_lead.mean()) if tp else 0.0,
            "median_over_true_alerts": float(np.median(tp_lead)) if tp else 0.0,
            "share_of_true_alerts_before_block_end": float((tp_lead > 0).mean()) if tp else 0.0,
            "expected_lead_per_alert": float(tp_lead.sum() / n_alerts) if n_alerts else 0.0,
            **(_seconds_summary(np.asarray(lead_secs, dtype=float)[alert & is_anomaly]) if lead_secs is not None else {}),
        },
    }


@dataclass
class ThresholdGAResult:
    alert_quantile: float
    threshold: float
    best_fitness: float
    best_fitness_per_generation: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    population_size: int = 0
    num_generations: int = 0

    def history_dict(self) -> dict:
        return {
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "best_fitness_per_generation": self.best_fitness_per_generation,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


def tune_threshold(
    scores: np.ndarray,
    is_anomaly: np.ndarray,
    population_size: int = 20,
    num_generations: int = 30,
    num_parents_mating: int = 8,
    random_seed: int = RANDOM_STATE,
) -> ThresholdGAResult:
    scores = np.asarray(scores, dtype=float)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    sorted_scores = np.sort(scores)

    def threshold_for(quantile: float) -> float:
        quantile = float(np.clip(quantile, ALERT_QUANTILE_RANGE["low"], ALERT_QUANTILE_RANGE["high"]))
        return float(np.quantile(sorted_scores, quantile))

    def fitness_func(ga_instance, solution, solution_idx):
        return f1_at(scores, is_anomaly, threshold_for(solution[0]))

    history: list[float] = []

    def on_generation(ga_instance):
        history.append(float(ga_instance.best_solution()[1]))

    start = time.perf_counter()
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=population_size,
        num_genes=len(GENE_NAMES),
        gene_space=copy.deepcopy(GENE_SPACE),
        gene_type=copy.deepcopy(GENE_TYPE),
        fitness_func=fitness_func,
        parent_selection_type="sss",
        crossover_type=None,  # one gene: nothing to recombine
        mutation_type="random",
        mutation_num_genes=1,
        keep_elitism=2,
        on_generation=on_generation,
        random_seed=random_seed,
        suppress_warnings=True,
    )
    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()
    quantile = float(np.clip(solution[0], ALERT_QUANTILE_RANGE["low"], ALERT_QUANTILE_RANGE["high"]))
    return ThresholdGAResult(
        alert_quantile=quantile,
        threshold=threshold_for(quantile),
        best_fitness=float(fitness),
        best_fitness_per_generation=history,
        wall_clock_seconds=time.perf_counter() - start,
        population_size=population_size,
        num_generations=num_generations,
    )


def anomaly_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous, the convention detection/predict.py uses."""
    return -model.score_samples(X[list(model.feature_names_in_)])


@dataclass
class KResult:
    k: int
    model: object
    feature_columns: list[str]
    dropped_constant_columns: list[str]
    ga: ThresholdGAResult
    train_metrics: dict
    test_metrics: dict
    test_in_progress_metrics: dict
    test_blocks_shorter_than_k: int

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "features": self.feature_columns,
            "dropped_constant_columns": self.dropped_constant_columns,
            "threshold": {"alert_quantile": self.ga.alert_quantile, "score_threshold": self.ga.threshold, "train_f1": self.ga.best_fitness},
            "train": self.train_metrics,
            "test": self.test_metrics,
            "test_in_progress": self.test_in_progress_metrics,
            "test_blocks_with_at_most_k_events": self.test_blocks_shorter_than_k,
            "ga_history": self.ga.history_dict(),
        }


def train_and_evaluate_k(
    k: int,
    frame: pd.DataFrame,
    is_anomaly: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_columns: list[str],
    ga_kwargs: dict | None = None,
    model_factory=build_isolation_forest,
) -> KResult:
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    constant = [c for c in feature_columns if train[c].nunique(dropna=False) <= 1]
    columns = [c for c in feature_columns if c not in constant]

    model = model_factory()
    model.fit(train[columns])  # unsupervised: labels never reach fit

    train_scores, test_scores = anomaly_scores(model, train), anomaly_scores(model, test)
    ga = tune_threshold(train_scores, is_anomaly[train_idx], **(ga_kwargs or {}))

    train_lead = lead_fraction(train[TOTAL_EVENTS_COLUMN], k)
    test_lead = lead_fraction(test[TOTAL_EVENTS_COLUMN], k)
    train_secs, test_secs = lead_seconds(train), lead_seconds(test)
    test_alert, test_truth = test_scores >= ga.threshold, is_anomaly[test_idx]
    running = test[TOTAL_EVENTS_COLUMN].to_numpy() > k
    in_progress = alert_metrics(test_alert[running], test_truth[running], test_lead[running],
                                None if test_secs is None else test_secs[running])
    in_progress["blocks"] = int(running.sum())
    in_progress["anomalies"] = int(test_truth[running].sum())
    return KResult(
        k=k,
        model=model,
        feature_columns=columns,
        dropped_constant_columns=constant,
        ga=ga,
        train_metrics=alert_metrics(train_scores >= ga.threshold, is_anomaly[train_idx], train_lead, train_secs),
        test_metrics=alert_metrics(test_alert, test_truth, test_lead, test_secs),
        test_in_progress_metrics=in_progress,
        test_blocks_shorter_than_k=int((test[TOTAL_EVENTS_COLUMN] <= k).sum()),
    )


def trade_off(results: dict[int, dict], split: str = "test") -> dict:
    """Precision vs lead time across k, on the held-out metrics (`split` is
    "test" for every test block, or "test_in_progress" for still-running ones).

    Pareto front: a k is kept unless another k has at least its precision and
    at least its mean true-alert lead time, and is strictly better on one.
    Best: the front member with the highest expected lead per alert
    (precision x mean true-alert lead -- the expected share of a block still
    ahead of you, per alert raised, with false alarms counting as zero).
    """
    points = {
        k: (r[split]["precision"], r[split]["lead_time"]["mean_over_true_alerts"], r[split]["lead_time"]["expected_lead_per_alert"])
        for k, r in results.items()
    }

    def dominated(k):
        p, lead, _ = points[k]
        return any(
            q >= p and other_lead >= lead and (q > p or other_lead > lead)
            for other, (q, other_lead, _) in points.items() if other != k
        )

    front = sorted(k for k in points if not dominated(k))
    best = max(front, key=lambda k: points[k][2]) if front else None
    return {
        "split": split,
        "criterion": "Pareto front on (test precision, mean lead over true alerts); best = highest expected_lead_per_alert (precision x mean true-alert lead) on the front",
        "points": [
            {"k": k, "precision": p, "mean_lead_over_true_alerts": lead, "expected_lead_per_alert": e, "on_pareto_front": k in front}
            for k, (p, lead, e) in sorted(points.items())
        ],
        "pareto_front": front,
        "best_k": best,
    }
