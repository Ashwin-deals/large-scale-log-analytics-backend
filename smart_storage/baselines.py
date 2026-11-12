"""Baselines the tuned policy has to beat.

All of them are keep-or-drop over whole blocks and are costed exactly like the
COLD tier (a dropped block still leaves one metadata line), so their savings
are directly comparable with the tiered policy's.
"""

from __future__ import annotations

import math

import numpy as np

from .policy import BlockTable, StorageOutcome, outcome_from_kept_mask


def keep_all(table: BlockTable) -> StorageOutcome:
    return outcome_from_kept_mask(table, np.ones(len(table), dtype=bool))


def keep_flagged(table: BlockTable) -> StorageOutcome:
    """Keep only the blocks the deployed model calls Anomaly."""
    return outcome_from_kept_mask(table, table.is_flagged)


def _score_order(table: BlockTable) -> np.ndarray:
    return np.argsort(-table.anomaly_score, kind="stable")


def top_k_percent(table: BlockTable, k_pct: float) -> StorageOutcome:
    """Keep the k% of blocks with the highest anomaly score."""
    n_keep = min(len(table), math.ceil(k_pct / 100.0 * len(table)))
    kept = np.zeros(len(table), dtype=bool)
    kept[_score_order(table)[:n_keep]] = True
    return outcome_from_kept_mask(table, kept)


def smallest_top_k_for_target(table: BlockTable, target: float) -> float:
    """Smallest k% whose top-k retains `target` of the ground-truth anomalies
    on this table. Meant to be fitted on the tuning half, then applied to the
    holdout, so the baseline gets the same tune/holdout treatment as the GA."""
    order = _score_order(table)
    total = table.is_anomaly.sum()
    if total == 0:
        return 0.0
    retained = np.cumsum(table.is_anomaly[order]) / total
    n_keep = int(np.searchsorted(retained, target, side="left")) + 1
    return 100.0 * min(n_keep, len(table)) / len(table)


def _count_for_lines(table: BlockTable, order: np.ndarray, target_lines: int) -> int:
    """How many blocks, taken in `order`, land closest to `target_lines` kept lines."""
    n = len(table)
    cost = n + np.concatenate([[0], np.cumsum(table.event_frequency[order] - 1)])
    at = int(np.searchsorted(cost, target_lines))
    candidates = [i for i in (at - 1, at) if 0 <= i <= n]
    return min(candidates, key=lambda i: abs(int(cost[i]) - target_lines))


def top_k_matching_lines(table: BlockTable, target_lines: int) -> StorageOutcome:
    """Highest-scoring blocks first, stopped where the kept-line count matches `target_lines`."""
    order = _score_order(table)
    kept = np.zeros(len(table), dtype=bool)
    kept[order[: _count_for_lines(table, order, target_lines)]] = True
    return outcome_from_kept_mask(table, kept)


def random_matching_lines(table: BlockTable, target_lines: int, n_trials: int = 20, seed: int = 0) -> list[StorageOutcome]:
    """Random blocks, stopped where the kept-line count matches `target_lines`. One outcome per trial."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(n_trials):
        order = rng.permutation(len(table))
        kept = np.zeros(len(table), dtype=bool)
        kept[order[: _count_for_lines(table, order, target_lines)]] = True
        outcomes.append(outcome_from_kept_mask(table, kept))
    return outcomes
