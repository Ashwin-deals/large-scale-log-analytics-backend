"""Importance scoring, tier assignment and storage accounting.

Importance
----------
Four normalized parts in [0, 1], combined as a weighted sum:

  anomaly_score    -- deployed model's score, min-max scaled
  event_frequency  -- log1p of lines per block, min-max scaled (the raw counts
                      are heavy-tailed: median ~19, max ~298)
  block_size       -- percentile rank (ties get their mid-rank)
  is_error         -- 0/1

The first three weights are the genes the optimizer tunes; they are rescaled
to sum to 1. is_error is not a gene: it carries a fixed share, ERROR_WEIGHT,
of the final score, so a block with an error line always gets a floor of that
share no matter how the tuned weights fall.

    importance = (1 - ERROR_WEIGHT) * sum(w_i * part_i) + ERROR_WEIGHT * is_error

Tiers and cost
--------------
  HOT   importance >= t_hot                 keep every line      (event_frequency)
  WARM  t_warm <= importance < t_hot        metadata + summary   (1 + round(f * event_frequency))
  COLD  importance < t_warm                 drop, keep metadata  (1 line)

A WARM block keeps the same metadata line as COLD plus round(f * lines)
summary lines, so WARM can never cost less than COLD and a summary that
rounds to zero lines is just COLD.

Storage cost is lines kept, using event_frequency as lines per block.

Retention
---------
Three numbers, all over ground-truth anomalies:

  full_retention      HOT anomalies / all anomalies
  coverage            (HOT + WARM) anomalies / all anomalies
  weighted_retention  (HOT + sum of WARM credits) / all -- what fitness and
                      the retention target use

WARM credit is proportional to how much of the block the summary keeps:

    credit = warm_credit * min(1, summary_lines / (warm_ref_fraction * event_frequency))

A summary that keeps at least warm_ref_fraction (default WARM_REF_FRACTION =
0.25) of the block earns the full warm_credit; a thinner one earns
proportionally less.

Why both rules: previously WARM cost max(1, round(f * lines)) and every WARM
anomaly earned a flat warm_credit. A 1-line "summary" then cost exactly what
COLD costs but still earned credit, so the optimizer emptied COLD into a WARM
tier that kept one line per block (the "WARM tier collapse"). Proportional
credit alone does not stop that -- the 1-line floor still hands out free
credit -- so the metadata line is now charged separately and credit is paid
only for summary lines bought on top of it.

warm_credit = 0 makes weighted_retention equal full_retention; warm_credit = 1
with summaries at or above the reference fraction makes it equal coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIERS = ("HOT", "WARM", "COLD")
HOT, WARM, COLD = 0, 1, 2

WEIGHT_NAMES = ("anomaly_score", "event_frequency", "block_size")
ERROR_WEIGHT = 0.2
RETENTION_TARGET = 0.95
WARM_CREDIT = 0.5
# Share of a block's lines a WARM summary must keep to earn the full
# warm_credit; thinner summaries earn proportionally less.
WARM_REF_FRACTION = 0.25
# Fitness points lost per unit (0..1) of anomaly-retention shortfall. Savings
# live on a 0..100 scale, so 1000 means missing the target by one percentage
# point costs ten points of savings -- steep enough that an infeasible policy
# can't win by trading retention for storage, while still leaving a gradient
# for the search to climb out of infeasible territory.
PENALTY_WEIGHT = 1000.0
PERCENTILE_GRID_SIZE = 1001
POSITIVE_LABEL = "Anomaly"


def _span(low: float, high: float) -> float:
    return (high - low) or 1.0


@dataclass
class Normalizer:
    """Scaling statistics, fitted once (on the tuning half) and then applied
    unchanged to any other block set, so the held-out half is never allowed to
    influence its own scoring. Values outside the fitted range clip to 0 / 1."""

    score_min: float
    score_max: float
    log_freq_min: float
    log_freq_max: float
    size_quantiles: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, anomaly_score, event_frequency, block_size) -> "Normalizer":
        score = np.asarray(anomaly_score, dtype=float)
        log_freq = np.log1p(np.asarray(event_frequency, dtype=float))
        grid = np.linspace(0.0, 1.0, PERCENTILE_GRID_SIZE)
        return cls(
            score_min=float(score.min()),
            score_max=float(score.max()),
            log_freq_min=float(log_freq.min()),
            log_freq_max=float(log_freq.max()),
            size_quantiles=[float(q) for q in np.quantile(np.asarray(block_size, dtype=float), grid)],
        )

    def size_percentile(self, block_size) -> np.ndarray:
        quantiles = np.asarray(self.size_quantiles, dtype=float)
        values = np.asarray(block_size, dtype=float)
        # block_size is dominated by ties (a fixed 64MiB block size and a pile
        # of zeros), so a value sitting on a plateau takes the middle of it
        # rather than either edge.
        left = np.searchsorted(quantiles, values, side="left")
        right = np.searchsorted(quantiles, values, side="right")
        return (left + right) / (2.0 * len(quantiles))

    def parts(self, anomaly_score, event_frequency, block_size) -> np.ndarray:
        """(3, n) matrix of normalized parts, rows in WEIGHT_NAMES order."""
        score = (np.asarray(anomaly_score, dtype=float) - self.score_min) / _span(self.score_min, self.score_max)
        log_freq = np.log1p(np.asarray(event_frequency, dtype=float))
        freq = (log_freq - self.log_freq_min) / _span(self.log_freq_min, self.log_freq_max)
        size = self.size_percentile(block_size)
        return np.clip(np.vstack([score, freq, size]), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "score_min": self.score_min,
            "score_max": self.score_max,
            "log_freq_min": self.log_freq_min,
            "log_freq_max": self.log_freq_max,
            "size_quantiles": self.size_quantiles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(
            score_min=float(data["score_min"]),
            score_max=float(data["score_max"]),
            log_freq_min=float(data["log_freq_min"]),
            log_freq_max=float(data["log_freq_max"]),
            size_quantiles=[float(q) for q in data["size_quantiles"]],
        )


@dataclass
class StoragePolicy:
    weight_anomaly_score: float
    weight_event_frequency: float
    weight_block_size: float
    t_hot: float
    t_warm: float
    summary_fraction: float

    def validate(self) -> "StoragePolicy":
        weights = self.raw_weights
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and not all zero")
        for name, value in (("t_hot", self.t_hot), ("t_warm", self.t_warm), ("summary_fraction", self.summary_fraction)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.t_warm < self.t_hot <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= t_warm < t_hot <= 1")
        if not 0.0 < self.summary_fraction <= 1.0:
            raise ValueError("summary_fraction must be in (0, 1]")
        return self

    @property
    def raw_weights(self) -> np.ndarray:
        return np.array(
            [self.weight_anomaly_score, self.weight_event_frequency, self.weight_block_size], dtype=float
        )

    @property
    def weights(self) -> np.ndarray:
        raw = self.raw_weights
        return raw / raw.sum()

    def to_dict(self) -> dict:
        return {
            "weights": dict(zip(WEIGHT_NAMES, (float(w) for w in self.weights))),
            "error_weight": ERROR_WEIGHT,
            "thresholds": {"hot": float(self.t_hot), "warm": float(self.t_warm)},
            "summary_fraction": float(self.summary_fraction),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoragePolicy":
        weights = data["weights"]
        return cls(
            weight_anomaly_score=float(weights["anomaly_score"]),
            weight_event_frequency=float(weights["event_frequency"]),
            weight_block_size=float(weights["block_size"]),
            t_hot=float(data["thresholds"]["hot"]),
            t_warm=float(data["thresholds"]["warm"]),
            summary_fraction=float(data["summary_fraction"]),
        ).validate()


@dataclass
class BlockTable:
    """The per-block arrays a policy is evaluated against."""

    parts: np.ndarray  # (3, n) normalized, WEIGHT_NAMES order
    is_error: np.ndarray  # (n,) 0/1 floats
    event_frequency: np.ndarray  # (n,) int lines per block, >= 1
    is_anomaly: np.ndarray  # (n,) bool, ground truth
    is_flagged: np.ndarray  # (n,) bool, the model's own Anomaly call
    anomaly_score: np.ndarray  # (n,) raw model score, for rank-based baselines

    def __len__(self) -> int:
        return len(self.event_frequency)

    @classmethod
    def from_frame(cls, frame, normalizer: Normalizer) -> "BlockTable":
        """frame needs anomaly_score, event_frequency, block_size, is_error,
        true_label ("Anomaly"/"Normal") and predicted_label."""
        event_frequency = np.maximum(1, np.asarray(frame["event_frequency"], dtype=np.int64))
        return cls(
            parts=normalizer.parts(frame["anomaly_score"], event_frequency, frame["block_size"]),
            is_error=(np.asarray(frame["is_error"], dtype=float) > 0).astype(float),
            event_frequency=event_frequency,
            is_anomaly=(np.asarray(frame["true_label"]) == POSITIVE_LABEL),
            is_flagged=(np.asarray(frame["predicted_label"]) == POSITIVE_LABEL),
            anomaly_score=np.asarray(frame["anomaly_score"], dtype=float),
        )

    def subset(self, index) -> "BlockTable":
        index = np.asarray(index)
        return BlockTable(
            parts=self.parts[:, index],
            is_error=self.is_error[index],
            event_frequency=self.event_frequency[index],
            is_anomaly=self.is_anomaly[index],
            is_flagged=self.is_flagged[index],
            anomaly_score=self.anomaly_score[index],
        )


def importance(table: BlockTable, policy: StoragePolicy) -> np.ndarray:
    return (1.0 - ERROR_WEIGHT) * (policy.weights @ table.parts) + ERROR_WEIGHT * table.is_error


def assign_tiers(scores: np.ndarray, policy: StoragePolicy) -> np.ndarray:
    tiers = np.full(len(scores), COLD, dtype=np.int8)
    tiers[scores >= policy.t_warm] = WARM
    tiers[scores >= policy.t_hot] = HOT
    return tiers


def penalized_fitness(storage_saved_pct: float, retention: float, target: float, penalty_weight: float = PENALTY_WEIGHT) -> float:
    return float(storage_saved_pct - penalty_weight * max(0.0, target - retention))


@dataclass
class StorageOutcome:
    n_blocks: int
    n_anomalies: int
    total_lines: int
    kept_lines: int
    storage_saved_pct: float
    weighted_retention: float  # HOT = 1, WARM = proportional credit (<= warm_credit), COLD = 0; used by fitness
    full_retention: float  # HOT anomalies only
    coverage: float  # HOT + WARM anomalies
    flagged_retention: float  # HOT share of the model's own Anomaly calls
    warm_credit: float  # the most a WARM anomaly can earn
    warm_ref_fraction: float  # summary share that earns the full warm_credit
    warm_credit_earned: float  # mean credit actually earned per WARM anomaly (0 if none)
    tier_counts: dict
    tier_lines: dict
    tier_anomalies: dict

    def fitness(self, target: float = RETENTION_TARGET, penalty_weight: float = PENALTY_WEIGHT) -> float:
        return penalized_fitness(self.storage_saved_pct, self.weighted_retention, target, penalty_weight)

    def meets_target(self, target: float = RETENTION_TARGET) -> bool:
        return self.weighted_retention >= target

    def to_dict(self) -> dict:
        return {
            "n_blocks": self.n_blocks,
            "n_anomalies": self.n_anomalies,
            "total_lines": self.total_lines,
            "kept_lines": self.kept_lines,
            "storage_saved_pct": self.storage_saved_pct,
            "weighted_retention": self.weighted_retention,
            "full_retention": self.full_retention,
            "coverage": self.coverage,
            "flagged_retention": self.flagged_retention,
            "warm_credit": self.warm_credit,
            "warm_ref_fraction": self.warm_ref_fraction,
            "warm_credit_earned": self.warm_credit_earned,
            "tier_counts": self.tier_counts,
            "tier_lines": self.tier_lines,
            "tier_anomalies": self.tier_anomalies,
        }


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _outcome(
    table: BlockTable,
    hot: np.ndarray,
    warm: np.ndarray,
    warm_lines: int,
    warm_anomaly_credit: float,
    warm_credit: float,
    warm_ref_fraction: float,
) -> StorageOutcome:
    """warm_anomaly_credit is the summed retention credit of the WARM anomalies."""
    cold = ~(hot | warm)
    tier_lines = {
        "HOT": int(table.event_frequency[hot].sum()),
        "WARM": int(warm_lines),
        "COLD": int(cold.sum()),  # one metadata line per dropped block
    }
    tier_anomalies = {
        "HOT": int((hot & table.is_anomaly).sum()),
        "WARM": int((warm & table.is_anomaly).sum()),
        "COLD": int((cold & table.is_anomaly).sum()),
    }
    total_lines = int(table.event_frequency.sum())
    kept_lines = sum(tier_lines.values())
    n_anomalies = int(table.is_anomaly.sum())
    return StorageOutcome(
        n_blocks=len(table),
        n_anomalies=n_anomalies,
        total_lines=total_lines,
        kept_lines=kept_lines,
        storage_saved_pct=100.0 * (1.0 - kept_lines / total_lines) if total_lines else 0.0,
        weighted_retention=_ratio(tier_anomalies["HOT"] + warm_anomaly_credit, n_anomalies),
        full_retention=_ratio(tier_anomalies["HOT"], n_anomalies),
        coverage=_ratio(tier_anomalies["HOT"] + tier_anomalies["WARM"], n_anomalies),
        flagged_retention=_ratio(int((hot & table.is_flagged).sum()), int(table.is_flagged.sum())),
        warm_credit=float(warm_credit),
        warm_ref_fraction=float(warm_ref_fraction),
        warm_credit_earned=float(warm_anomaly_credit / tier_anomalies["WARM"]) if tier_anomalies["WARM"] else 0.0,
        tier_counts={"HOT": int(hot.sum()), "WARM": int(warm.sum()), "COLD": int(cold.sum())},
        tier_lines=tier_lines,
        tier_anomalies=tier_anomalies,
    )


def summary_lines(event_frequency, summary_fraction: float) -> np.ndarray:
    """Summary lines a WARM block keeps on top of its metadata line: round(f * lines).
    np.rint is round-half-even, the same as Python's round()."""
    return np.rint(summary_fraction * np.asarray(event_frequency)).astype(np.int64)


def warm_block_credit(event_frequency, summary, warm_credit: float = WARM_CREDIT,
                      warm_ref_fraction: float = WARM_REF_FRACTION) -> np.ndarray:
    """Retention credit per WARM block: warm_credit * min(1, summary / (ref * lines))."""
    reference = warm_ref_fraction * np.asarray(event_frequency, dtype=float)
    return warm_credit * np.minimum(1.0, np.asarray(summary, dtype=float) / reference)


def evaluate_policy(table: BlockTable, policy: StoragePolicy, warm_credit: float = WARM_CREDIT,
                    warm_ref_fraction: float = WARM_REF_FRACTION) -> StorageOutcome:
    if not 0.0 < warm_ref_fraction <= 1.0:
        raise ValueError("warm_ref_fraction must be in (0, 1]")
    tiers = assign_tiers(importance(table, policy), policy)
    hot, warm = tiers == HOT, tiers == WARM
    warm_ef = table.event_frequency[warm]
    summary = summary_lines(warm_ef, policy.summary_fraction)
    credit = warm_block_credit(warm_ef, summary, warm_credit, warm_ref_fraction)
    warm_anomaly_credit = float(credit[table.is_anomaly[warm]].sum())
    warm_lines = int(warm.sum()) + int(summary.sum())  # one metadata line each, plus the summaries
    return _outcome(table, hot, warm, warm_lines, warm_anomaly_credit, warm_credit, warm_ref_fraction)


def outcome_from_kept_mask(table: BlockTable, kept: np.ndarray) -> StorageOutcome:
    """Cost of a keep-or-drop policy (the baselines): kept blocks keep every
    line, the rest keep one metadata line -- the same accounting as COLD.
    With no WARM tier, all three retention numbers coincide."""
    kept = np.asarray(kept, dtype=bool)
    return _outcome(table, kept, np.zeros(len(table), dtype=bool), 0, 0.0, WARM_CREDIT, WARM_REF_FRACTION)
