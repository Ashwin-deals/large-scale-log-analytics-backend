"""The batch models, loaded once and scored in micro-batches.

  early warning   data/models/early_warning/isolation_forest_k{k}.pkl, alert
                  when -score_samples >= the GA-tuned alert_threshold_
  full score      the deployed model (current_version.json), the same
                  Normal/Anomaly + anomaly_score detection/predict.py produces
  storage tier    the Smart Storage policy (policy_best.json) with the model
                  it was tuned on, so importance uses the score scale the
                  policy's normalizer was fitted to

Scoring is per micro-batch (one DataFrame per call, not per block): a
single-row IsolationForest call is dominated by fixed overhead. n_jobs is
pinned to 1 because thread fan-out costs more than it saves at these sizes.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from detection.predict import check_encoder_compatibility
from smart_storage.data import POLICY_PATH, resolve_deployed_model
from smart_storage.policy import COLD, HOT, WARM, BlockTable, Normalizer, StoragePolicy, assign_tiers, importance, summary_lines

EARLY_WARNING_DIR = Path("data/models/early_warning")
TIER_NAMES = {HOT: "HOT", WARM: "WARM", COLD: "COLD"}


def _load(path, vocab_path=None):
    model = joblib.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pre-fingerprint models only warn, as in batch
        # A renumbered vocabulary refuses to score, as in batch; each source checks against its own.
        check_encoder_compatibility(model, **({"vocab_path": vocab_path} if vocab_path else {}))
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


@dataclass
class EarlyWarningModel:
    k: int
    model: object
    threshold: float

    def scores(self, rows: list[dict]) -> np.ndarray:
        frame = pd.DataFrame(rows)
        return -self.model.score_samples(frame[list(self.model.feature_names_in_)])


def scorer_for(source: str, ks) -> "Scorer":
    """The Scorer for one log source: its deployed model, early-warning
    models and Smart Storage policy (HDFS keeps the original paths)."""
    from pipeline_api import resolve_current_deployment
    from source_context import get_context

    ctx = get_context(source)
    if ctx.is_hdfs:
        return Scorer(ks)
    current = resolve_current_deployment(ctx)
    ctx.require(Path(current["model_path"]), ctx.policy_path, ctx.vocab_path)
    return Scorer(ks, early_warning_dir=ctx.models_dir / "early_warning", deployed_model_path=current["model_path"],
                  policy_path=ctx.policy_path, deployed_version=current.get("version"), vocab_path=ctx.vocab_path)


class Scorer:
    def __init__(self, ks, early_warning_dir=EARLY_WARNING_DIR, deployed_model_path=None, policy_path=POLICY_PATH,
                 deployed_version=None, vocab_path=None):
        self.early = {}
        for k in sorted(ks):
            model = _load(Path(early_warning_dir) / f"isolation_forest_k{k}.pkl", vocab_path)
            self.early[k] = EarlyWarningModel(k, model, float(model.alert_threshold_))

        deployed = resolve_deployed_model() if deployed_model_path is None else None
        self.deployed_version = deployed["version"] if deployed else deployed_version
        self.deployed = _load(deployed_model_path or deployed["model_path"], vocab_path)

        policy_payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        self.policy = StoragePolicy.from_dict(policy_payload["policy"])
        self.normalizer = Normalizer.from_dict(policy_payload["normalizer"])
        self.storage_model = _load(policy_payload["model"]["model_path"], vocab_path)

    def full(self, rows: list[dict]) -> dict:
        """Deployed-model verdict and Smart Storage tier for finished blocks."""
        frame = pd.DataFrame(rows)
        deployed_x = frame[list(self.deployed.feature_names_in_)]
        labels = np.where(self.deployed.predict(deployed_x) == -1, "Anomaly", "Normal")
        scores = -self.deployed.decision_function(deployed_x)

        storage_scores = -self.storage_model.decision_function(frame[list(self.storage_model.feature_names_in_)])
        event_frequency = np.maximum(1, frame["event_frequency"].to_numpy(dtype=np.int64))
        table = BlockTable(
            parts=self.normalizer.parts(storage_scores, event_frequency, frame["block_size"].to_numpy(dtype=float)),
            is_error=frame["is_error"].to_numpy(dtype=float),
            event_frequency=event_frequency,
            is_anomaly=np.zeros(len(frame), dtype=bool),  # unknown in a live stream
            is_flagged=labels == "Anomaly",
            anomaly_score=storage_scores,
        )
        block_importance = importance(table, self.policy)
        tiers = assign_tiers(block_importance, self.policy)
        # Lines kept per tier, as smart_storage.policy.evaluate_policy costs them.
        lines_kept = np.where(
            tiers == HOT, event_frequency,
            # WARM: COLD's metadata line plus the summary, exactly as evaluate_policy charges it.
            np.where(tiers == WARM, 1 + summary_lines(event_frequency, self.policy.summary_fraction), 1),
        )
        return {
            "predicted_label": labels,
            "anomaly_score": scores,
            "importance": block_importance,
            "tier": [TIER_NAMES[t] for t in tiers],
            "lines_kept": lines_kept,
        }
