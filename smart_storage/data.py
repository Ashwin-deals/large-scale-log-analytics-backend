"""Scores the labeled blocks with an Isolation Forest.

Everything the storage policy needs per block -- the model's anomaly score
plus event_frequency, block_size, is_error and the ground-truth label -- is
assembled here, so the optimizer script and the API see the same frame.

Two score sources:
  - the deployed model, which was fitted on every block, so both halves of
    the storage split are in-sample for it;
  - a clean refit: the deployed model's hyperparameters, refitted on the tune
    half only (refit_on_rows), so holdout scores come from a model that never
    saw those blocks. This is the one the tuned policy is built on.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from detection.predict import predict
from detection.train import load_labeled_features
from optimization.model_evolution import DEFAULT_CURRENT_VERSION_PATH

STORAGE_DIR = Path("data/storage")
POLICY_PATH = STORAGE_DIR / "policy_best.json"
REPORT_PATH = STORAGE_DIR / "storage_report.json"
TUNE_HALF_MODEL_PATH = STORAGE_DIR / "isolation_forest_tune_half.pkl"
# predict() always writes its output CSV; this is the scratch copy.
SCRATCH_PREDICTIONS_PATH = STORAGE_DIR / "_scored_blocks.csv"

DEFAULT_MODEL_PATH = Path("data/models/isolation_forest_v1.pkl")

FRAME_COLUMNS = [
    "block_id", "anomaly_score", "predicted_label", "true_label",
    "event_frequency", "block_size", "is_error",
]


def resolve_deployed_model() -> dict:
    """{version, model_path} for whatever current_version.json says is live
    (V1 before the first promotion, as everywhere else in the pipeline)."""
    if not DEFAULT_CURRENT_VERSION_PATH.exists():
        return {"version": 1, "model_path": str(DEFAULT_MODEL_PATH)}
    current = json.loads(DEFAULT_CURRENT_VERSION_PATH.read_text(encoding="utf-8"))
    return {"version": current.get("version"), "model_path": current["model_path"]}


def score_blocks(model, merged: pd.DataFrame) -> pd.DataFrame:
    """One row per row of `merged` (load_labeled_features() output), in the same order."""
    with warnings.catch_warnings():
        # A model saved before encoder fingerprints only warns; the pipeline's
        # own predict() contract is to proceed, so don't spam every call.
        warnings.simplefilter("ignore", RuntimeWarning)
        predictions = predict(
            model, merged, output_path=SCRATCH_PREDICTIONS_PATH, feature_columns=list(model.feature_names_in_)
        )
    frame = pd.DataFrame(
        {
            "block_id": merged["block_id"].to_numpy(),
            "anomaly_score": predictions["anomaly_score"].to_numpy(),
            "predicted_label": predictions["predicted_label"].to_numpy(),
            "true_label": merged["Label"].to_numpy(),
            "event_frequency": merged["event_frequency"].to_numpy(),
            "block_size": merged["block_size"].to_numpy(),
            "is_error": merged["is_error"].to_numpy(),
        }
    )
    return frame[FRAME_COLUMNS]


def refit_on_rows(template, merged: pd.DataFrame, rows):
    """A fresh fit of `template`'s exact hyperparameters (sklearn clone: same
    get_params(), random_state included) on `merged.iloc[rows]` only, using the
    feature columns `template` was trained on."""
    features = list(template.feature_names_in_)
    model = clone(template)
    model.fit(merged.iloc[rows][features])
    return model


def load_labeled_blocks() -> pd.DataFrame:
    return load_labeled_features().reset_index(drop=True)


def load_scored_blocks(model_path: str | Path | None = None) -> pd.DataFrame:
    """One row per labeled block, scored by `model_path` (default: the deployed model)."""
    model = joblib.load(model_path or resolve_deployed_model()["model_path"])
    return score_blocks(model, load_labeled_blocks())
