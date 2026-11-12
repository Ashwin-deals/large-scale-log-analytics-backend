"""One place both Log Sources (sources_api.py) and the dashboard
(pipeline_api.py) read an upload's anomaly count from.

An upload's count used to be stored once, as a number on its Mongo doc, by
whichever server scored it. The cluster is shared, so the same file uploaded
from a server with a different code version, vocabulary or model produced a
different number (115 / 120 / 1,173 for review_dataset_full_columns.log), and
Log Sources showed that stored number while the dashboard, which can only
read result files on this machine, showed another upload's. Both now count
from the per-block results file on this server, and an upload scored
elsewhere reports no count rather than a conflicting one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering.category_encoder import CategoryEncoder

RESULTS_DIR = Path("data/uploads/results")
NOT_ON_THIS_SERVER = ("Scored by another server: its results are not on this machine, so no count is shown. "
                      "Re-upload the file here to score it with this server's model.")


def results_path(upload_id, results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / f"{upload_id}.csv"


def result_counts(upload_id, results_dir: Path = RESULTS_DIR) -> dict | None:
    """{blocks_analyzed, anomalies_detected, anomaly_rate_pct} from this
    server's results file for the upload, or None if it has none."""
    path = results_path(upload_id, results_dir)
    try:
        labels = pd.read_csv(path, usecols=["predicted_label"])["predicted_label"]
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    total = int(len(labels))
    anomalies = int((labels == "Anomaly").sum())
    return {
        "blocks_analyzed": total,
        "anomalies_detected": anomalies,
        "anomaly_rate_pct": round(anomalies / total * 100, 2) if total else 0.0,
    }


def scoring_provenance(model, model_info: dict, vocab_path) -> dict:
    """What produced a count. model_version alone is not enough: a refit keeps
    the version number but changes the model (both 115 and 120 were "v2")."""
    return {
        "model_version": model_info.get("version"),
        "model_path": str(model_info.get("model_path")),
        "model_encoder_fingerprint": getattr(model, "encoder_fingerprint_", None),
        "vocab_fingerprint": CategoryEncoder.load(vocab_path).fingerprint(),
    }


def require_vocabulary(vocab_path) -> None:
    """Refuse to score without the persisted vocabulary. Extracting features
    without it silently assigns fresh codes from 0, which the model reads as
    different categories (the 5,000-of-5,000 / 1,173 failure mode), and also
    writes that fresh vocabulary to disk."""
    if not Path(vocab_path).exists():
        raise FileNotFoundError(
            f"Category vocabulary {vocab_path} not found, so this upload cannot be encoded the way the model was "
            "trained. Is the server running from the repo root with features built (scripts/build_hdfs_features.py)?"
        )
