"""Score an uploaded log of any supported source against that source's model.

HDFS uploads keep their existing path in sources_api.py (which also folds new
blocks into the HDFS dashboard totals). Every other source runs the same
clean -> extract -> predict stages with its own vocabulary and its own
deployed model (data/models/<source>/current_version.json). Its blocks never
extend the HDFS dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import upload_store
from detection.predict import predict
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor

from .pipeline import Paths
from .sources import SOURCES, get_source

UPLOAD_SOURCES = ("hdfs",) + tuple(SOURCES)


def deployed_model(source: str) -> dict | None:
    path = Paths(source).current_version
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def parse_upload(source: str, stored_path: str | Path) -> pd.DataFrame:
    return get_source(source).parser_factory().parse(str(stored_path))


def recognized_mask(frame: pd.DataFrame) -> pd.Series:
    if not len(frame):
        return pd.Series([], dtype=bool)
    return frame["date"].notna() & frame["block_id"].notna()


def score_upload(source: str, recognized_frame: pd.DataFrame, upload_id, processed_dir: Path, results_dir: Path) -> dict:
    current = deployed_model(source)
    if current is None:
        raise FileNotFoundError(f"No trained model for source '{source}'. Run: python3 scripts/run_source_pipeline.py --source {source}")
    paths = Paths(source)
    upload_store.require_vocabulary(paths.vocab)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cleaned = HDFSDataCleaner(drop_duplicates=get_source(source).dedupe).clean(
        recognized_frame, processed_dir / f"{upload_id}_cleaned.csv")
    features_path = processed_dir / f"{upload_id}_features.csv"
    # Same contract as an HDFS upload: unseen values are appended to this
    # source's vocabulary (never renumbered), so the model's codes stay valid.
    HDFSFeatureExtractor().extract(cleaned, features_path, vocab_path=paths.vocab)
    features = pd.read_csv(features_path)
    if len(features) == 0:
        return {"blocks_analyzed": 0}

    model = joblib.load(current["model_path"])
    predictions = predict(model, features, output_path=upload_store.results_path(upload_id, results_dir),
                          feature_columns=list(model.feature_names_in_), vocab_path=paths.vocab, require_encoder_vocab=True)
    total = len(predictions)
    anomalies = int((predictions["predicted_label"] == "Anomaly").sum())
    return {
        "blocks_analyzed": total,
        "anomalies_detected": anomalies,
        "anomaly_rate_pct": round(anomalies / total * 100, 2) if total else 0.0,
        "model_version": current.get("version"),
        "scoring": upload_store.scoring_provenance(model, current, paths.vocab),
    }
