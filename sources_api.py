"""Log Sources API — file upload, upload history, and connector status.

Uploads are parsed off the request thread: a large log would otherwise hold
the connection open past any sane timeout. The record lands in Mongo as
"processing" and a worker flips it to "processed" or "failed", which is the
lifecycle the dashboard's status column already renders.

A processed upload isn't just parsed — it runs through the same clean ->
extract-features -> predict pipeline as the base HDFS dataset, scored
against whichever model is currently deployed. There's no per-account
retraining yet (the base dataset is still the one shared training set); this
only wires up real *inference* on a user's own file. See _score_upload.
"""

import threading
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, g, jsonify, request
from werkzeug.utils import secure_filename

from auth import token_required
from db import uploads
from detection.predict import predict
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from parser.hdfs_parser import HDFSParser
from pipeline_api import resolve_current_deployment

sources_bp = Blueprint("sources", __name__)

UPLOAD_PROCESSED_DIR = Path("data/uploads/processed")
UPLOAD_RESULTS_DIR = Path("data/uploads/results")

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".log", ".json", ".csv", ".txt"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # matches the 500MB the upload UI advertises

# Connectors are declared here rather than in the dashboard so the "not
# connected" set is a backend fact. Only manual upload is actually wired up.
CONNECTORS = [
    {"name": "Manual File Upload", "icon": "Database", "enabled": True, "accent": "primary"},
    {"name": "AWS CloudWatch", "icon": "Cloud", "enabled": False},
    {"name": "Azure Monitor", "icon": "Server", "enabled": False},
    {"name": "GCP Logging", "icon": "Waypoints", "enabled": False},
    {"name": "Kafka Stream", "icon": "ListTree", "enabled": False},
    {"name": "Syslog (RFC 5424)", "icon": "FileText", "enabled": False},
]


def _serialize(doc):
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name"),
        "source": doc.get("source", "Manual Upload"),
        "size_bytes": doc.get("size_bytes", 0),
        "records": doc.get("records"),
        "total_lines": doc.get("total_lines"),
        "status": doc.get("status", "processing"),
        "error": doc.get("error"),
        "blocks_analyzed": doc.get("blocks_analyzed"),
        "anomalies_detected": doc.get("anomalies_detected"),
        "anomaly_rate_pct": doc.get("anomaly_rate_pct"),
        "model_version": doc.get("model_version"),
        "uploaded_at": doc["uploaded_at"].isoformat() if doc.get("uploaded_at") else None,
        "uploaded_by": doc.get("uploaded_by"),
    }


def _upload_block_records(cleaned: pd.DataFrame, features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-block records for this upload, in the same shape the dashboard's
    block_metadata.csv provides for the base dataset.

    The dashboard needs a timestamp, component, event_type and IPs per block
    to fold an upload into its totals; the predictions CSV alone only carries
    block_id/label/score. Aggregation matches scripts/build_block_metadata.py
    (dominant value per block) so an uploaded block is described the same way
    a base-dataset block is.
    """
    frame = cleaned.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["block_size"] = pd.to_numeric(frame["block_size"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for column in ["component", "event_type", "source_ip", "destination_ip", "block_id"]:
        frame[column] = frame[column].astype("string").str.strip().fillna("UNKNOWN")
        frame[column] = frame[column].replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})

    aggregated = frame.groupby("block_id", sort=False).agg(
        first_seen=("datetime", "first"),
        block_size=("block_size", "max"),
    ).reset_index()

    for column in ["component", "event_type", "source_ip", "destination_ip"]:
        dominant = HDFSFeatureExtractor._dominant_value_per_group(frame, "block_id", column)
        aggregated[column] = aggregated["block_id"].map(dominant)

    # event_frequency is this block's raw log-line count, which is what the
    # dashboard's "Total Logs Processed" counts (log lines, not blocks).
    aggregated = aggregated.merge(features[["block_id", "event_frequency"]], on="block_id", how="left")
    aggregated["event_frequency"] = aggregated["event_frequency"].fillna(0).astype("int64")

    return aggregated.merge(
        predictions[["block_id", "predicted_label", "anomaly_score"]], on="block_id", how="inner"
    )


def _score_upload(recognized_frame: pd.DataFrame, upload_id) -> dict:
    """Runs a freshly uploaded (and already-recognized) HDFS log through the
    same clean -> extract-features -> predict stages as the base dataset,
    scored against whichever model is currently deployed.

    Per-block predictions are written to UPLOAD_RESULTS_DIR/<upload_id>.csv
    for /api/sources/uploads/<id>/detections to read back, and the richer
    per-block record (adding timestamp/component/event_type/IPs/line count)
    to <upload_id>_blocks.csv, which is what pipeline_api.py folds into the
    dashboard totals. This function returns just the summary that lands on
    the upload's Mongo doc.

    HDFSFeatureExtractor.extract()'s *return value* drops block_id (only the
    CSV it writes keeps it — see feature_extractor.py), so read the CSV back
    rather than chaining its return value into predict(), same as every other
    caller in this codebase (scripts/build_hdfs_features.py included).
    """
    UPLOAD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = HDFSDataCleaner().clean(recognized_frame, UPLOAD_PROCESSED_DIR / f"{upload_id}_cleaned.csv")
    features_path = UPLOAD_PROCESSED_DIR / f"{upload_id}_features.csv"
    HDFSFeatureExtractor().extract(cleaned, features_path)
    features = pd.read_csv(features_path)

    if len(features) == 0:
        return {"blocks_analyzed": 0}

    current = resolve_current_deployment()
    model = joblib.load(current["model_path"])
    feature_columns = list(model.feature_names_in_)

    results_path = UPLOAD_RESULTS_DIR / f"{upload_id}.csv"
    predictions = predict(model, features, output_path=results_path, feature_columns=feature_columns)

    # Written last: pipeline_api.py treats the presence of this file as "this
    # upload is ready to count", so it must not appear before the predictions
    # it describes are on disk.
    records = _upload_block_records(cleaned, features, predictions)
    records.to_csv(UPLOAD_RESULTS_DIR / f"{upload_id}_blocks.csv", index=False)

    total = len(predictions)
    anomalies = int((predictions["predicted_label"] == "Anomaly").sum())
    return {
        "blocks_analyzed": total,
        "anomalies_detected": anomalies,
        "anomaly_rate_pct": round(anomalies / total * 100, 2) if total else 0.0,
        "model_version": current.get("version"),
    }


def _process_upload(upload_id, stored_path):
    """Parse the saved file, then run recognized lines through detection.

    The parser never rejects a line: anything it cannot match comes back as a
    row of nulls with event_type "OTHER". Counting rows would therefore report
    a plain CSV as successfully ingested with one "record" per line, so count
    only lines that actually matched the HDFS format, fail the upload when
    none of them did, and only score the recognized subset.
    """
    try:
        frame = HDFSParser().parse(str(stored_path))
        total_lines = int(len(frame))
        recognized_mask = frame["date"].notna() if total_lines else frame.index < 0
        recognized = int(recognized_mask.sum())

        if recognized == 0:
            update = {
                "status": "failed",
                "records": 0,
                "total_lines": total_lines,
                "error": "No lines matched the HDFS log format — this file does not look like an HDFS log.",
            }
        else:
            summary = _score_upload(frame[recognized_mask].reset_index(drop=True), upload_id)
            if summary["blocks_analyzed"] == 0:
                update = {
                    "status": "failed",
                    "records": recognized,
                    "total_lines": total_lines,
                    "error": "Recognized lines didn't contain any complete blocks to analyze.",
                }
            else:
                update = {"status": "processed", "records": recognized, "total_lines": total_lines, **summary}

        update["processed_at"] = datetime.now(timezone.utc)
        uploads.update_one({"_id": upload_id}, {"$set": update})
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via the row's status
        uploads.update_one(
            {"_id": upload_id},
            {"$set": {"status": "failed", "error": str(exc)[:500], "processed_at": datetime.now(timezone.utc)}},
        )


@sources_bp.post("/api/sources/upload")
@token_required
def upload_log():
    if "file" not in request.files:
        return jsonify({"error": "No file was provided."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "No file was selected."}), 400

    filename = secure_filename(uploaded.filename)
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Prefix with a timestamp so re-uploading the same filename doesn't clobber.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stored_path = UPLOAD_DIR / f"{stamp}__{filename}"
    # Anything over MAX_UPLOAD_BYTES is already rejected with a 413 by Flask's
    # MAX_CONTENT_LENGTH (set in app.py), so nothing oversized reaches disk.
    uploaded.save(stored_path)
    size_bytes = stored_path.stat().st_size

    doc = {
        "name": filename,
        "stored_path": str(stored_path),
        "source": "Manual Upload",
        "size_bytes": size_bytes,
        "records": None,
        "status": "processing",
        "uploaded_at": datetime.now(timezone.utc),
        "uploaded_by": (g.user or {}).get("email"),
    }
    result = uploads.insert_one(doc)
    doc["_id"] = result.inserted_id

    threading.Thread(target=_process_upload, args=(result.inserted_id, stored_path), daemon=True).start()

    return jsonify(_serialize(doc)), 201


def current_user_email():
    return (g.user or {}).get("email")


@sources_bp.get("/api/sources/uploads")
@token_required
def list_uploads():
    try:
        limit = min(int(request.args.get("limit", 25)), 100)
    except ValueError:
        limit = 25

    # Scoped to the signed-in account. MONGO_URI points at a shared cluster,
    # so an unfiltered find() listed uploads made by other people on other
    # machines — rows whose result files live on those machines and whose
    # anomaly counts came from a different local model, which read as this
    # pipeline being non-deterministic for the same file.
    owner = {"uploaded_by": current_user_email()}
    rows = list(uploads.find(owner).sort("uploaded_at", -1).limit(limit))
    return jsonify({"uploads": [_serialize(r) for r in rows], "total": uploads.count_documents(owner)})


@sources_bp.get("/api/sources/connectors")
@token_required
def list_connectors():
    return jsonify({"connectors": CONNECTORS})


@sources_bp.get("/api/sources/uploads/<upload_id>/detections")
@token_required
def upload_detections(upload_id):
    try:
        doc = uploads.find_one({"_id": ObjectId(upload_id)})
    except InvalidId:
        return jsonify({"error": "Invalid upload id."}), 400

    if doc is None or doc.get("uploaded_by") != current_user_email():
        # Same-cluster uploads from another account are not this user's to read.
        return jsonify({"error": "Upload not found."}), 404

    results_path = UPLOAD_RESULTS_DIR / f"{upload_id}.csv"
    if doc.get("status") != "processed" or not results_path.exists():
        return jsonify({"error": "Detection results are not available for this upload."}), 404

    try:
        page = max(int(request.args.get("page", 1)), 1)
        limit = min(max(int(request.args.get("limit", 25)), 1), 200)
    except (TypeError, ValueError):
        page, limit = 1, 25

    predictions = pd.read_csv(results_path).sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    total = len(predictions)
    start = (page - 1) * limit
    page_rows = predictions.iloc[start : start + limit]

    return jsonify(
        {
            "upload": {"id": str(doc["_id"]), "name": doc.get("name")},
            "blocks_analyzed": doc.get("blocks_analyzed"),
            "anomalies_detected": doc.get("anomalies_detected"),
            "anomaly_rate_pct": doc.get("anomaly_rate_pct"),
            "model_version": doc.get("model_version"),
            "total": total,
            "page": page,
            "limit": limit,
            "results": [
                {
                    "block_id": row.block_id,
                    "predicted_label": row.predicted_label,
                    "anomaly_score": round(float(row.anomaly_score), 4),
                }
                for row in page_rows.itertuples()
            ],
        }
    )
