"""Log Sources API — file upload, upload history, and connector status.

Uploads are parsed off the request thread: a large log would otherwise hold
the connection open past any sane timeout. The record lands in Mongo as
"processing" and a worker flips it to "processed" or "failed", which is the
lifecycle the dashboard's status column already renders.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, g, jsonify, request
from werkzeug.utils import secure_filename

from auth import token_required
from db import uploads
from parser.hdfs_parser import HDFSParser

sources_bp = Blueprint("sources", __name__)

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
        "status": doc.get("status", "processing"),
        "error": doc.get("error"),
        "uploaded_at": doc["uploaded_at"].isoformat() if doc.get("uploaded_at") else None,
        "uploaded_by": doc.get("uploaded_by"),
    }


def _process_upload(upload_id, stored_path):
    """Parse the saved file and record how many log lines it yielded."""
    try:
        frame = HDFSParser().parse(str(stored_path))
        uploads.update_one(
            {"_id": upload_id},
            {"$set": {"status": "processed", "records": int(len(frame)), "processed_at": datetime.now(timezone.utc)}},
        )
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


@sources_bp.get("/api/sources/uploads")
@token_required
def list_uploads():
    try:
        limit = min(int(request.args.get("limit", 25)), 100)
    except ValueError:
        limit = 25

    rows = list(uploads.find().sort("uploaded_at", -1).limit(limit))
    return jsonify({"uploads": [_serialize(r) for r in rows], "total": uploads.count_documents({})})


@sources_bp.get("/api/sources/connectors")
@token_required
def list_connectors():
    return jsonify({"connectors": CONNECTORS})
