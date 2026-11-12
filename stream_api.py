"""Flask Blueprint for the real-time layer, every route behind token_required.

  GET /api/stream/status   the newest processor run's live counters
  GET /api/stream/alerts   alerts from stream_alerts, newest first

The processor (python3 -m streaming.processor) upserts its status into
stream_status every ~2 s; `live` is false once that document is older than
STALE_AFTER_SECONDS, i.e. the processor has stopped or is stuck.
"""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from auth import token_required
from db import stream_alerts, stream_status

stream_bp = Blueprint("stream", __name__)

STALE_AFTER_SECONDS = 10
DEFAULT_ALERT_LIMIT = 50
MAX_ALERT_LIMIT = 500
ALERT_TYPES = ("early_warning", "block_anomaly")


def _int_arg(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


def _latest_status(run_id: str | None):
    query = {"_id": run_id} if run_id else {}
    return stream_status.find_one(query, sort=[("updated_at", DESCENDING)])


@stream_bp.route("/api/stream/status", methods=["GET"])
@token_required
def get_status():
    try:
        status = _latest_status(request.args.get("run_id"))
    except PyMongoError as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc.__class__.__name__}"}), 503
    if status is None:
        return jsonify({"live": False, "error": "No stream processor has reported yet. Start it with `python3 -m streaming.processor`."}), 404

    updated = datetime.fromisoformat(status["updated_at"])
    age = (datetime.now(timezone.utc) - updated).total_seconds()
    status["live"] = age <= STALE_AFTER_SECONDS and "stopped_at" not in status
    status["seconds_since_update"] = round(age, 1)
    status.pop("_id", None)
    return jsonify(status)


@stream_bp.route("/api/stream/alerts", methods=["GET"])
@token_required
def get_alerts():
    """?limit=50&page=1&type=early_warning|block_anomaly&run_id=... (default: newest run).

    An early_warning item gains lead_fraction (share of its block still to
    come when it fired) and events_total once the processor finalizes the block;
    until then the block is still running and both are absent."""
    limit = _int_arg("limit", DEFAULT_ALERT_LIMIT, 1, MAX_ALERT_LIMIT)
    page = _int_arg("page", 1, 1, 10**6)
    alert_type = request.args.get("type")
    if alert_type and alert_type not in ALERT_TYPES:
        return jsonify({"error": f"type must be one of {', '.join(ALERT_TYPES)}"}), 400
    try:
        run_id = request.args.get("run_id")
        if not run_id:
            latest = _latest_status(None)
            run_id = latest["run_id"] if latest else None
        query = {"run_id": run_id} if run_id else {}
        if alert_type:
            query["type"] = alert_type
        total = stream_alerts.count_documents(query)
        cursor = stream_alerts.find(query).sort("detected_at", DESCENDING).skip((page - 1) * limit).limit(limit)
        items = [{**doc, "id": doc.pop("_id")} for doc in cursor]
    except PyMongoError as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc.__class__.__name__}"}), 503
    return jsonify({"run_id": run_id, "total": total, "page": page, "limit": limit, "items": items})
