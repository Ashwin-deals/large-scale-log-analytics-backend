"""Analyst feedback on alerts, and how the next retrain uses it (status doc D2).

The "positive / unlabeled data flow" box in the architecture diagram: an
analyst marks a flagged block as a real anomaly (a confirmed positive) or a
false alarm. Every verdict is stored in MongoDB (alert_feedback, one document
per source + block, latest verdict wins) and the next retrain of that source
uses it:

  - the block's label is overridden by the analyst's verdict before the GA's
    fitness sample is drawn and before the candidate is scored, so the
    fitness the GA optimizes reflects what analysts confirmed;
  - the incumbent is re-scored on the same corrected labels, so
    promote-if-better still compares like with like.

Feedback on blocks outside the labeled dataset (uploads, live stream) is
stored and counted as unlabeled positives/negatives, but cannot enter
training until those blocks have features in the dataset.

  POST /api/feedback          {"source", "block_id", "verdict": "anomaly"|"normal", "note"?}
  GET  /api/feedback?source=  verdict counts and the most recent items
  DELETE /api/feedback?source=&block_id=   withdraw a verdict
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from flask import Blueprint, g, jsonify, request
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from auth import token_required
from db import alert_feedback
from source_context import SOURCES, register_error_handlers, request_context

feedback_bp = Blueprint("feedback", __name__)
register_error_handlers(feedback_bp)

VERDICT_LABELS = {"anomaly": "Anomaly", "normal": "Normal"}
MAX_NOTE = 500


def load_feedback(source: str) -> dict[str, str]:
    """{block_id: "Anomaly"|"Normal"} for one source (empty if MongoDB is unreachable)."""
    try:
        return {doc["block_id"]: VERDICT_LABELS[doc["verdict"]]
                for doc in alert_feedback.find({"source": source}, {"block_id": 1, "verdict": 1})}
    except PyMongoError:
        return {}


def apply_feedback(merged: pd.DataFrame, feedback: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """A copy of `merged` whose Label follows the analysts' verdicts, and what changed.

    Only blocks present in `merged` can be relabeled; the rest are reported
    as unlabeled feedback."""
    out = merged.copy()
    if not feedback:
        return out, {"applied": 0, "changed": 0, "outside_dataset": 0}
    verdicts = out["block_id"].map(feedback)
    present = verdicts.notna()
    changed = present & (verdicts != out["Label"])
    out.loc[present, "Label"] = verdicts[present]
    return out, {
        "applied": int(present.sum()),
        "changed": int(changed.sum()),
        "confirmed_anomalies": int((verdicts[present] == "Anomaly").sum()),
        "marked_normal": int((verdicts[present] == "Normal").sum()),
        "outside_dataset": int(len(feedback) - present.sum()),
    }


@feedback_bp.post("/api/feedback")
@token_required
def give_feedback():
    body = request.get_json(silent=True) or {}
    ctx = request_context()
    block_id = str(body.get("block_id") or "").strip()
    verdict = str(body.get("verdict") or "").strip().lower()
    if not block_id:
        return jsonify({"error": "block_id is required"}), 400
    if verdict not in VERDICT_LABELS:
        return jsonify({"error": "verdict must be 'anomaly' or 'normal'"}), 400
    doc = {
        "source": ctx.name,
        "block_id": block_id,
        "verdict": verdict,
        "note": str(body.get("note") or "")[:MAX_NOTE],
        "by": (g.user or {}).get("email"),
        "at": datetime.now(timezone.utc),
    }
    try:
        alert_feedback.update_one({"source": ctx.name, "block_id": block_id}, {"$set": doc}, upsert=True)
    except PyMongoError as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc.__class__.__name__}"}), 503
    return jsonify({**doc, "at": doc["at"].isoformat()}), 201


@feedback_bp.delete("/api/feedback")
@token_required
def withdraw_feedback():
    ctx = request_context()
    block_id = (request.args.get("block_id") or "").strip()
    if not block_id:
        return jsonify({"error": "block_id is required"}), 400
    try:
        result = alert_feedback.delete_one({"source": ctx.name, "block_id": block_id})
    except PyMongoError as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc.__class__.__name__}"}), 503
    return jsonify({"deleted": result.deleted_count})


@feedback_bp.get("/api/feedback")
@token_required
def list_feedback():
    ctx = request_context()
    try:
        docs = list(alert_feedback.find({"source": ctx.name}, {"_id": 0}).sort("at", DESCENDING).limit(500))
    except PyMongoError as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc.__class__.__name__}"}), 503
    for doc in docs:
        if isinstance(doc.get("at"), datetime):
            doc["at"] = doc["at"].isoformat()
    counts = {verdict: sum(1 for d in docs if d["verdict"] == verdict) for verdict in VERDICT_LABELS}
    return jsonify({"source": ctx.name, "counts": counts, "total": len(docs), "items": docs,
                    "by_block": {d["block_id"]: d["verdict"] for d in docs}, "sources": list(SOURCES)})
