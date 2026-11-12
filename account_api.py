"""Account settings for the signed-in user, behind token_required.

  GET  /api/auth/me                profile (email, name, created_at; never the hash)
  POST /api/auth/change-password   re-checks the current password, then stores a new bcrypt hash
"""

from datetime import datetime, timezone

import bcrypt
from flask import Blueprint, g, jsonify, request
from pymongo.errors import PyMongoError

from auth import token_required
from db import user_details

account_bp = Blueprint("account", __name__)


@account_bp.post("/api/auth/change-password")
@token_required
def change_password():
    """Body: {"current_password": ..., "new_password": ...}. The signed-in
    account only; the current password is re-checked before anything changes."""
    data = request.get_json(silent=True) or {}
    current = data.get("current_password") or ""
    new = data.get("new_password") or ""
    email = (g.user or {}).get("email")
    if len(new) < 8:
        return jsonify({"error": "New password must be at least 8 characters."}), 400
    if new == current:
        return jsonify({"error": "New password must differ from the current one."}), 400
    try:
        user = user_details.find_one({"email": email})
    except PyMongoError:
        return jsonify({"error": "Database temporarily unavailable. Please try again."}), 503
    if not user or not bcrypt.checkpw(current.encode("utf-8"), user["password_hash"]):
        return jsonify({"error": "Current password is incorrect."}), 403
    try:
        user_details.update_one(
            {"_id": user["_id"]},
            {"$set": {"password_hash": bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt()),
                      "password_changed_at": datetime.now(timezone.utc)}},
        )
    except PyMongoError:
        return jsonify({"error": "Database temporarily unavailable. Please try again."}), 503
    return jsonify({"ok": True})


@account_bp.get("/api/auth/me")
@token_required
def me():
    """The signed-in account's profile (never the password hash)."""
    email = (g.user or {}).get("email")
    try:
        user = user_details.find_one({"email": email}, {"password_hash": 0})
    except PyMongoError:
        return jsonify({"error": "Database temporarily unavailable. Please try again."}), 503
    if not user:
        return jsonify({"email": email, "name": "", "created_at": None, "registered": False})
    created = user.get("created_at")
    return jsonify({"email": user["email"], "name": user.get("name", ""), "registered": True,
                    "created_at": created.isoformat() if created else None})
