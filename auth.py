"""Shared JWT verification.

Lives outside app.py so pipeline_api can guard its routes without importing
the Flask app back (which would be circular).
"""

import os
from functools import wraps

import jwt
from flask import g, jsonify, request

def _secret():
    # Read on each call, not at import: app.py imports this module (via
    # pipeline_api) *before* it calls load_dotenv(), so reading at import time
    # would pin the fallback and silently reject tokens signed with the real
    # secret from .env.
    return os.getenv("JWT_SECRET", "change-this-secret")


def decode_token(token):
    return jwt.decode(token, _secret(), algorithms=["HS256"])


def token_required(view):
    """Reject the request unless it carries a valid `Authorization: Bearer <jwt>`.

    On success the decoded claims are put on `g.user` for the view to use.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Authentication required."}), 401

        token = header.split(" ", 1)[1].strip()
        try:
            g.user = decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired. Please sign in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid session. Please sign in again."}), 401

        return view(*args, **kwargs)

    return wrapper
