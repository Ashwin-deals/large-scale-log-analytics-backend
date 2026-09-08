import os
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo.errors import DuplicateKeyError, PyMongoError
from werkzeug.exceptions import HTTPException

from db import user_details
from pipeline_api import pipeline_bp
from sources_api import MAX_UPLOAD_BYTES, sources_bp

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
# Comma-separated, so a deployment can name several. The default covers the
# dev server under both spellings of the loopback address: "localhost" and
# "127.0.0.1" are different *origins* to a browser, so listing only one meant
# opening the dashboard at the other spelling had every response blocked and
# surfaced in the UI as a bare "Failed to fetch" with a healthy backend.
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGIN.split(",") if origin.strip()]
JWT_TTL_HOURS = 24

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
# allow_headers must include Authorization, or the browser's preflight blocks
# the bearer token the dashboard sends on every pipeline request.
CORS(app, origins=CORS_ORIGINS, allow_headers=["Content-Type", "Authorization"])
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.register_blueprint(pipeline_bp)
app.register_blueprint(sources_bp)

user_details.create_index("email", unique=True)


def make_token(user_id, email):
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_TTL_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        result = user_details.insert_one({
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })
    except DuplicateKeyError:
        return jsonify({"error": "An account with that email already exists."}), 409
    except PyMongoError:
        return jsonify({"error": "Database temporarily unavailable. Please try again."}), 503

    token = make_token(result.inserted_id, email)
    return jsonify({"token": token, "user": {"email": email, "name": name}}), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    try:
        user = user_details.find_one({"email": email})
    except PyMongoError:
        return jsonify({"error": "Database temporarily unavailable. Please try again."}), 503

    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401

    token = make_token(user["_id"], email)
    return jsonify({"token": token, "user": {"email": email, "name": user.get("name", "")}}), 200


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    """Return unhandled errors as JSON.

    Werkzeug's HTML debug page is returned before Flask-CORS can attach its
    headers, so a browser sees a CORS failure ("Failed to fetch") instead of
    the real error. Answering with JSON keeps the CORS headers on the response
    and lets the dashboard show what actually went wrong.
    """
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code

    app.logger.exception("Unhandled error on %s", request.path)
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    # Debug defaults off: the Werkzeug debugger exposes an interactive console
    # and full tracebacks. Opt in with FLASK_DEBUG=1 when you need it.
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    # "::" binds dual-stack (IPv6 *and* IPv4), which "0.0.0.0" does not.
    # On macOS "localhost" resolves to IPv6 ::1 before IPv4 and Vite serves
    # the dashboard from [::1]:3000, so an IPv4-only server left a browser
    # fetching http://localhost:5000 talking to nothing: connection refused,
    # surfaced in the UI as a bare "Failed to fetch" while curl against
    # 127.0.0.1 looked perfectly healthy. Binding "::" answers on
    # localhost, 127.0.0.1 and [::1] alike, so it cannot depend on which
    # address family the browser happens to pick.
    app.run(host="::", port=5000, debug=debug, threaded=True)
