import os
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import certifi
import jwt
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from pipeline_api import pipeline_bp

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "users")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "user_details")
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000")
JWT_TTL_HOURS = 24

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
CORS(app, origins=[CORS_ORIGIN])
app.register_blueprint(pipeline_bp)

# tlsCAFile pins certifi's bundle so Atlas connections work on machines whose
# Python has no system root certificates installed (common on macOS).
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)
db = client[MONGO_DB]
user_details = db[MONGO_COLLECTION]
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


if __name__ == "__main__":
    app.run(port=5000, debug=True, threaded=True)
