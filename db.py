"""Single MongoDB connection shared by the auth and sources APIs.

load_dotenv() runs here rather than relying on app.py, because app.py imports
the blueprints (and therefore this module) before it loads the environment.
Reading the env without this would silently fall back to localhost instead of
the configured Atlas cluster.
"""

import os

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "users")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "user_details")

# tlsCAFile pins certifi's bundle so Atlas connections work on machines whose
# Python has no system root certificates installed (common on macOS).
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)
db = client[MONGO_DB]

user_details = db[MONGO_COLLECTION]
uploads = db["uploads"]
