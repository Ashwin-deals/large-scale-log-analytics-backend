"""Change password and profile endpoints (account_api.py), on a fake user store."""

import os
import time
import unittest
from unittest import mock

import bcrypt
import jwt
from flask import Flask

EMAIL = "someone@example.test"


class FakeUsers:
    def __init__(self, password: str):
        self.doc = {"_id": "u1", "email": EMAIL, "name": "Some One",
                    "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt(4))}

    def find_one(self, query, projection=None):
        if query.get("email") != EMAIL:
            return None
        doc = dict(self.doc)
        if projection and projection.get("password_hash") == 0:
            doc.pop("password_hash")
        return doc

    def update_one(self, query, update):
        assert query == {"_id": "u1"}
        self.doc.update(update["$set"])


class AccountApiTest(unittest.TestCase):
    def setUp(self):
        import account_api

        self.users = FakeUsers("old-password-1")
        for patch in (mock.patch.object(account_api, "user_details", self.users),
                      mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"})):
            patch.start()
            self.addCleanup(patch.stop)
        app = Flask(__name__)
        app.register_blueprint(account_api.account_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "u1", "email": EMAIL, "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def change(self, current, new):
        return self.client.post("/api/auth/change-password", json={"current_password": current, "new_password": new},
                                headers=self.headers)

    def test_changes_the_hash_after_checking_the_current_password(self):
        self.assertEqual(self.change("old-password-1", "new-password-2").status_code, 200)
        self.assertTrue(bcrypt.checkpw(b"new-password-2", self.users.doc["password_hash"]))
        self.assertFalse(bcrypt.checkpw(b"old-password-1", self.users.doc["password_hash"]))

    def test_wrong_current_password_changes_nothing(self):
        before = self.users.doc["password_hash"]
        self.assertEqual(self.change("wrong", "new-password-2").status_code, 403)
        self.assertEqual(self.users.doc["password_hash"], before)

    def test_rejects_short_or_unchanged_passwords(self):
        self.assertEqual(self.change("old-password-1", "short").status_code, 400)
        self.assertEqual(self.change("old-password-1", "old-password-1").status_code, 400)

    def test_requires_a_token(self):
        self.assertEqual(self.client.post("/api/auth/change-password", json={}).status_code, 401)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_profile_never_includes_the_hash(self):
        body = self.client.get("/api/auth/me", headers=self.headers).get_json()
        self.assertEqual(body["email"], EMAIL)
        self.assertNotIn("password_hash", body)


if __name__ == "__main__":
    unittest.main()
