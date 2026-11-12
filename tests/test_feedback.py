"""Analyst feedback (feedback_api.py) and its effect on the next retrain (D2)."""

import os
import time
import unittest
from unittest import mock

import jwt
import numpy as np
import pandas as pd
from flask import Flask

from feedback_api import apply_feedback
from optimization.ga_isolation_forest import stratified_subsample

EMAIL = "analyst@example.test"


def labeled_frame(n=400, anomalies=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["Normal"] * n, dtype=object)
    labels[rng.choice(n, anomalies, replace=False)] = "Anomaly"
    return pd.DataFrame({
        "block_id": [f"blk_{i}" for i in range(n)],
        "event_frequency": rng.integers(2, 60, n),
        "component_encoded": rng.integers(0, 5, n),
        "Label": labels,
    })


class ApplyFeedbackTest(unittest.TestCase):
    def test_verdicts_override_labels_and_report_what_changed(self):
        frame = labeled_frame()
        normal = frame.loc[frame.Label == "Normal", "block_id"].iloc[0]
        anomaly = frame.loc[frame.Label == "Anomaly", "block_id"].iloc[0]
        out, stats = apply_feedback(frame, {normal: "Anomaly", anomaly: "Anomaly", "blk_upload_9": "Normal"})
        self.assertEqual(out.set_index("block_id").loc[normal, "Label"], "Anomaly")
        self.assertEqual(stats, {"applied": 2, "changed": 1, "confirmed_anomalies": 2, "marked_normal": 0, "outside_dataset": 1})
        self.assertEqual(frame.set_index("block_id").loc[normal, "Label"], "Normal")  # input untouched

    def test_no_feedback_is_a_no_op(self):
        frame = labeled_frame()
        out, stats = apply_feedback(frame, {})
        pd.testing.assert_frame_equal(out, frame)
        self.assertEqual(stats["applied"], 0)

    def test_marking_an_alert_changes_the_next_retrains_fitness_data(self):
        # The retrain draws its GA fitness sample from the feedback-adjusted
        # frame; a block an analyst confirmed must carry its new label there.
        frame = labeled_frame(n=2000, anomalies=60)
        marked = frame.loc[frame.Label == "Normal", "block_id"].iloc[:30].tolist()
        adjusted, _ = apply_feedback(frame, {block: "Anomaly" for block in marked})

        before = stratified_subsample(frame, 2000, random_state=7)
        after = stratified_subsample(adjusted, 2000, random_state=7)
        self.assertEqual((after.Label == "Anomaly").sum() - (before.Label == "Anomaly").sum(), 30)
        self.assertTrue((after.set_index("block_id").loc[marked, "Label"] == "Anomaly").all())


class FakeCollection:
    def __init__(self):
        self.docs = []

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                doc.update(update["$set"])
                return
        self.docs.append(dict(update["$set"]))

    def delete_one(self, query):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]
        return mock.Mock(deleted_count=before - len(self.docs))

    def find(self, query, projection=None):
        rows = [dict(d) for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        cursor = mock.Mock()
        cursor.__iter__ = lambda self_: iter(rows)
        cursor.sort = lambda *a, **k: mock.Mock(limit=lambda n: rows[:n])
        return cursor


class FeedbackApiTest(unittest.TestCase):
    def setUp(self):
        import feedback_api

        self.api = feedback_api
        self.collection = FakeCollection()
        for patch in (mock.patch.object(feedback_api, "alert_feedback", self.collection),
                      mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"})):
            patch.start()
            self.addCleanup(patch.stop)
        app = Flask(__name__)
        app.register_blueprint(feedback_api.feedback_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "a", "email": EMAIL, "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def post(self, **body):
        return self.client.post("/api/feedback", json=body, headers=self.headers)

    def test_latest_verdict_wins_and_loads_for_retraining(self):
        self.assertEqual(self.post(source="django", block_id="user-7", verdict="normal").status_code, 201)
        self.assertEqual(self.post(source="django", block_id="user-7", verdict="anomaly").status_code, 201)
        self.post(source="hdfs", block_id="blk_1", verdict="normal")
        self.assertEqual(self.api.load_feedback("django"), {"user-7": "Anomaly"})
        self.assertEqual(self.api.load_feedback("hdfs"), {"blk_1": "Normal"})
        self.assertEqual(self.collection.docs[0]["by"], EMAIL)

    def test_rejects_bad_input(self):
        self.assertEqual(self.post(source="django", block_id="", verdict="normal").status_code, 400)
        self.assertEqual(self.post(source="django", block_id="x", verdict="maybe").status_code, 400)
        self.assertEqual(self.post(source="nope", block_id="x", verdict="normal").status_code, 400)

    def test_withdraw(self):
        self.post(source="django", block_id="user-7", verdict="normal")
        r = self.client.delete("/api/feedback?source=django&block_id=user-7", headers=self.headers)
        self.assertEqual(r.get_json()["deleted"], 1)
        self.assertEqual(self.api.load_feedback("django"), {})

    def test_requires_a_token(self):
        self.assertEqual(self.client.post("/api/feedback", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
