"""The same uploaded file must report the same anomaly count on Log Sources
(sources_api.py) and on the dashboard (pipeline_api.py), and match the batch
parse -> clean -> extract -> predict path. See upload_store.py for the bug
this guards: one file showed 1,173 on Log Sources and 120 elsewhere."""

import os
import random
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from detection.isolation_forest import save_model
from detection.predict import predict
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from parser.hdfs_parser import HDFSParser

FEATURES = ["component_encoded", "event_frequency", "destination_ip_encoded"]
EMAIL = "tester@example.com"


def hdfs_log(path: Path, n_blocks: int, seed: int):
    """Plausible HDFS lines: most blocks follow the normal lifecycle, a few are short or odd."""
    rng = random.Random(seed)
    start = datetime(2008, 11, 9, 20, 0, 0)
    lines = []
    for b in range(n_blocks):
        block = f"blk_{seed * 10000 + b}"
        t = start + timedelta(seconds=b * 7)
        ip = f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 9)}"
        steps = [
            ("dfs.FSNamesystem", f"BLOCK* NameSystem.allocateBlock: /job/{b}. {block}"),
            ("dfs.DataNode$DataXceiver", f"Receiving block {block} src: /{ip}:5000 dest: /{ip}:50010"),
            ("dfs.DataNode$PacketResponder", f"Received block {block} of size 67108864 from /{ip}"),
            ("dfs.FSNamesystem", f"BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:50010 is added to {block} size 67108864"),
            ("dfs.DataNode$PacketResponder", f"PacketResponder 1 for block {block} terminating"),
        ]
        if b % 9 == 0:  # an odd one out: truncated, with a WARN
            steps = steps[:1] + [("dfs.DataNode$DataXceiver", f"Got exception while serving {block} to /{ip}:")]
        for i, (component, message) in enumerate(steps * (3 if b % 13 == 0 else 1)):
            level = "WARN" if "exception" in message else "INFO"
            moment = t + timedelta(seconds=i)
            lines.append(f"{moment:%y%m%d %H%M%S} {100 + i} {level} {component}: {message}")
    path.write_text("\n".join(lines) + "\n")


class FakeUploads:
    def __init__(self):
        self.docs = {}

    def update_one(self, query, update):
        self.docs[query["_id"]].update(update["$set"])

    def _match(self, query):
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in query.items())]

    def find(self, query):
        rows = self._match(query)

        class Cursor(list):
            def sort(self, key, direction):
                return Cursor(sorted(self, key=lambda d: d[key], reverse=direction == -1))

            def limit(self, n):
                return Cursor(self[:n])

        return Cursor(rows)

    def count_documents(self, query):
        return len(self._match(query))

    def find_one(self, query):
        found = self._match(query)
        return found[0] if found else None


class UploadConsistencyTest(unittest.TestCase):
    def setUp(self):
        import jwt
        from flask import Flask

        import pipeline_api
        import sources_api

        self.sources_api, self.pipeline_api = sources_api, pipeline_api
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.vocab = self.tmp / "vocab.json"
        self.results = self.tmp / "results"

        # A model trained on its own vocabulary and stamped with it, as save_model() does.
        train_log = self.tmp / "train.log"
        hdfs_log(train_log, 120, seed=1)
        cleaned = HDFSDataCleaner().clean(HDFSParser().parse(str(train_log)), self.tmp / "train_cleaned.csv")
        HDFSFeatureExtractor().extract(cleaned, self.tmp / "train_features.csv", vocab_path=self.vocab)
        train = pd.read_csv(self.tmp / "train_features.csv")
        model = IsolationForest(n_estimators=50, contamination=0.15, random_state=0).fit(train[FEATURES])
        self.model_path = self.tmp / "model.pkl"
        save_model(model, self.model_path, vocab_path=self.vocab)
        self.deployment = {"version": 7, "model_path": str(self.model_path), "metrics_path": str(self.tmp / "none.json")}

        self.upload_file = self.tmp / "upload.log"
        hdfs_log(self.upload_file, 60, seed=2)

        self.uploads = FakeUploads()
        patches = [
            mock.patch.object(sources_api, "uploads", self.uploads),
            mock.patch.object(sources_api, "UPLOAD_PROCESSED_DIR", self.tmp / "processed"),
            mock.patch.object(sources_api, "UPLOAD_RESULTS_DIR", self.results),
            mock.patch.object(sources_api, "VOCAB_PATH", self.vocab),
            mock.patch.object(sources_api, "resolve_current_deployment", lambda ctx=None: dict(self.deployment)),
            mock.patch.object(pipeline_api, "uploads", self.uploads),
            mock.patch.object(pipeline_api, "UPLOAD_BLOCKS_DIR", self.results),
            mock.patch.object(pipeline_api, "resolve_current_deployment", lambda ctx=None: dict(self.deployment)),
            mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        app = Flask(__name__)
        app.register_blueprint(sources_api.sources_bp)
        app.register_blueprint(pipeline_api.pipeline_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "t", "email": EMAIL, "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def upload(self, upload_id: str, minutes: int = 0) -> dict:
        self.uploads.docs[upload_id] = {
            "_id": upload_id, "name": "upload.log", "status": "processing", "uploaded_by": EMAIL,
            "uploaded_at": datetime(2026, 11, 12, tzinfo=timezone.utc) + timedelta(minutes=minutes),
        }
        self.sources_api._process_upload(upload_id, self.upload_file)
        return self.uploads.docs[upload_id]

    def log_sources_count(self, upload_id):
        rows = self.client.get("/api/sources/uploads", headers=self.headers).get_json()["uploads"]
        return next(r for r in rows if r["id"] == upload_id)

    def batch_count(self) -> int:
        """The offline path (scripts/verify_manually.py): parse -> clean -> extract -> predict."""
        frame = HDFSParser().parse(str(self.upload_file))
        cleaned = HDFSDataCleaner().clean(frame[frame["date"].notna()].reset_index(drop=True), self.tmp / "b_cleaned.csv")
        HDFSFeatureExtractor().extract(cleaned, self.tmp / "b_features.csv", vocab_path=self.vocab)
        model = joblib.load(self.model_path)
        predictions = predict(model, pd.read_csv(self.tmp / "b_features.csv"), output_path=self.tmp / "b_pred.csv",
                              feature_columns=FEATURES, vocab_path=self.vocab)
        return int((predictions["predicted_label"] == "Anomaly").sum())

    def test_log_sources_dashboard_and_batch_agree_on_the_same_file(self):
        doc = self.upload("u1")
        self.assertEqual(doc["status"], "processed", doc.get("error"))
        listed = self.log_sources_count("u1")
        dashboard = self.client.get("/api/dashboard", headers=self.headers).get_json()
        self.assertEqual(dashboard["view"]["mode"], "upload")
        self.assertGreater(listed["anomalies_detected"], 0)
        self.assertEqual(listed["anomalies_detected"], dashboard["kpis"]["anomalies_detected"])
        self.assertEqual(listed["blocks_analyzed"], dashboard["kpis"]["total_blocks_analyzed"])
        self.assertEqual(listed["anomalies_detected"], self.batch_count())
        self.assertTrue(listed["results_available"])

    def test_uploading_the_same_file_twice_gives_the_same_count(self):
        first, second = self.upload("u1", 0), self.upload("u2", 1)
        self.assertEqual(first["anomalies_detected"], second["anomalies_detected"])
        self.assertEqual(first["scoring"], second["scoring"])

    def test_count_comes_from_the_results_file_not_the_stored_number(self):
        doc = self.upload("u1")
        real = doc["anomalies_detected"]
        doc_in_store = self.uploads.docs["u1"]
        doc_in_store["anomalies_detected"] = 1173  # what a server with a different vocabulary stored
        self.assertEqual(self.log_sources_count("u1")["anomalies_detected"], real)

    def test_upload_scored_elsewhere_shows_no_count_on_either_page(self):
        self.uploads.docs["remote"] = {"_id": "remote", "name": "upload.log", "status": "processed", "uploaded_by": EMAIL,
                                       "uploaded_at": datetime(2026, 11, 12, tzinfo=timezone.utc), "anomalies_detected": 1173,
                                       "blocks_analyzed": 5000}
        row = self.log_sources_count("remote")
        self.assertFalse(row["results_available"])
        self.assertIsNone(row["anomalies_detected"])
        self.assertIn("another server", row["results_note"])
        self.assertIsNone(self.pipeline_api.latest_upload_view(EMAIL))  # the dashboard skips it too

    def test_scoring_provenance_is_recorded(self):
        doc = self.upload("u1")
        scoring = doc["scoring"]
        self.assertEqual(scoring["model_version"], 7)
        self.assertEqual(scoring["model_encoder_fingerprint"], joblib.load(self.model_path).encoder_fingerprint_)
        self.assertEqual(len(scoring["vocab_fingerprint"]), 16)

    def test_missing_vocabulary_refuses_instead_of_renumbering(self):
        self.vocab.unlink()
        doc = self.upload("u1")
        self.assertEqual(doc["status"], "failed")
        self.assertIn("vocabulary", doc["error"])
        self.assertFalse(self.vocab.exists())  # and no fresh vocabulary was written
        self.assertIsNone(self.log_sources_count("u1")["anomalies_detected"])

    def test_model_without_an_encoder_stamp_is_refused(self):
        model = joblib.load(self.model_path)
        del model.encoder_vocab_, model.encoder_fingerprint_
        joblib.dump(model, self.model_path)
        doc = self.upload("u1")
        self.assertEqual(doc["status"], "failed")
        self.assertIn("no encoder vocabulary", doc["error"])


if __name__ == "__main__":
    unittest.main()
