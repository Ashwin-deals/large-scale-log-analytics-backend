import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from early_warning.prefix_features import build_prefix_features
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import FEATURE_COLUMNS, HDFSFeatureExtractor
from parser.hdfs_parser import HDFSParser
from smart_storage.policy import BlockTable, Normalizer, StoragePolicy, evaluate_policy
from streaming.block_state import BlockState, StreamEncoder, parse_event
from streaming.processor import StreamProcessor
from streaming.producer import RateLimiter, block_key

# Raw HDFS lines: interleaved blocks, a WARN line, sizes, a duplicate line,
# an unparseable line, and a block id with no other lines.
RAW_LINES = [
    "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1 src: /10.0.0.1:54106 dest: /10.0.0.1:50010",
    "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/job.jar. blk_-1",
    "081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_22 src: /10.0.0.2:40524 dest: /10.0.0.2:50010",
    "081109 203521 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1 of size 67108864 from /10.0.0.1",
    "081109 203521 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1 of size 67108864 from /10.0.0.1",  # duplicate
    "081109 203522 146 WARN dfs.DataNode$DataXceiver: Got exception while serving blk_22 to /10.0.0.3:",
    "081109 203523 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.0.0.1:50010 is added to blk_-1 size 67108864",
    "this line does not match the HDFS format at all",
    "081109 203524 147 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1 terminating",
    "081109 203530 148 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/other. blk_333",
    "081109 203531 149 INFO dfs.DataNode$PacketResponder: Received block blk_22 of size 1024 from /10.0.0.2",
    "081109 203540 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: blk_-1 is added to invalidSet of 10.0.0.1:50010",
]


def stream_states(lines, vocab_path):
    """Run lines through the streaming path; returns ({block: BlockState}, {k: {block: prefix features}}, encoder)."""
    parser, encoder = HDFSParser(), StreamEncoder(vocab_path)
    states, prefixes = {}, {2: {}, 3: {}}
    for line in lines:
        event = parse_event(parser, line)
        if event is None:
            continue
        state = states.setdefault(event["block_id"], BlockState(event["block_id"]))
        if state.add(event, encoder) and state.n in prefixes:
            prefixes[state.n][state.block_id] = state.prefix_features(encoder.event_type_names(), encoder)
    for k, captured in prefixes.items():  # a block shorter than k is described by all its events
        for block_id, state in states.items():
            captured.setdefault(block_id, state.prefix_features(encoder.event_type_names(), encoder))
    return states, prefixes, encoder


class BatchParityTest(unittest.TestCase):
    """The streaming state must reproduce the batch builders exactly, or the
    batch-trained models would be scoring different features."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        raw, cls.vocab = tmp / "raw.log", tmp / "vocab.json"
        raw.write_text("\n".join(RAW_LINES) + "\n")
        structured = HDFSParser().parse(str(raw))
        cleaned = HDFSDataCleaner().clean(structured, output_path=tmp / "cleaned.csv")
        cls.n_cleaned = len(cleaned)
        HDFSFeatureExtractor().extract(tmp / "cleaned.csv", output_path=tmp / "features.csv", vocab_path=cls.vocab)
        cls.batch_full = pd.read_csv(tmp / "features.csv").set_index("block_id")
        frames, _ = build_prefix_features(tmp / "cleaned.csv", ks=(2, 3), vocab_path=cls.vocab)
        cls.batch_prefix = {k: f.set_index("block_id") for k, f in frames.items()}
        cls.states, cls.prefixes, cls.encoder = stream_states(RAW_LINES, cls.vocab)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_lines_survive_cleaning(self):
        self.assertEqual(sum(s.n for s in self.states.values()), self.n_cleaned)
        self.assertEqual(set(self.states), set(self.batch_full.index))

    def test_full_block_features_match_the_feature_extractor(self):
        for block_id, state in self.states.items():
            streamed = state.full_features()
            for column in FEATURE_COLUMNS:
                self.assertEqual(streamed[column], self.batch_full.loc[block_id, column], f"{block_id} {column}")

    def test_prefix_features_match_the_early_warning_builder(self):
        for k in (2, 3):
            batch = self.batch_prefix[k]
            for block_id, streamed in self.prefixes[k].items():
                for column, value in streamed.items():
                    self.assertEqual(value, batch.loc[block_id, column], f"k={k} {block_id} {column}")

    def test_stream_never_writes_the_vocabulary(self):
        before = self.vocab.read_text()
        encoder = StreamEncoder(self.vocab)
        code = encoder.code("component", "dfs.Never.Seen")
        self.assertEqual(code, max(json.loads(before)["component"].values()) + 1)
        self.assertEqual(encoder.unseen["component"], 1)
        self.assertEqual(self.vocab.read_text(), before)


class ParseEventTest(unittest.TestCase):
    def test_drops_lines_the_cleaner_drops(self):
        parser = HDFSParser()
        self.assertIsNone(parse_event(parser, "not a log line"))
        self.assertIsNone(parse_event(parser, "081109 203518 35 INFO dfs.FSNamesystem: no block id here"))
        event = parse_event(parser, RAW_LINES[0])
        self.assertEqual((event["block_id"], event["source_ip"], event["log_level"]), ("blk_-1", "10.0.0.1", "INFO"))
        self.assertEqual(event["seconds"] % 86400, 20 * 3600 + 35 * 60 + 18)

    def test_missing_ips_become_unknown_and_size_defaults_to_zero(self):
        event = parse_event(HDFSParser(), RAW_LINES[1])
        self.assertEqual((event["source_ip"], event["destination_ip"], event["block_size"]), ("UNKNOWN", "UNKNOWN", 0.0))


class FakeEarly:
    def __init__(self, threshold, score_fn):
        self.threshold, self.score_fn = threshold, score_fn

    def scores(self, rows):
        return np.array([self.score_fn(r) for r in rows], dtype=float)


class FakeScorer:
    """Early score = warn_count; full verdict Anomaly when any warn line."""

    deployed_version = 9

    def __init__(self, ks=(2, 3), threshold=1.0):
        self.early = {k: FakeEarly(threshold, lambda row: row["warn_count"]) for k in ks}
        self.full_calls = []

    def full(self, rows):
        self.full_calls.append(len(rows))
        anomalous = np.array([r["event_frequency"] >= 3 for r in rows])
        return {
            "predicted_label": np.where(anomalous, "Anomaly", "Normal"),
            "anomaly_score": anomalous.astype(float),
            "importance": anomalous.astype(float),
            "tier": ["HOT" if a else "COLD" for a in anomalous],
            "lines_kept": np.array([r["event_frequency"] if a else 1 for r, a in zip(rows, anomalous)]),
        }


def hdfs_line(seconds_after: int, block: str, level="INFO", pid=1):
    stamp = datetime(2008, 11, 9, 20, 0, 0) + timedelta(seconds=seconds_after)
    return f"{stamp:%y%m%d %H%M%S} {pid} {level} dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /x. {block}"


class StreamProcessorTest(unittest.TestCase):
    def make(self, ks=(2, 3), timeout=100, threshold=1.0):
        vocab = Path(tempfile.mkdtemp()) / "vocab.json"
        return StreamProcessor(FakeScorer(ks, threshold), ks=ks, idle_timeout_seconds=timeout, run_id="test-run",
                               encoder=StreamEncoder(vocab))

    def test_early_warning_fires_once_at_the_first_k_that_scores_above_threshold(self):
        p = self.make()
        alerts, _ = p.process([hdfs_line(0, "blk_1", "WARN", pid=1), hdfs_line(1, "blk_1", pid=2)])
        self.assertEqual([(a["type"], a["k"], a["events_seen"]) for a in alerts], [("early_warning", 2, 2)])
        alerts, _ = p.process([hdfs_line(2, "blk_1", pid=3)])  # reaches k=3 too, but already alerted
        self.assertEqual(alerts, [])
        self.assertEqual(p.counts["early_warning_alerts"], 1)
        self.assertEqual(alerts, [])

    def test_prefix_is_snapshotted_at_the_kth_event_even_mid_batch(self):
        # One batch: k=2 is reached at the 2nd line; the WARN lines after it
        # must not leak into the k=2 features (they would push the score over).
        p = self.make(ks=(2,))
        lines = [hdfs_line(0, "blk_7", pid=1), hdfs_line(1, "blk_7", pid=2)] + [hdfs_line(2 + i, "blk_7", "WARN", pid=3 + i) for i in range(5)]
        alerts, _ = p.process(lines)
        self.assertEqual(alerts, [])  # warn_count was 0 at the 2nd event
        self.assertEqual(p.blocks["blk_7"].n, 7)

        q = self.make(ks=(2,))
        alerts, _ = q.process([hdfs_line(0, "blk_8", "WARN", pid=1), hdfs_line(1, "blk_8", pid=2), hdfs_line(9, "blk_8", pid=3)])
        self.assertEqual((alerts[0]["events_seen"], alerts[0]["event_time"]), (2, "2008-11-09T20:00:01"))

    def test_no_alert_below_threshold(self):
        p = self.make()
        alerts, _ = p.process([hdfs_line(i, "blk_1", pid=i) for i in range(3)])
        self.assertEqual(alerts, [])

    def test_alert_ids_are_deterministic_per_run_type_block(self):
        p = self.make()
        alerts, _ = p.process([hdfs_line(0, "blk_9", "WARN", pid=1), hdfs_line(1, "blk_9", pid=2)])
        self.assertEqual(alerts[0]["_id"], "test-run:early_warning:blk_9")

    def test_quiet_blocks_time_out_on_log_time_and_late_lines_are_ignored(self):
        p = self.make(timeout=100)
        p.process([hdfs_line(0, "blk_201", pid=1), hdfs_line(1, "blk_201", pid=2), hdfs_line(2, "blk_201", pid=3)])
        _, decisions = p.process([hdfs_line(50, "blk_202", pid=4)])
        self.assertEqual(decisions, [])  # 50s of log time < 100s timeout
        alerts, decisions = p.process([hdfs_line(200, "blk_202", pid=5)])
        self.assertEqual([d["block_id"] for d in decisions], ["blk_201"])
        self.assertEqual(decisions[0]["reason"], "timeout")
        self.assertEqual(decisions[0]["events"], 3)
        self.assertEqual([a["type"] for a in alerts], ["block_anomaly"])  # FakeScorer: >= 3 events
        self.assertEqual(alerts[0]["finalize_reason"], "timeout")
        p.process([hdfs_line(201, "blk_201", pid=6)])
        self.assertEqual(p.counts["late_events"], 1)
        self.assertNotIn("blk_201", p.blocks)

    def test_flush_all_finalizes_everything_in_flight(self):
        p = self.make()
        p.process([hdfs_line(0, "blk_101", pid=1), hdfs_line(1, "blk_102", pid=2)])
        _, decisions = p.flush_all()
        self.assertEqual(sorted(d["block_id"] for d in decisions), ["blk_101", "blk_102"])
        self.assertEqual({d["reason"] for d in decisions}, {"end_of_stream"})
        self.assertEqual(p.blocks, {})
        self.assertEqual(p.scorer.full_calls, [2])  # one batched scoring call, not one per block

    def test_duplicates_and_unparsed_lines_are_counted_not_processed(self):
        p = self.make()
        line = hdfs_line(0, "blk_101")
        p.process([line, line, "garbage"])
        self.assertEqual((p.counts["events"], p.counts["duplicate_lines"], p.counts["lines_dropped_unparsed"]), (1, 1, 1))
        self.assertEqual(p.counts["lines_received"], 3)

    def test_status_reports_live_counts_and_storage_saved(self):
        clock = {"t": 100.0}
        p = self.make()
        p.clock = lambda: clock["t"]
        p.process([hdfs_line(i, "blk_301", pid=i) for i in range(4)] + [hdfs_line(0, "blk_302", pid=9)])
        clock["t"] = 102.0
        p.flush_all()
        status = p.status()
        self.assertEqual(status["blocks_in_flight"], 0)
        self.assertEqual(status["storage"]["lines_seen"], 5)
        self.assertEqual(status["storage"]["lines_kept"], 4 + 1)  # big HOT keeps 4, small COLD keeps 1
        self.assertEqual(status["storage"]["tier_counts"], {"HOT": 1, "WARM": 0, "COLD": 1})
        self.assertEqual(status["alerts"]["block_anomaly"], 1)
        self.assertEqual(status["events_per_sec"], 2.5)  # 5 lines over a 2s window
        json.dumps(status)

    def test_rejects_k_without_a_model(self):
        with self.assertRaises(ValueError):
            StreamProcessor(FakeScorer(ks=(3,)), ks=(3, 5), encoder=StreamEncoder(Path(tempfile.mkdtemp()) / "v.json"))


class ScorerTest(unittest.TestCase):
    """Real (tiny) models through streaming.scoring.Scorer."""

    def test_full_scores_and_storage_tiers_match_the_batch_policy(self):
        from streaming.scoring import Scorer

        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        n = 400
        full = pd.DataFrame({column: rng.integers(0, 20, n) for column in FEATURE_COLUMNS})
        full["block_size"] = rng.integers(0, 10**6, n).astype(float)
        full["is_error"] = 0
        model = IsolationForest(n_estimators=20, random_state=0).fit(full[["component_encoded", "event_frequency"]])
        joblib.dump(model, tmp / "deployed.pkl")
        joblib.dump(model, tmp / "storage.pkl")

        prefix = pd.DataFrame({"n_events": rng.integers(1, 4, n), "warn_count": rng.integers(0, 2, n)})
        early = IsolationForest(n_estimators=10, random_state=0).fit(prefix)
        early.alert_threshold_ = 0.5
        (tmp / "ew").mkdir()
        joblib.dump(early, tmp / "ew" / "isolation_forest_k3.pkl")

        storage_scores = -model.decision_function(full[["component_encoded", "event_frequency"]])
        normalizer = Normalizer.fit(storage_scores, full["event_frequency"], full["block_size"])
        policy = StoragePolicy(0.6, 0.2, 0.2, 0.5, 0.3, 0.2)
        (tmp / "policy.json").write_text(json.dumps({
            "policy": policy.to_dict(), "normalizer": normalizer.to_dict(), "model": {"model_path": str(tmp / "storage.pkl")},
        }))

        scorer = Scorer([3], early_warning_dir=tmp / "ew", deployed_model_path=tmp / "deployed.pkl", policy_path=tmp / "policy.json")
        self.assertEqual(scorer.early[3].threshold, 0.5)
        self.assertEqual(scorer.deployed.n_jobs, 1)
        out = scorer.full(full.to_dict("records"))

        expected_labels = np.where(model.predict(full[["component_encoded", "event_frequency"]]) == -1, "Anomaly", "Normal")
        np.testing.assert_array_equal(out["predicted_label"], expected_labels)

        # Per-block lines kept must sum to what evaluate_policy charges for the same blocks.
        frame = full.assign(anomaly_score=storage_scores, true_label="Normal", predicted_label="Normal")
        batch = evaluate_policy(BlockTable.from_frame(frame, normalizer), policy)
        self.assertEqual(int(out["lines_kept"].sum()), batch.kept_lines)
        self.assertEqual({t: out["tier"].count(t) for t in ("HOT", "WARM", "COLD")}, batch.tier_counts)


class ProducerTest(unittest.TestCase):
    def test_block_key_is_the_first_block_id(self):
        self.assertEqual(block_key(RAW_LINES[0]), b"blk_-1")
        self.assertIsNone(block_key("no block here"))

    def test_rate_limiter_paces_against_the_start_time(self):
        now = {"t": 0.0}
        slept = []
        limiter = RateLimiter(1000, clock=lambda: now["t"], sleep=lambda s: (slept.append(s), now.update(t=now["t"] + s)))
        limiter.wait(500)  # 500 sent at t=0 -> half a second ahead
        self.assertAlmostEqual(slept[-1], 0.5)
        now["t"] = 2.0
        limiter.wait(1000)  # behind schedule: no sleep
        self.assertEqual(len(slept), 1)
        RateLimiter(0, clock=lambda: 0.0, sleep=lambda s: slept.append(s)).wait(10**6)
        self.assertEqual(len(slept), 1)  # rate 0 = unthrottled


class FakeCursor(list):
    def sort(self, key, direction):
        return FakeCursor(sorted(self, key=lambda d: d[key], reverse=direction == -1))

    def skip(self, n):
        return FakeCursor(self[n:])

    def limit(self, n):
        return FakeCursor(self[:n])


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def _match(self, query):
        return [dict(d) for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    def find_one(self, query, sort=None):
        found = self._match(query)
        if sort:
            key, direction = sort[0]
            found.sort(key=lambda d: d[key], reverse=direction == -1)
        return found[0] if found else None

    def find(self, query):
        return FakeCursor(self._match(query))

    def count_documents(self, query):
        return len(self._match(query))


class MongoSinkLeadTest(unittest.TestCase):
    def test_lead_is_written_onto_the_early_warning_alert_when_its_block_finalizes(self):
        from streaming.processor import MongoSink

        class Collection:
            def __init__(self):
                self.ops = []

            def create_index(self, *args, **kwargs):
                pass

            def bulk_write(self, ops, ordered=True):
                self.ops.extend(ops)

        alerts = Collection()
        sink = MongoSink(alerts, Collection())
        sink.record_leads([
            {"run_id": "r", "block_id": "blk_1", "early_warning_k": 20, "early_warning_lead": 0.6, "events": 50, "reason": "timeout"},
            {"run_id": "r", "block_id": "blk_2", "early_warning_k": None, "early_warning_lead": None, "events": 5, "reason": "timeout"},
        ])
        self.assertEqual(len(alerts.ops), 1)
        self.assertEqual(alerts.ops[0]._filter, {"_id": "r:early_warning:blk_1"})
        self.assertEqual(alerts.ops[0]._doc["$set"]["lead_fraction"], 0.6)


class StreamApiTest(unittest.TestCase):
    def setUp(self):
        import jwt
        from flask import Flask

        import stream_api

        now = datetime.now(timezone.utc)
        self.status_docs = [
            {"_id": "old", "run_id": "old", "updated_at": (now - timedelta(hours=1)).isoformat(), "stopped_at": "x", "events_per_sec": 0.0},
            {"_id": "new", "run_id": "new", "updated_at": now.isoformat(), "events_per_sec": 4200.0, "blocks_in_flight": 12},
        ]
        alert = lambda i, run, kind: {"_id": f"{run}:{kind}:blk_{i}", "run_id": run, "type": kind, "block_id": f"blk_{i}",
                                      "detected_at": (now + timedelta(seconds=i)).isoformat()}
        self.alert_docs = [alert(i, "new", "early_warning") for i in range(3)] + [alert(9, "new", "block_anomaly"), alert(1, "old", "early_warning")]
        for patch in (mock.patch.object(stream_api, "stream_status", FakeCollection(self.status_docs)),
                      mock.patch.object(stream_api, "stream_alerts", FakeCollection(self.alert_docs)),
                      mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"})):
            patch.start()
            self.addCleanup(patch.stop)
        app = Flask(__name__)
        app.register_blueprint(stream_api.stream_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "t", "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def get(self, path):
        return self.client.get(path, headers=self.headers)

    def test_requires_a_token(self):
        self.assertEqual(self.client.get("/api/stream/status").status_code, 401)
        self.assertEqual(self.client.get("/api/stream/alerts").status_code, 401)

    def test_status_is_the_newest_run_and_live_when_fresh(self):
        body = self.get("/api/stream/status").get_json()
        self.assertEqual((body["run_id"], body["events_per_sec"], body["live"]), ("new", 4200.0, True))
        old = self.get("/api/stream/status?run_id=old").get_json()
        self.assertFalse(old["live"])  # stopped and stale

    def test_status_404_before_any_processor_reported(self):
        self.status_docs.clear()
        self.assertEqual(self.get("/api/stream/status").status_code, 404)

    def test_alerts_default_to_the_newest_run_newest_first(self):
        body = self.get("/api/stream/alerts").get_json()
        self.assertEqual(body["run_id"], "new")
        self.assertEqual(body["total"], 4)
        self.assertEqual([a["block_id"] for a in body["items"]], ["blk_9", "blk_2", "blk_1", "blk_0"])
        self.assertEqual(body["items"][0]["id"], "new:block_anomaly:blk_9")

    def test_alerts_filter_page_and_validate(self):
        body = self.get("/api/stream/alerts?type=early_warning&limit=2&page=2").get_json()
        self.assertEqual((body["total"], [a["block_id"] for a in body["items"]]), (3, ["blk_0"]))
        self.assertEqual(self.get("/api/stream/alerts?type=nope").status_code, 400)
        self.assertEqual(self.get("/api/stream/alerts?run_id=old").get_json()["total"], 1)


if __name__ == "__main__":
    unittest.main()
