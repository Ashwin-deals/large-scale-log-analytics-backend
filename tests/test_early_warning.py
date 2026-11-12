import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from early_warning.model import (
    alert_metrics,
    lead_fraction,
    lead_seconds,
    stratified_split,
    trade_off,
    train_and_evaluate_k,
    tune_threshold,
)
from early_warning.prefix_features import (
    BLOCK_END_COLUMN,
    BOOKKEEPING_COLUMNS,
    MISSING_FILE_HELP,
    PREFIX_END_COLUMN,
    TOTAL_EVENTS_COLUMN,
    build_prefix_features,
    feature_columns,
    require_cleaned_log,
)

CLEANED_COLUMNS = [
    "date", "time", "pid", "log_level", "component", "event_type", "block_id",
    "source_ip", "destination_ip", "block_size", "raw_message", "datetime",
]
VOCAB = {
    "component": {"dfs.DataNode$PacketResponder": 0, "dfs.FSNamesystem": 1, "dfs.DataNode$DataXceiver": 2},
    "event_type": {"RECEIVED_BLOCK": 0, "ALLOCATE_BLOCK": 1, "OTHER": 2, "RECEIVE_BLOCK": 3},
}


def write_cleaned(path, events):
    """events: (block_id, seconds, log_level, component, event_type, block_size) in file order."""
    base = pd.Timestamp("2008-11-09 20:00:00")
    rows = [
        {
            "date": "2008-11-09", "time": "", "pid": 1, "log_level": level, "component": component,
            "event_type": event_type, "block_id": block_id, "source_ip": "UNKNOWN", "destination_ip": "UNKNOWN",
            "block_size": block_size, "raw_message": "m",
            "datetime": (base + pd.Timedelta(seconds=int(seconds))).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for block_id, seconds, level, component, event_type, block_size in events
    ]
    pd.DataFrame(rows, columns=CLEANED_COLUMNS).to_csv(path, index=False)


NS, DX, PR = "dfs.FSNamesystem", "dfs.DataNode$DataXceiver", "dfs.DataNode$PacketResponder"

# Block a: 5 events, interleaved with b (3 events) and c (1 event).
EVENTS = [
    ("blk_a", 0, "INFO", NS, "ALLOCATE_BLOCK", 0.0),
    ("blk_b", 1, "INFO", NS, "ALLOCATE_BLOCK", 0.0),
    ("blk_a", 2, "INFO", DX, "RECEIVE_BLOCK", 0.0),
    ("blk_c", 3, "INFO", NS, "OTHER", 0.0),
    ("blk_a", 4, "WARN", DX, "RECEIVE_BLOCK", 0.0),
    ("blk_b", 6, "WARN", PR, "RECEIVED_BLOCK", 64.0),
    ("blk_a", 9, "INFO", PR, "RECEIVED_BLOCK", 128.0),
    ("blk_b", 10, "INFO", NS, "OTHER", 32.0),
    ("blk_a", 20, "INFO", NS, "OTHER", 256.0),
]


class PrefixFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.cleaned = self.dir / "hdfs_cleaned.csv"
        self.vocab = self.dir / "vocab.json"
        self.vocab.write_text(json.dumps(VOCAB))
        write_cleaned(self.cleaned, EVENTS)

    def build(self, ks=(2, 3), chunksize=1000):
        frames, stats = build_prefix_features(self.cleaned, ks, chunksize=chunksize, vocab_path=self.vocab)
        return {k: f.set_index("block_id") for k, f in frames.items()}, stats

    def test_only_the_first_k_events_count(self):
        frames, _ = self.build(ks=(3,))
        a = frames[3].loc["blk_a"]
        # First 3 events of a: ALLOCATE(t0, NS), RECEIVE(t2, DX), RECEIVE(t4, DX, WARN).
        self.assertEqual(a["n_events"], 3)
        self.assertEqual(a["elapsed_seconds"], 4)
        self.assertEqual(a["count_ALLOCATE_BLOCK"], 1)
        self.assertEqual(a["count_RECEIVE_BLOCK"], 2)
        self.assertEqual(a["count_RECEIVED_BLOCK"], 0)  # 4th event, not seen yet
        self.assertEqual(a["count_OTHER"], 0)
        self.assertEqual(a["distinct_event_types"], 2)
        self.assertEqual(a["distinct_components"], 2)
        self.assertEqual(a["warn_count"], 1)
        self.assertEqual(a["block_size_max"], 0.0)  # the 128/256 sizes come later
        self.assertEqual(a[TOTAL_EVENTS_COLUMN], 5)

    def test_encoded_columns_use_the_persisted_vocabulary(self):
        frames, _ = self.build(ks=(3,))
        a = frames[3].loc["blk_a"]
        self.assertEqual(a["first_component_encoded"], VOCAB["component"][NS])
        self.assertEqual(a["first_event_type_encoded"], VOCAB["event_type"]["ALLOCATE_BLOCK"])
        self.assertEqual(a["dominant_component_encoded"], VOCAB["component"][DX])
        self.assertEqual(a["dominant_event_type_encoded"], VOCAB["event_type"]["RECEIVE_BLOCK"])

    def test_alert_time_and_block_end_are_recorded(self):
        frames, _ = self.build(ks=(2, 3))
        base = pd.Timestamp("2008-11-09 20:00:00").value // 10**9
        a3, b2, c3 = frames[3].loc["blk_a"], frames[2].loc["blk_b"], frames[3].loc["blk_c"]
        # blk_a: 3rd event at t=4, last event at t=20.
        self.assertEqual((a3[PREFIX_END_COLUMN], a3[BLOCK_END_COLUMN]), (base + 4, base + 20))
        # blk_b: 2nd event at t=6, last at t=10.
        self.assertEqual((b2[PREFIX_END_COLUMN], b2[BLOCK_END_COLUMN]), (base + 6, base + 10))
        # A block shorter than k ends where its prefix ends.
        self.assertEqual(c3[PREFIX_END_COLUMN], c3[BLOCK_END_COLUMN])
        np.testing.assert_array_equal(lead_seconds(frames[3].loc[["blk_a", "blk_c"]]), [16.0, 0.0])

    def test_time_columns_are_never_model_features(self):
        columns = feature_columns(list(VOCAB["event_type"]))
        for bookkeeping in BOOKKEEPING_COLUMNS:
            self.assertNotIn(bookkeeping, columns)

    def test_dominant_ties_go_to_the_smallest_code(self):
        frames, _ = self.build(ks=(2,))
        # b's first two events: ALLOCATE (code 1) and RECEIVED_BLOCK (code 0), one each.
        self.assertEqual(frames[2].loc["blk_b", "dominant_event_type_encoded"], 0)

    def test_short_blocks_use_all_their_events(self):
        frames, _ = self.build(ks=(3,))
        c = frames[3].loc["blk_c"]
        self.assertEqual((c["n_events"], c[TOTAL_EVENTS_COLUMN], c["elapsed_seconds"]), (1, 1, 0))

    def test_block_size_so_far_is_the_running_max(self):
        frames, _ = self.build(ks=(2, 3))
        self.assertEqual(frames[2].loc["blk_b", "block_size_max"], 64.0)
        self.assertEqual(frames[3].loc["blk_b", "block_size_max"], 64.0)  # 32 later does not lower it

    def test_results_do_not_depend_on_chunk_size(self):
        whole, _ = self.build(ks=(2, 3), chunksize=1000)
        for chunksize in (1, 2, 4):
            chunked, stats = self.build(ks=(2, 3), chunksize=chunksize)
            self.assertGreater(stats["chunks"], 1)
            for k in (2, 3):
                pd.testing.assert_frame_equal(whole[k], chunked[k])

    def test_columns_and_block_order_are_stable(self):
        frames, stats = build_prefix_features(self.cleaned, (2, 3), vocab_path=self.vocab)
        expected = ["block_id"] + feature_columns(list(VOCAB["event_type"])) + BOOKKEEPING_COLUMNS
        self.assertEqual(list(frames[2].columns), expected)
        self.assertEqual(list(frames[2]["block_id"]), ["blk_a", "blk_b", "blk_c"])  # first-appearance order
        self.assertTrue(frames[2]["block_id"].equals(frames[3]["block_id"]))
        self.assertEqual((stats["rows"], stats["blocks"], stats["events"]), (9, 3, 9))

    def test_unseen_categories_are_appended_and_saved(self):
        write_cleaned(self.cleaned, EVENTS + [("blk_d", 30, "INFO", "dfs.Brand.New", "NEW_EVENT", 0.0)])
        frames, _ = self.build(ks=(2,))
        saved = json.loads(self.vocab.read_text())
        self.assertEqual(saved["event_type"]["NEW_EVENT"], 4)  # after the highest existing code
        self.assertEqual(saved["component"]["dfs.Brand.New"], 3)
        for name, code in VOCAB["event_type"].items():
            self.assertEqual(saved["event_type"][name], code)  # existing codes never move
        self.assertIn("count_NEW_EVENT", frames[2].columns)
        self.assertEqual(frames[2].loc["blk_d", "first_event_type_encoded"], 4)

    def test_vocab_file_is_untouched_when_nothing_is_new(self):
        before = self.vocab.read_text()
        self.build()
        self.assertEqual(self.vocab.read_text(), before)

    def test_missing_categories_encode_as_unknown(self):
        write_cleaned(self.cleaned, EVENTS + [("blk_d", 30, "INFO", None, "OTHER", 0.0)])
        frames, _ = self.build(ks=(2,))
        saved = json.loads(self.vocab.read_text())
        self.assertEqual(frames[2].loc["blk_d", "first_component_encoded"], saved["component"]["UNKNOWN"])
        self.assertNotIn("UNKNOWN", saved["event_type"])  # nothing was missing there

    def test_out_of_order_timestamps_are_counted(self):
        write_cleaned(self.cleaned, [("blk_a", 5, "INFO", NS, "OTHER", 0.0), ("blk_b", 1, "INFO", NS, "OTHER", 0.0)])
        _, stats = self.build(ks=(2,), chunksize=1)
        self.assertEqual(stats["rows_out_of_time_order"], 1)

    def test_missing_file_explains_how_to_rebuild(self):
        missing = self.dir / "nope.csv"
        with self.assertRaises(FileNotFoundError) as ctx:
            require_cleaned_log(missing)
        message = str(ctx.exception)
        self.assertIn("scripts/download_dataset.py", message)
        self.assertIn("scripts/build_hdfs_features.py", message)
        self.assertEqual(message, MISSING_FILE_HELP.format(path=missing))
        with self.assertRaises(FileNotFoundError):
            build_prefix_features(missing, (3,), vocab_path=self.vocab)

    def test_bad_k_is_rejected(self):
        for ks in ((0,), (256,), ()):
            with self.assertRaises(ValueError):
                build_prefix_features(self.cleaned, ks, vocab_path=self.vocab)


class MetricsTest(unittest.TestCase):
    def test_lead_fraction(self):
        np.testing.assert_allclose(lead_fraction([20, 10, 5, 3, 0], 5), [0.75, 0.5, 0.0, 0.0, 0.0])

    def test_alert_metrics_on_a_known_confusion_matrix(self):
        alert = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
        truth = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
        lead = np.array([0.5, 0.0, 0.9, 0.3, 0, 0, 0, 0])
        m = alert_metrics(alert, truth, lead)
        self.assertEqual(m["confusion_matrix"], {"tp": 2, "fp": 1, "fn": 1, "tn": 4})
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["false_alarm_rate"], 1 / 5)
        # Lead counts only correct alerts; the false alarm's 0.9 is ignored.
        self.assertAlmostEqual(m["lead_time"]["mean_over_true_alerts"], 0.25)
        self.assertAlmostEqual(m["lead_time"]["share_of_true_alerts_before_block_end"], 0.5)
        self.assertAlmostEqual(m["lead_time"]["expected_lead_per_alert"], 0.5 / 3)

    def test_lead_seconds_cover_only_correct_alerts(self):
        alert = np.array([1, 1, 1, 0, 0], bool)
        truth = np.array([1, 1, 0, 1, 0], bool)
        lead = np.zeros(5)
        seconds = np.array([120.0, 30.0, 9999.0, 500.0, 0.0])  # the false alarm's and the miss's seconds are ignored
        m = alert_metrics(alert, truth, lead, seconds)
        self.assertAlmostEqual(m["lead_time"]["mean_seconds_over_true_alerts"], 75.0)
        self.assertAlmostEqual(m["lead_time"]["median_seconds_over_true_alerts"], 75.0)
        self.assertNotIn("mean_seconds_over_true_alerts", alert_metrics(alert, truth, lead)["lead_time"])  # only when given

    def test_lead_seconds_skip_missing_timestamps(self):
        m = alert_metrics(np.array([1, 1], bool), np.array([1, 1], bool), np.zeros(2), np.array([np.nan, 40.0]))
        self.assertEqual(m["lead_time"]["median_seconds_over_true_alerts"], 40.0)
        none = alert_metrics(np.array([1], bool), np.array([1], bool), np.zeros(1), np.array([np.nan]))
        self.assertIsNone(none["lead_time"]["mean_seconds_over_true_alerts"])

    def test_no_alerts_gives_zeros_not_errors(self):
        m = alert_metrics(np.zeros(4, bool), np.array([1, 0, 0, 0], bool), np.zeros(4))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(m["lead_time"]["expected_lead_per_alert"], 0.0)

    def test_split_is_stratified_disjoint_and_complete(self):
        labels = np.where(np.random.default_rng(0).random(2000) < 0.05, "Anomaly", "Normal")
        train, test = stratified_split(labels, 0.3, 42)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 2000)
        self.assertAlmostEqual((labels[train] == "Anomaly").mean(), (labels[test] == "Anomaly").mean(), delta=0.005)


class ThresholdGATest(unittest.TestCase):
    def test_ga_finds_a_threshold_close_to_the_best_possible(self):
        rng = np.random.default_rng(1)
        is_anomaly = rng.random(5000) < 0.05
        scores = np.where(is_anomaly, rng.normal(2.0, 0.7, 5000), rng.normal(0.0, 0.7, 5000))
        result = tune_threshold(scores, is_anomaly, population_size=16, num_generations=20)

        def f1(threshold):
            alert = scores >= threshold
            tp = (alert & is_anomaly).sum()
            return 2 * tp / (alert.sum() + is_anomaly.sum())

        best_possible = max(f1(t) for t in np.quantile(scores, np.linspace(0.5, 0.9995, 2000)))
        self.assertGreater(result.best_fitness, best_possible - 0.02)
        self.assertAlmostEqual(result.best_fitness, f1(result.threshold), places=9)
        self.assertEqual(len(result.best_fitness_per_generation), 20)
        self.assertTrue(np.all(np.diff(result.best_fitness_per_generation) >= -1e-12))  # elitism: never worse

    def test_same_seed_same_threshold_and_repeatable_in_process(self):
        rng = np.random.default_rng(2)
        is_anomaly = rng.random(1000) < 0.1
        scores = rng.normal(size=1000) + is_anomaly
        a = tune_threshold(scores, is_anomaly, population_size=8, num_generations=5, random_seed=3)
        b = tune_threshold(scores, is_anomaly, population_size=8, num_generations=5, random_seed=3)
        self.assertEqual(a.threshold, b.threshold)


class TrainEvaluateTest(unittest.TestCase):
    def make_frame(self, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        is_anomaly = rng.random(n) < 0.06
        total = np.where(is_anomaly, rng.integers(3, 8, n), rng.integers(15, 25, n))
        frame = pd.DataFrame({
            "block_id": [f"blk_{i}" for i in range(n)],
            "n_events": np.minimum(total, 5),
            "elapsed_seconds": np.where(is_anomaly, rng.integers(200, 900, n), rng.integers(0, 30, n)),
            "warn_count": np.where(is_anomaly, rng.integers(1, 4, n), 0),
            "error_count": 0,  # constant, must be dropped
            "count_OTHER": rng.integers(0, 3, n),
            TOTAL_EVENTS_COLUMN: total,
        })
        return frame, is_anomaly

    def test_train_and_evaluate_reports_held_out_metrics(self):
        frame, is_anomaly = self.make_frame()
        labels = np.where(is_anomaly, "Anomaly", "Normal")
        train_idx, test_idx = stratified_split(labels, 0.3, 42)
        features = ["n_events", "elapsed_seconds", "warn_count", "error_count", "count_OTHER"]
        result = train_and_evaluate_k(5, frame, is_anomaly, train_idx, test_idx, features,
                                      ga_kwargs={"population_size": 10, "num_generations": 8})
        self.assertEqual(result.dropped_constant_columns, ["error_count"])
        self.assertNotIn(TOTAL_EVENTS_COLUMN, result.model.feature_names_in_)  # the lead-time column never reaches fit
        self.assertEqual(list(result.model.feature_names_in_), result.feature_columns)
        test = result.test_metrics
        self.assertEqual(sum(test["confusion_matrix"].values()), len(test_idx))
        self.assertGreater(test["f1"], 0.5)  # the synthetic anomalies are easy
        self.assertEqual(result.test_blocks_shorter_than_k, int((frame[TOTAL_EVENTS_COLUMN].iloc[test_idx] <= 5).sum()))
        # The still-running view covers exactly the test blocks with more than k events.
        running = result.test_in_progress_metrics
        self.assertEqual(running["blocks"], len(test_idx) - result.test_blocks_shorter_than_k)
        self.assertEqual(sum(running["confusion_matrix"].values()), running["blocks"])
        self.assertEqual(running["anomalies"], int((is_anomaly[test_idx] & (frame[TOTAL_EVENTS_COLUMN].iloc[test_idx] > 5).to_numpy()).sum()))
        self.assertEqual(running["lead_time"]["share_of_true_alerts_before_block_end"], 1.0 if running["confusion_matrix"]["tp"] else 0.0)
        json.dumps(result.to_dict())  # serializable for the report

    def test_lead_seconds_reach_the_still_running_metrics(self):
        frame, is_anomaly = self.make_frame()
        # Alert at t=0 for everyone; anomalous blocks run on for 600 s, normal ones for 60 s.
        frame[PREFIX_END_COLUMN] = 1_000_000.0
        frame[BLOCK_END_COLUMN] = np.where(is_anomaly, 1_000_600.0, 1_000_060.0)
        labels = np.where(is_anomaly, "Anomaly", "Normal")
        train_idx, test_idx = stratified_split(labels, 0.3, 42)
        features = ["n_events", "elapsed_seconds", "warn_count", "error_count", "count_OTHER"]
        result = train_and_evaluate_k(5, frame, is_anomaly, train_idx, test_idx, features,
                                      ga_kwargs={"population_size": 10, "num_generations": 8})
        lead = result.test_in_progress_metrics["lead_time"]
        self.assertGreater(result.test_in_progress_metrics["confusion_matrix"]["tp"], 0)
        self.assertEqual(lead["median_seconds_over_true_alerts"], 600.0)  # only true alerts count
        self.assertEqual(lead["mean_seconds_over_true_alerts"], 600.0)
        for column in (PREFIX_END_COLUMN, BLOCK_END_COLUMN):
            self.assertNotIn(column, result.model.feature_names_in_)

    def test_trade_off_picks_from_the_pareto_front(self):
        def r(p, lead):
            return {"test": {"precision": p, "lead_time": {"mean_over_true_alerts": lead, "expected_lead_per_alert": p * lead}}}

        results = {3: r(0.4, 0.8), 5: r(0.6, 0.7), 10: r(0.5, 0.5), 20: r(0.9, 0.1)}
        out = trade_off(results)
        self.assertEqual(out["split"], "test")
        self.assertEqual(out["pareto_front"], [3, 5, 20])  # 10 is beaten by 5 on both
        self.assertEqual(out["best_k"], 5)  # 0.42 vs 0.32 and 0.09
        self.assertFalse(next(p for p in out["points"] if p["k"] == 10)["on_pareto_front"])

    def test_trade_off_can_rank_a_different_split(self):
        def r(test, running):
            entry = lambda p, lead: {"precision": p, "lead_time": {"mean_over_true_alerts": lead, "expected_lead_per_alert": p * lead}}
            return {"test": entry(*test), "test_in_progress": entry(*running)}

        results = {5: r((0.9, 0.05), (0.1, 0.8)), 20: r((0.8, 0.1), (0.7, 0.4))}
        self.assertEqual(trade_off(results, "test_in_progress")["best_k"], 20)
        self.assertEqual(trade_off(results, "test_in_progress")["pareto_front"], [5, 20])


class EarlyWarningApiTest(unittest.TestCase):
    def setUp(self):
        import jwt
        from flask import Flask

        import early_warning_api

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report_path = Path(self.tmp.name) / "report.json"
        report = {
            "results": {
                "3": {"k": 3, "test": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "false_alarm_rate": 0.01, "alerts": 10,
                                       "lead_time": {"mean_over_true_alerts": 0.8, "expected_lead_per_alert": 0.4}},
                      "ga_history": {"best_fitness_per_generation": [0.1, 0.2]}},
                "20": {"k": 20, "test": {"precision": 0.7, "recall": 0.5, "f1": 0.58, "false_alarm_rate": 0.005, "alerts": 8,
                                        "lead_time": {"mean_over_true_alerts": 0.1, "expected_lead_per_alert": 0.07}},
                       "test_in_progress": {"blocks": 40, "anomalies": 5, "precision": 0.6, "recall": 0.4, "f1": 0.48,
                                            "lead_time": {"mean_over_true_alerts": 0.3, "mean_seconds_over_true_alerts": 3600.0,
                                                          "median_seconds_over_true_alerts": 1800.0}},
                       "ga_history": {"best_fitness_per_generation": [0.3]}},
            },
            "trade_off": {"best_k": 3, "pareto_front": [3, 20]},
        }
        self.report_path.write_text(json.dumps(report))
        for patch in (mock.patch.object(early_warning_api, "REPORT_PATH", self.report_path),
                      mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"})):
            patch.start()
            self.addCleanup(patch.stop)

        app = Flask(__name__)
        app.register_blueprint(early_warning_api.early_warning_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "t", "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_requires_a_token(self):
        self.assertEqual(self.client.get("/api/early-warning/summary").status_code, 401)

    def test_summary_returns_a_table_sorted_by_k_without_ga_history(self):
        response = self.client.get("/api/early-warning/summary", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual([row["k"] for row in body["table"]], [3, 20])
        self.assertEqual(body["table"][0]["mean_lead_over_true_alerts"], 0.8)
        self.assertEqual(body["trade_off"]["best_k"], 3)
        running = body["table"][1]["in_progress"]
        self.assertEqual((running["median_lead_seconds"], running["mean_lead_seconds"]), (1800.0, 3600.0))
        self.assertEqual(running["mean_lead_over_true_alerts"], 0.3)  # the event-share lead stays
        self.assertIsNone(body["table"][0]["in_progress"])  # a report without the still-running view
        self.assertNotIn("ga_history", json.dumps(body))

    def test_404_with_a_hint_until_the_script_has_run(self):
        self.report_path.unlink()
        response = self.client.get("/api/early-warning/summary", headers=self.headers)
        self.assertEqual(response.status_code, 404)
        self.assertIn("build_early_warning.py", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
