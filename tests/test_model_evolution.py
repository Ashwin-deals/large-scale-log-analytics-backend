import json
import tempfile
import unittest
from pathlib import Path

from optimization.model_evolution import evaluate_and_promote, record_manual_promotion, record_rollback


class ModelEvolutionTest(unittest.TestCase):
    def test_promotes_when_candidate_f1_beats_current(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_metrics(paths["current_metrics"], f1=0.20)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])

            decision = self._evaluate_and_promote(paths)

            self.assertTrue(decision["promoted"])
            self.assertEqual(decision["new_version"], 2)
            self.assertTrue(Path(decision["promoted_model_path"]).exists())
            self.assertTrue(Path(decision["promoted_metrics_path"]).exists())

            with paths["current_version_path"].open(encoding="utf-8") as version_file:
                current_version = json.load(version_file)
            self.assertEqual(current_version["version"], 2)

            history = self._read_history(paths["version_history_path"])
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0]["promoted"])
            self.assertEqual(history[0]["new_version"], 2)

    def test_does_not_promote_when_candidate_f1_does_not_beat_current(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_metrics(paths["current_metrics"], f1=0.70)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])

            decision = self._evaluate_and_promote(paths)

            self.assertFalse(decision["promoted"])
            self.assertIsNone(decision["new_version"])
            self.assertIsNotNone(decision["archived_candidate_path"])
            self.assertTrue(Path(decision["archived_candidate_path"]).exists())
            self.assertFalse((paths["models_dir"] / "isolation_forest_v2.pkl").exists())

            history = self._read_history(paths["version_history_path"])
            self.assertEqual(len(history), 1)
            self.assertFalse(history[0]["promoted"])

    def test_version_history_is_append_only_across_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])

            self._write_metrics(paths["current_metrics"], f1=0.20)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._evaluate_and_promote(paths)

            self._write_metrics(paths["candidate_metrics"], f1=0.10)
            self._evaluate_and_promote(paths)

            history = self._read_history(paths["version_history_path"])
            self.assertEqual(len(history), 2)
            self.assertTrue(history[0]["promoted"])
            self.assertFalse(history[1]["promoted"])
            self.assertEqual(history[1]["current_version"], 2)

    def test_rollback_points_back_and_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])
            self._write_metrics(paths["current_metrics"], f1=0.20)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._evaluate_and_promote(paths)  # V2
            self._write_model(paths["models_dir"] / "isolation_forest_v1.pkl")
            self._write_metrics(paths["evaluation_dir"] / "isolation_forest_v1_metrics.json", f1=0.20)

            entry = record_rollback(1, "keep review numbers stable", **self._rollback_paths(paths))

            current = json.loads(paths["current_version_path"].read_text())
            self.assertEqual(current["version"], 1)
            self.assertEqual((entry["event"], entry["from_version"], entry["to_version"]), ("rollback", 2, 1))
            self.assertEqual(entry["to_metric_value"], 0.20)
            history = self._read_history(paths["version_history_path"])
            self.assertEqual(history[-1]["reason"], "keep review numbers stable")

    def test_backfilled_rollback_leaves_the_deployment_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_model(paths["models_dir"] / "isolation_forest_v1.pkl")
            paths["current_version_path"].write_text(json.dumps({"version": 1, "model_path": "x"}))
            entry = record_rollback(1, "backfill", timestamp="2026-09-09T04:40:48+00:00", update_current=False,
                                    from_version=2, **self._rollback_paths(paths))
            self.assertEqual((entry["from_version"], entry["to_version"]), (2, 1))
            self.assertEqual(json.loads(paths["current_version_path"].read_text())["model_path"], "x")
            self.assertEqual(self._read_history(paths["version_history_path"])[0]["timestamp"], "2026-09-09T04:40:48+00:00")

    def test_manual_promotion_puts_a_rolled_back_version_live_again(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])
            self._write_metrics(paths["current_metrics"], f1=0.20)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._evaluate_and_promote(paths)  # V2
            self._write_model(paths["models_dir"] / "isolation_forest_v1.pkl")
            self._write_metrics(paths["evaluation_dir"] / "isolation_forest_v1_metrics.json", f1=0.20)
            record_rollback(1, "unknown", **self._rollback_paths(paths))

            entry = record_manual_promotion(2, "wins held-out", **self._rollback_paths(paths))

            self.assertEqual(json.loads(paths["current_version_path"].read_text())["version"], 2)
            self.assertEqual((entry["event"], entry["from_version"], entry["to_version"]), ("manual_promotion", 1, 2))
            self.assertEqual(entry["to_metric_value"], 0.70)
            # The next retrain winner still gets a fresh number.
            paths["candidate_model"].write_bytes(b"newer")
            self._write_metrics(paths["candidate_metrics"], f1=0.90)
            self.assertEqual(self._evaluate_and_promote(paths)["new_version"], 3)

    def test_rollback_to_a_missing_version_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            with self.assertRaises(FileNotFoundError):
                record_rollback(7, "no such model", **self._rollback_paths(paths))

    def test_promotion_after_rollback_never_reuses_a_version_number(self):
        # V2 promoted, rolled back to V1, then a new candidate wins: it must
        # become V3, leaving the rolled-back V2 model and metrics untouched.
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_paths(directory)
            self._write_model(paths["current_model"])
            self._write_model(paths["candidate_model"])
            self._write_metrics(paths["current_metrics"], f1=0.20)
            self._write_metrics(paths["candidate_metrics"], f1=0.70)
            self._evaluate_and_promote(paths)
            v2_model = paths["models_dir"] / "isolation_forest_v2.pkl"
            v2_model.write_bytes(b"the-rolled-back-v2")
            self._write_model(paths["models_dir"] / "isolation_forest_v1.pkl")
            self._write_metrics(paths["evaluation_dir"] / "isolation_forest_v1_metrics.json", f1=0.20)
            record_rollback(1, "manual", **self._rollback_paths(paths))

            paths["candidate_model"].write_bytes(b"a-newer-candidate")
            self._write_metrics(paths["candidate_metrics"], f1=0.80)
            decision = self._evaluate_and_promote(paths)

            self.assertEqual(decision["new_version"], 3)
            self.assertEqual(v2_model.read_bytes(), b"the-rolled-back-v2")
            self.assertEqual(json.loads(paths["current_version_path"].read_text())["version"], 3)

    def _rollback_paths(self, paths: dict) -> dict:
        return {k: paths[k] for k in ("models_dir", "evaluation_dir", "current_version_path", "version_history_path")}

    def _evaluate_and_promote(self, paths: dict) -> dict:
        return evaluate_and_promote(
            current_model_path=paths["current_model"],
            candidate_model_path=paths["candidate_model"],
            current_metrics_path=paths["current_metrics"],
            candidate_metrics_path=paths["candidate_metrics"],
            metric="f1",
            models_dir=paths["models_dir"],
            evaluation_dir=paths["evaluation_dir"],
            archive_dir=paths["archive_dir"],
            current_version_path=paths["current_version_path"],
            version_history_path=paths["version_history_path"],
        )

    def _build_paths(self, directory: str) -> dict:
        root = Path(directory)
        return {
            "models_dir": root / "models",
            "evaluation_dir": root / "evaluation",
            "archive_dir": root / "models" / "archive",
            "current_version_path": root / "models" / "current_version.json",
            "version_history_path": root / "models" / "version_history.json",
            "current_model": root / "models" / "current.pkl",
            "candidate_model": root / "models" / "candidate.pkl",
            "current_metrics": root / "evaluation" / "current_metrics.json",
            "candidate_metrics": root / "evaluation" / "candidate_metrics.json",
        }

    def _write_metrics(self, path: Path, f1: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as metrics_file:
            json.dump({"accuracy": 0.9, "precision": 0.5, "recall": 0.5, "f1": f1}, metrics_file)

    def _write_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-model-bytes")

    def _read_history(self, path: Path) -> list:
        with path.open("r", encoding="utf-8") as history_file:
            return json.load(history_file)


if __name__ == "__main__":
    unittest.main()


class TimelineTest(unittest.TestCase):
    def test_manual_promotion_is_its_own_timeline_entry(self):
        import pipeline_api

        history = [
            {"timestamp": "t1", "event": "rollback", "metric": "f1", "promoted": False, "from_version": 3, "to_version": 2, "reason": "r1"},
            {"timestamp": "t2", "event": "manual_promotion", "metric": "f1", "promoted": False, "from_version": 2, "to_version": 3, "reason": "r2"},
        ]
        newest = pipeline_api._timeline_from_history(history)[0]
        self.assertEqual(newest["title"], "Re-promoted by hand: V2 -> V3")
        self.assertEqual((newest["tone"], newest["desc"]), ("primary", "r2"))


class HoldoutWarningTest(unittest.TestCase):
    def test_warns_when_held_out_numbers_predate_the_live_model(self):
        from unittest import mock

        import pipeline_api

        full = {"data": {"blocks": 575_061}, "deployed_version": 2}
        with mock.patch.object(pipeline_api, "resolve_current_deployment", lambda ctx=None: {"version": 3}):
            self.assertIn("V2 was live", pipeline_api._holdout_warnings(full, None)[0])
            self.assertEqual(pipeline_api._holdout_warnings({**full, "deployed_version": 3}, None), [])
