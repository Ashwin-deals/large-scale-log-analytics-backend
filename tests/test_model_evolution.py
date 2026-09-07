import json
import tempfile
import unittest
from pathlib import Path

from optimization.model_evolution import evaluate_and_promote


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
