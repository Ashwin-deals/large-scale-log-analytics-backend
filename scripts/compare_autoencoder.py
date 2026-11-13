"""Autoencoder vs Isolation Forest on the HDFS held-out split (status doc D1).

Answers "why Isolation Forest?" with numbers instead of an argument. Both
detectors are unsupervised and see the same blocks:

  Isolation Forest   the deployed V2 settings (features, trees, contamination),
                     refitted on the training split
  Autoencoder        a dense autoencoder (sklearn MLPRegressor trained to
                     reproduce its input, bottleneck layer) on standardized
                     features (StandardScaler fitted on the training split);
                     anomaly score = reconstruction error

Two feature sets for the autoencoder: V2's 3 features, and all 7. Each
detector is scored at the same alert rate as V2 (its contamination: the top
2.35% of training scores) so the comparison is like for like, and also at the
threshold that maximizes F1 on the training split (labels used for the
threshold only, never for fitting). Repeated over several random splits.

Writes data/evaluation/autoencoder_comparison.json.

Usage:
    python3 scripts/compare_autoencoder.py
    python3 scripts/compare_autoencoder.py --seeds 42 1 7 123 2024
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS
from detection.train import load_labeled_features

DEPLOYED_PATH = Path("data/models/current_version.json")
OUTPUT_PATH = Path("data/evaluation/autoencoder_comparison.json")
POSITIVE = "Anomaly"


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    # Strictly above, as IsolationForest.predict does (score_samples < offset_):
    # HDFS features are discrete, so many blocks tie and ">=" would flag every
    # tie at the cutoff, giving the forest a different alert rate than its own
    # predict() -- and than scripts/evaluate_holdout.py.
    flagged = scores > threshold
    return {
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "flagged": int(flagged.sum()),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing F1 on these (training) scores, over a quantile grid."""
    grid = np.unique(np.quantile(scores, np.linspace(0.90, 0.999, 200)))
    return float(max(grid, key=lambda t: f1_score(y_true, scores > t, zero_division=0)))


class Autoencoder:
    """Standardize -> MLP trained to reproduce its input -> per-row squared error."""

    def __init__(self, n_features: int, seed: int):
        bottleneck = max(1, n_features // 2)
        self.scaler = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=(max(8, 2 * n_features), bottleneck, max(8, 2 * n_features)),
            activation="relu", alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
            max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=seed,
        )

    def fit(self, x: np.ndarray) -> "Autoencoder":
        z = self.scaler.fit_transform(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net.fit(z, z)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        return ((self.net.predict(z) - z) ** 2).mean(axis=1)


def evaluate(name, train_scores, test_scores, y_train, y_test, alert_rate, fit_seconds) -> dict:
    same_rate = float(np.quantile(train_scores, 1 - alert_rate))
    tuned = best_f1_threshold(y_train, train_scores)
    return {
        "detector": name,
        "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
        "at_deployed_alert_rate": metrics_at(y_test, test_scores, same_rate),
        "at_train_best_f1_threshold": metrics_at(y_test, test_scores, tuned),
        "fit_seconds": round(fit_seconds, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    deployed = json.loads(DEPLOYED_PATH.read_text())
    template = joblib.load(deployed["model_path"])
    v2_features = list(template.feature_names_in_)
    alert_rate = float(template.contamination)

    merged = load_labeled_features().reset_index(drop=True)
    labels = merged["Label"].to_numpy()
    y = labels == POSITIVE
    print(f"{len(merged):,} blocks ({y.sum():,} anomalies); deployed V{deployed['version']} features {v2_features}, "
          f"alert rate {alert_rate:.4f}; seeds {args.seeds}")

    runs = []
    for seed in args.seeds:
        tr, te = train_test_split(np.arange(len(merged)), test_size=args.test_size, stratify=labels, random_state=seed)
        train, test = merged.iloc[tr], merged.iloc[te]
        results = []

        start = time.perf_counter()
        forest = clone(template).fit(train[v2_features])
        results.append(evaluate("Isolation Forest (V2 settings)", -forest.score_samples(train[v2_features]),
                                -forest.score_samples(test[v2_features]), y[tr], y[te], alert_rate,
                                time.perf_counter() - start))

        for label, columns in (("V2's 3 features", v2_features), ("all 7 features", list(TRAINING_FEATURE_COLUMNS))):
            start = time.perf_counter()
            auto = Autoencoder(len(columns), seed).fit(train[columns].to_numpy(dtype=float))
            results.append(evaluate(f"Autoencoder ({label})", auto.score(train[columns].to_numpy(dtype=float)),
                                    auto.score(test[columns].to_numpy(dtype=float)), y[tr], y[te], alert_rate,
                                    time.perf_counter() - start))
        for r in results:
            same, tuned = r["at_deployed_alert_rate"], r["at_train_best_f1_threshold"]
            print(f"  seed {seed:>4} {r['detector']:<32} AUC {r['roc_auc_test']:.3f}  "
                  f"F1 @same rate {same['f1']:.3f} (P {same['precision']:.3f} R {same['recall']:.3f})  "
                  f"F1 @tuned {tuned['f1']:.3f}  fit {r['fit_seconds']:.1f}s")
        runs.append({"seed": seed, "train_blocks": int(len(tr)), "test_blocks": int(len(te)),
                     "test_anomalies": int(y[te].sum()), "results": results})

    summary = {}
    for name in [r["detector"] for r in runs[0]["results"]]:
        rows = [next(r for r in run["results"] if r["detector"] == name) for run in runs]
        summary[name] = {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for key, values in {
                "roc_auc_test": [r["roc_auc_test"] for r in rows],
                "f1_at_deployed_alert_rate": [r["at_deployed_alert_rate"]["f1"] for r in rows],
                "f1_at_train_best_f1_threshold": [r["at_train_best_f1_threshold"]["f1"] for r in rows],
                "fit_seconds": [r["fit_seconds"] for r in rows],
            }.items()
        }
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": ("Unsupervised fit on a stratified 70% split, scored on the unseen 30%. Thresholds: the deployed "
                     "model's alert rate (top contamination share of training scores), and the F1-best threshold "
                     "on training scores (labels used for the threshold only)."),
        "data": {"blocks": int(len(merged)), "anomalies": int(y.sum()),
                 "note": "575,061 blocks means the full HDFS corpus; fewer means the local sample."},
        "deployed_version": deployed["version"],
        "alert_rate": alert_rate,
        "summary": summary,
        "runs": runs,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nMean over seeds:")
    for name, row in summary.items():
        print(f"  {name:<32} AUC {row['roc_auc_test']['mean']:.3f}  F1 @same rate {row['f1_at_deployed_alert_rate']['mean']:.3f} "
              f"± {row['f1_at_deployed_alert_rate']['std']:.3f}  F1 @tuned {row['f1_at_train_best_f1_threshold']['mean']:.3f}  "
              f"fit {row['fit_seconds']['mean']:.1f}s")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
