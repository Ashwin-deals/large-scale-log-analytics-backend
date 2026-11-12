"""Held-out comparison of every HDFS model version (V1, V2, V3, ...).

Why this exists: each version's published metrics come from scoring the very
blocks it was trained on, so they are in-sample. This refits each version's
exact hyperparameters and feature set (sklearn clone, random_state included)
on a stratified training split only, then scores the held-out split it never
saw. The deployed pickles are scored on the same held-out rows too, as a
reference: they were fitted on every block, so for them those rows are
in-sample and the gap shows how optimistic the published numbers are.

Repeated over several random splits (--repeat-seeds), because a single split
can't tell a real F1 difference between versions from split-to-split noise.
Full detail is kept for --seed; the repeats keep per-split F1 and mean/std.

Writes data/evaluation/holdout_comparison.json.

Usage:
    python3 scripts/evaluate_holdout.py                    # 30% held out, seed 42, + 4 repeat splits
    python3 scripts/evaluate_holdout.py --test-size 0.2 --seed 7 --repeat-seeds 1 2 3
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.predict import check_encoder_compatibility
from detection.train import load_labeled_features

MODELS_DIR = Path("data/models")
CURRENT_VERSION_PATH = MODELS_DIR / "current_version.json"
OUTPUT_PATH = Path("data/evaluation/holdout_comparison.json")
POSITIVE = "Anomaly"


def metrics(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    y = y_true == POSITIVE
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[False, True]).ravel()
    return {
        "accuracy": float(accuracy_score(y, flagged)),
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "f1": float(f1_score(y, flagged, zero_division=0)),
        "flagged": int(tp + fp),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "missed": int(fn),
    }


def flags(model, frame) -> np.ndarray:
    return model.predict(frame[list(model.feature_names_in_)]) == -1


def model_versions() -> list[int]:
    found = []
    for path in MODELS_DIR.glob("isolation_forest_v*.pkl"):
        suffix = path.stem.removeprefix("isolation_forest_v")
        if suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", type=int, nargs="*", default=[1, 7, 123, 2024],
                        help="extra split seeds for the stability check (default %(default)s)")
    args = parser.parse_args()

    merged = load_labeled_features().reset_index(drop=True)
    labels = merged["Label"].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(merged)), test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train, test = merged.iloc[train_idx], merged.iloc[test_idx]
    print(f"{len(merged):,} labeled blocks -> train {len(train):,} ({(train.Label == POSITIVE).sum():,} anomalies) "
          f"/ test {len(test):,} ({(test.Label == POSITIVE).sum():,} anomalies)")

    deployed = json.loads(CURRENT_VERSION_PATH.read_text())["version"] if CURRENT_VERSION_PATH.exists() else None
    results = {}
    for version in model_versions():
        template = joblib.load(MODELS_DIR / f"isolation_forest_v{version}.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # legacy models without a vocabulary stamp only warn
            check_encoder_compatibility(template)
        start = time.perf_counter()
        refit = clone(template).fit(train[list(template.feature_names_in_)])
        params = template.get_params()
        results[f"V{version}"] = {
            "deployed": version == deployed,
            "features": list(template.feature_names_in_),
            "hyperparameters": {k: params[k] for k in ("n_estimators", "max_samples", "max_features", "contamination", "bootstrap", "random_state")},
            "refit_train": metrics(train.Label.to_numpy(), flags(refit, train)),
            "refit_test": metrics(test.Label.to_numpy(), flags(refit, test)),
            "published_model_on_test_rows_in_sample": metrics(test.Label.to_numpy(), flags(template, test)),
            "fit_seconds": round(time.perf_counter() - start, 2),
        }
        r = results[f"V{version}"]
        print(f"V{version}{' (deployed)' if r['deployed'] else ''}: train F1 {r['refit_train']['f1']:.4f}  "
              f"test F1 {r['refit_test']['f1']:.4f}  (P {r['refit_test']['precision']:.4f}, R {r['refit_test']['recall']:.4f})  "
              f"| published model on the same rows {r['published_model_on_test_rows_in_sample']['f1']:.4f}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": (
            "Each version's hyperparameters and features are refitted (sklearn clone) on the stratified training "
            "split only and scored on the held-out split. 'published_model_on_test_rows_in_sample' scores the "
            "saved pickle, which was fitted on every block, on the same rows -- in-sample, for reference."
        ),
        "data": {
            "features_path": "data/features/features.csv",
            "blocks": int(len(merged)),
            "anomalies": int((labels == POSITIVE).sum()),
            "note": "Blocks in the local feature set. 575,061 means the full HDFS corpus; fewer means a sample.",
        },
        "split": {
            "test_size": args.test_size,
            "seed": args.seed,
            "train": {"blocks": int(len(train)), "anomalies": int((train.Label == POSITIVE).sum())},
            "test": {"blocks": int(len(test)), "anomalies": int((test.Label == POSITIVE).sum())},
        },
        "deployed_version": deployed,
        "versions": results,
    }
    seeds = [args.seed] + [s for s in args.repeat_seeds if s != args.seed]
    templates = {v: joblib.load(MODELS_DIR / f"isolation_forest_v{v}.pkl") for v in model_versions()}
    per_seed = {f"V{v}": {"train_f1": [], "test_f1": []} for v in templates}
    for seed in seeds:
        tr, te = train_test_split(np.arange(len(merged)), test_size=args.test_size, stratify=labels, random_state=seed)
        tr_frame, te_frame = merged.iloc[tr], merged.iloc[te]
        for v, template in templates.items():
            model = clone(template).fit(tr_frame[list(template.feature_names_in_)])
            per_seed[f"V{v}"]["train_f1"].append(metrics(tr_frame.Label.to_numpy(), flags(model, tr_frame))["f1"])
            per_seed[f"V{v}"]["test_f1"].append(metrics(te_frame.Label.to_numpy(), flags(model, te_frame))["f1"])
    stability = {"seeds": seeds, "versions": {}}
    for name, runs in per_seed.items():
        test_f1 = np.array(runs["test_f1"])
        stability["versions"][name] = {
            **runs,
            "test_f1_mean": float(test_f1.mean()),
            "test_f1_std": float(test_f1.std()),
        }
    best_per_seed = [max(per_seed, key=lambda n: per_seed[n]["test_f1"][i]) for i in range(len(seeds))]
    stability["wins_per_version"] = {name: best_per_seed.count(name) for name in per_seed}
    payload["stability_across_splits"] = stability
    print("\nHeld-out F1 across splits " + str(seeds))
    for name, row in stability["versions"].items():
        print(f"  {name}: mean {row['test_f1_mean']:.4f} +/- {row['test_f1_std']:.4f}  "
              f"best on {stability['wins_per_version'][name]} of {len(seeds)} splits")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
