"""End-to-end run for one non-HDFS source.

  ingest        parse every input line/record -> structured.csv; HDFSDataCleaner
                -> cleaned.csv; HDFSFeatureExtractor (the source's own vocab) ->
                features.csv. CloudTrail is streamed and cleaned/extracted in
                chunks cut at hour boundaries (its blocks are identity x hour,
                so no block straddles a chunk; checked, not assumed).
  labels        per sources.SourceConfig (ground truth / INJECTED / SYNTHETIC / WEAK rule)
  models        V1 = baseline Isolation Forest; GA candidate (optimization/
                ga_isolation_forest.run_ga, same 7 TRAINING_FEATURE_COLUMNS as
                HDFS); evaluate_and_promote() with this source's own
                current_version.json / version_history.json. Each run
                re-bootstraps the lineage from V1 so reruns are reproducible.
  smart storage the deployed model's hyperparameters refitted on a 50/50
                tune half, policy GA on tune, reported on the held-out half
  early warning prefix models per k, GA-tuned thresholds, held-out metrics
  report        data/evaluation/<source>_report.json

Split protocol: stratified 70/30 train/test for the models and early warning,
50/50 for smart storage. A source with protocol "in_sample" (OpenStack, 4
anomalies) fits and scores on the same blocks, and its report says so.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from detection.evaluate import evaluate
from detection.isolation_forest import TRAINING_FEATURE_COLUMNS, build_isolation_forest, save_model
from detection.predict import predict
from early_warning.model import train_and_evaluate_k, trade_off
from early_warning.prefix_features import TOTAL_EVENTS_COLUMN, build_prefix_features, feature_columns
from feature_engineering.block_metadata import build_block_metadata
from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import HDFSFeatureExtractor
from optimization.ga_isolation_forest import build_model, run_ga, stratified_subsample
from optimization.model_evolution import evaluate_and_promote
from parser.base_parser import COMMON_LOG_SCHEMA
from smart_storage.data import refit_on_rows
from smart_storage.optimizer import ScoringVariant, build_report
from smart_storage.optimizer import run_ga as run_storage_ga
from smart_storage.policy import BlockTable, Normalizer

from . import labels as label_rules
from .sources import SourceConfig, get_source

RANDOM_STATE = 42
TEST_SIZE = 0.3
STORAGE_TEST_SIZE = 0.5
EARLY_WARNING_KS = (3, 5, 10, 20)
FITNESS_SAMPLE_SIZE = 40_000
GA_POPULATION, GA_GENERATIONS, GA_PARENTS = 16, 12, 8  # the adaptation-cycle settings
STORAGE_GA = {"population_size": 60, "num_generations": 60, "num_parents_mating": 20}
CHUNK_ROWS = 200_000
POSITIVE = "Anomaly"


class Paths:
    def __init__(self, source: str):
        self.processed = Path("data/processed") / source
        self.features_dir = Path("data/features") / source
        self.models = Path("data/models") / source
        self.evaluation = Path("data/evaluation") / f"{source}_metrics"
        self.storage = Path("data/storage") / source
        self.report = Path("data/evaluation") / f"{source}_report.json"
        self.structured = self.processed / "structured.csv"
        self.cleaned = self.processed / "cleaned.csv"
        self.features = self.features_dir / "features.csv"
        self.labels = self.features_dir / "labels.csv"
        self.vocab = self.features_dir / "category_encoders.json"
        self.current_version = self.models / "current_version.json"
        self.version_history = self.models / "version_history.json"
        self.early_warning = self.models / "early_warning"

    def make(self):
        for directory in (self.processed, self.features_dir, self.models, self.evaluation, self.storage, self.early_warning):
            directory.mkdir(parents=True, exist_ok=True)


def _log(source, message):
    print(f"[{source}] {message}", flush=True)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def _iter_rows(config: SourceConfig, parser, files):
    for path in files:
        if hasattr(parser, "iter_rows"):
            yield from parser.iter_rows(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield parser.parse_line(line)


def _append_csv(frame: pd.DataFrame, path: Path, first: bool):
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def ingest(config: SourceConfig, paths: Paths) -> dict:
    files = config.inputs()
    if not files:
        raise FileNotFoundError(f"No input files for {config.name}. {config.prepare_hint}")
    for stale in (paths.cleaned, paths.features, paths.vocab):
        stale.unlink(missing_ok=True)  # fresh vocabulary per run: codes are this source's alone

    parser = config.parser_factory()
    stats = {"input_files": [str(f) for f in files], "lines_or_records": 0, "unparsed": 0, "no_block_id": 0}
    cleaner, extractor = HDFSDataCleaner(), HDFSFeatureExtractor()
    weak = label_rules.CloudTrailWeakLabeler() if config.name == "cloudtrail" else None
    tmp_cleaned, tmp_features = paths.processed / "_chunk_cleaned.csv", paths.processed / "_chunk_features.csv"

    buffer, first_chunk, cleaned_rows, last_hour, inversions, last_key = [], True, 0, None, 0, None

    def flush():
        nonlocal buffer, first_chunk, cleaned_rows
        if not buffer:
            return
        frame = pd.DataFrame(buffer, columns=COMMON_LOG_SCHEMA)
        _append_csv(frame, paths.structured, first_chunk)
        recognized = frame[frame["date"].notna() & frame["block_id"].notna()].reset_index(drop=True)
        if len(recognized):
            cleaned = cleaner.clean(recognized, tmp_cleaned)
            extractor.extract(cleaned, tmp_features, vocab_path=paths.vocab)
            features = pd.read_csv(tmp_features)
            _append_csv(cleaned, paths.cleaned, first_chunk)
            _append_csv(features, paths.features, first_chunk)
            cleaned_rows += len(cleaned)
            if weak is not None:
                weak.update(cleaned)
            first_chunk = False
        buffer = []

    for row in _iter_rows(config, parser, files):
        stats["lines_or_records"] += 1
        if row.get("date") is None:
            stats["unparsed"] += 1
        elif row.get("block_id") is None:
            stats["no_block_id"] += 1
        buffer.append(row)
        if config.chunked_by_time and row.get("date") is not None:
            key = (row["date"], row["time"])
            inversions += last_key is not None and key < last_key
            last_key = key
            hour = (row["date"], row["time"][:2])
            if len(buffer) >= CHUNK_ROWS and hour != last_hour and last_hour is not None:
                buffer.pop()
                flush()
                buffer.append(row)
            last_hour = hour
    if not config.chunked_by_time:
        # Small sources: one pass, exactly the batch HDFS path.
        frame = pd.DataFrame(buffer, columns=COMMON_LOG_SCHEMA)
        frame.to_csv(paths.structured, index=False)
        recognized = frame[frame["date"].notna() & frame["block_id"].notna()].reset_index(drop=True)
        cleaned = cleaner.clean(recognized, paths.cleaned)
        extractor.extract(cleaned, paths.features, vocab_path=paths.vocab)
        cleaned_rows = len(cleaned)
    else:
        flush()
        stats["timestamp_inversions"] = int(inversions)
    for tmp in (tmp_cleaned, tmp_features):
        tmp.unlink(missing_ok=True)

    features = pd.read_csv(paths.features)
    duplicated = int(features["block_id"].duplicated().sum())
    if duplicated:
        raise RuntimeError(f"{duplicated} blocks were split across ingest chunks; the input is not in time order")
    stats.update({
        "recognized": stats["lines_or_records"] - stats["unparsed"] - stats["no_block_id"],
        "events_after_cleaning": int(cleaned_rows),
        "dropped_by_cleaner": int(stats["lines_or_records"] - stats["unparsed"] - stats["no_block_id"] - cleaned_rows),
        "blocks": int(len(features)),
    })
    return {"stats": stats, "weak_labeler": weak}


def build_labels(config: SourceConfig, paths: Paths, block_ids: pd.Series, weak) -> tuple[pd.Series, dict]:
    info = {}
    if config.name == "openstack":
        abnormal_run = label_rules.instances_in("data/raw/openstack/openstack_abnormal.log")
        labels = pd.Series(np.where(block_ids.isin(abnormal_run), POSITIVE, "Normal"), index=block_ids.values)
        info["rule"] = "Anomaly = instance session found in openstack_abnormal.log; Normal = found in openstack_normal1/2.log"
        info["sessions_in_abnormal_run"] = len(abnormal_run)
        faulty = label_rules.openstack_anomalous_instances("data/raw/openstack/anomaly_labels.txt")
        info["ground_truth_faulty_instances"] = sorted(faulty)
    elif config.name == "cloudtrail":
        table = weak.labels()
        table.to_csv(paths.features_dir / "weak_labels.csv", index=False)
        labels = table.set_index("block_id")["label"].reindex(block_ids.values)
        info["rule"] = label_rules.WEAK_RULE_DESCRIPTION
        info["blocks_by_rule"] = {
            "r1_auth_error_burst": int(table["rule_r1_auth_error_burst"].sum()),
            "r2_new_ip_sensitive_action": int(table["rule_r2_new_ip_sensitive_action"].sum()),
            "both": int((table["rule_r1_auth_error_burst"] & table["rule_r2_new_ip_sensitive_action"]).sum()),
        }
    else:
        label_file = {"apache": "data/raw/apache/injected/Apache_INJECTED_labels.csv",
                      "django": "data/raw/django/django_SYNTHETIC_labels.csv"}[config.name]
        labels = label_rules.labels_from_csv(label_file).reindex(block_ids.values)
        info["label_file"] = label_file
    missing = int(labels.isna().sum())
    info["blocks_without_a_label_treated_normal"] = missing
    labels = labels.fillna("Normal")
    pd.DataFrame({"block_id": block_ids.values, "Label": labels.values}).to_csv(paths.labels, index=False)
    return labels, info


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def split(labels: np.ndarray, protocol: str, test_size: float):
    positions = np.arange(len(labels))
    if protocol == "in_sample":
        return positions, positions
    train, test = train_test_split(positions, test_size=test_size, stratify=labels, random_state=RANDOM_STATE)
    return np.sort(train), np.sort(test)


def _caught(metrics: dict) -> str:
    matrix = metrics["confusion_matrix"]["matrix"]
    return f"caught {matrix[1][1]} of {matrix[1][0] + matrix[1][1]}"


def _evaluate(model, frame, columns, paths, name, vocab):
    predictions = predict(model, frame, output_path=paths.evaluation / f"{name}_predictions.csv",
                          feature_columns=columns, vocab_path=vocab)
    metrics = evaluate(predictions, metrics_path=paths.evaluation / f"{name}_metrics.json")
    metrics["caught"] = _caught(metrics)
    (paths.evaluation / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_models(config: SourceConfig, paths: Paths, merged: pd.DataFrame, train_idx, test_idx) -> dict:
    train, test = merged.iloc[train_idx], merged.iloc[test_idx]
    for stale in (paths.current_version, paths.version_history):
        stale.unlink(missing_ok=True)  # fresh lineage per run, see module docstring
    for old in paths.models.glob("isolation_forest_v*.pkl"):
        old.unlink()

    v1 = build_isolation_forest()
    v1.fit(train[TRAINING_FEATURE_COLUMNS])
    v1_path = paths.models / "isolation_forest_v1.pkl"
    save_model(v1, v1_path, vocab_path=paths.vocab)
    v1_metrics = _evaluate(v1, test, TRAINING_FEATURE_COLUMNS, paths, "isolation_forest_v1", paths.vocab)
    v1_metrics_path = paths.evaluation / "isolation_forest_v1_metrics.json"
    paths.current_version.write_text(json.dumps({
        "version": 1, "model_path": str(v1_path), "metrics_path": str(v1_metrics_path),
        "metric": "f1", "metric_value": v1_metrics["f1"], "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": f"Baseline Isolation Forest for {config.name}, bootstrapped by multi_source/pipeline.py.",
    }, indent=2), encoding="utf-8")

    fitness = stratified_subsample(train.reset_index(drop=True), min(FITNESS_SAMPLE_SIZE, len(train)), RANDOM_STATE)
    _, ga = run_ga(X=fitness[TRAINING_FEATURE_COLUMNS], y=fitness["Label"], population_size=GA_POPULATION,
                   num_generations=GA_GENERATIONS, num_parents_mating=GA_PARENTS)
    best = ga.best_chromosome
    candidate = build_model(best)
    candidate.fit(train[best.selected_features])
    candidate_path = paths.models / "isolation_forest_candidate.pkl"
    save_model(candidate, candidate_path, vocab_path=paths.vocab)
    candidate_metrics = _evaluate(candidate, test, best.selected_features, paths, "isolation_forest_candidate", paths.vocab)

    decision = evaluate_and_promote(
        current_model_path=v1_path, candidate_model_path=candidate_path,
        current_metrics_path=v1_metrics_path, candidate_metrics_path=paths.evaluation / "isolation_forest_candidate_metrics.json",
        metric="f1", models_dir=paths.models, evaluation_dir=paths.evaluation, archive_dir=paths.models / "archive",
        current_version_path=paths.current_version, version_history_path=paths.version_history,
    )
    deployed = json.loads(paths.current_version.read_text(encoding="utf-8"))
    return {
        "v1_baseline": v1_metrics,
        "ga_candidate": {**candidate_metrics, "chromosome": best.as_readable_dict(), "ga_best_fitness": ga.best_fitness,
                         "ga_generations": ga.generations_completed, "ga_seconds": ga.wall_clock_seconds},
        "promotion": decision,
        "deployed": {"version": deployed["version"], "model_path": deployed["model_path"], "f1": deployed["metric_value"]},
    }


# ---------------------------------------------------------------------------
# smart storage + early warning
# ---------------------------------------------------------------------------

def _scored_frame(model, merged, paths, name):
    predictions = predict(model, merged, output_path=paths.storage / f"_{name}_scores.csv",
                          feature_columns=list(model.feature_names_in_), vocab_path=paths.vocab)
    return pd.DataFrame({
        "block_id": merged["block_id"].to_numpy(),
        "anomaly_score": predictions["anomaly_score"].to_numpy(),
        "predicted_label": predictions["predicted_label"].to_numpy(),
        "true_label": merged["Label"].to_numpy(),
        "event_frequency": merged["event_frequency"].to_numpy(),
        "block_size": merged["block_size"].to_numpy(),
        "is_error": merged["is_error"].to_numpy(),
    })


def feature_separability(merged: pd.DataFrame) -> dict:
    """ROC AUC of each single feature against the label (max of AUC and 1-AUC,
    so 0.5 = no signal, 1.0 = one feature separates the classes perfectly).
    A near-1.0 feature with no plausible link to faults points at a label leak."""
    from sklearn.metrics import roc_auc_score

    y = (merged["Label"] == POSITIVE).to_numpy()
    out = {}
    for column in TRAINING_FEATURE_COLUMNS:
        values = merged[column].to_numpy(dtype=float)
        if np.unique(values).size < 2:
            out[column] = 0.5
            continue
        auc = roc_auc_score(y, values)
        out[column] = round(float(max(auc, 1 - auc)), 4)
    return out


def ground_truth_check(merged: pd.DataFrame, model_path: str, paths: Paths) -> dict:
    """OpenStack only: of the 4 instances anomaly_labels.txt says were faulty,
    how many does the deployed model flag, scoring every block."""
    faulty = label_rules.openstack_anomalous_instances("data/raw/openstack/anomaly_labels.txt")
    model = joblib.load(model_path)
    predictions = predict(model, merged, output_path=paths.evaluation / "_ground_truth_scores.csv",
                          feature_columns=list(model.feature_names_in_), vocab_path=paths.vocab)
    (paths.evaluation / "_ground_truth_scores.csv").unlink(missing_ok=True)
    flagged = set(predictions.loc[predictions["predicted_label"] == POSITIVE, "block_id"])
    present = faulty & set(merged["block_id"])
    return {
        "faulty_instances_present": len(present),
        "caught": f"caught {len(present & flagged)} of {len(present)}",
        "flagged_overall": len(flagged),
        "note": "In-sample over every block (the 4 are too few to hold out). Diagnostic only, not the training target.",
    }


def run_smart_storage(config: SourceConfig, paths: Paths, merged: pd.DataFrame, deployed_model_path: str) -> dict:
    tune_idx, holdout_idx = split(merged["Label"].to_numpy(), config.protocol, STORAGE_TEST_SIZE)
    template = joblib.load(deployed_model_path)
    refit = refit_on_rows(template, merged, tune_idx)
    model_path = paths.storage / "isolation_forest_tune_half.pkl"
    save_model(refit, model_path, vocab_path=paths.vocab)
    frame = _scored_frame(refit, merged, paths, "tune_half")
    tune_frame, holdout_frame = frame.iloc[tune_idx], frame.iloc[holdout_idx]
    normalizer = Normalizer.fit(tune_frame["anomaly_score"], tune_frame["event_frequency"], tune_frame["block_size"])
    tune_table, holdout_table = BlockTable.from_frame(tune_frame, normalizer), BlockTable.from_frame(holdout_frame, normalizer)
    _, ga = run_storage_ga(tune_table, verbose=False, random_seed=RANDOM_STATE, **STORAGE_GA)
    variant = ScoringVariant({"kind": "tune_half_refit", "model_path": str(model_path), "hyperparameters_from": deployed_model_path,
                              "fitted_on": "all blocks (in-sample)" if config.protocol == "in_sample" else "tune half only"},
                             normalizer, tune_table, holdout_table, ga)
    policy_payload, report = build_report(variant, None, 0.95, 1000.0, STORAGE_TEST_SIZE, RANDOM_STATE, n_random_trials=10)
    # The API rebuilds the same tune/holdout split from this; in-sample sources use every block for both.
    policy_payload["split"]["protocol"] = config.protocol
    (paths.storage / "policy_best.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
    (paths.storage / "storage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for scratch in paths.storage.glob("_*_scores.csv"):
        scratch.unlink()

    keep = ("storage_saved_pct", "weighted_retention", "full_retention", "coverage", "fitness", "meets_retention_target")
    baselines = {name: {k: r[k] for k in keep} for name, r in report["baselines"].items() if name != "top_k_percent_sweep"}
    return {
        "protocol": "in-sample (tune = holdout = all blocks)" if config.protocol == "in_sample" else "50/50 stratified tune/holdout",
        "policy": report["policy"],
        "holdout": {k: report["holdout"][k] for k in keep + ("tier_counts", "tier_anomalies", "n_blocks", "n_anomalies")},
        "baselines_on_holdout": baselines,
        "ga_beats_baselines": report["ga_beats_baselines"],
        "policy_path": str(paths.storage / "policy_best.json"),
    }


def run_early_warning(config: SourceConfig, paths: Paths, label_of: pd.Series) -> dict:
    frames, scan = build_prefix_features(paths.cleaned, EARLY_WARNING_KS, vocab_path=paths.vocab)
    first = frames[EARLY_WARNING_KS[0]]
    labels = first["block_id"].map(label_of).fillna("Normal").to_numpy()
    is_anomaly = labels == POSITIVE
    train_idx, test_idx = split(labels, config.protocol, TEST_SIZE)
    count_columns = [c for c in first.columns if c.startswith("count_")]
    columns = feature_columns([c[len("count_"):] for c in count_columns])
    per_k = {}
    for k in EARLY_WARNING_KS:
        result = train_and_evaluate_k(k, frames[k], is_anomaly, train_idx, test_idx, columns,
                                      ga_kwargs={"population_size": 20, "num_generations": 30, "random_seed": RANDOM_STATE})
        result.model.prefix_k_, result.model.alert_threshold_ = k, result.ga.threshold
        save_model(result.model, paths.early_warning / f"isolation_forest_k{k}.pkl", vocab_path=paths.vocab)
        entry = result.to_dict()
        entry.pop("ga_history")
        entry["test"]["caught"] = f"caught {entry['test']['confusion_matrix']['tp']} of {int(is_anomaly[test_idx].sum())}"
        per_k[str(k)] = entry
    numeric = {int(k): v for k, v in per_k.items()}
    return {
        "ks": list(EARLY_WARNING_KS),
        "blocks": int(len(first)),
        "median_events_per_block": float(np.median(first[TOTAL_EVENTS_COLUMN])),
        "results": per_k,
        "trade_off": trade_off(numeric),
        "trade_off_in_progress": trade_off(numeric, split="test_in_progress"),
        "scan": scan,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _ingest_warnings(config: SourceConfig, stats: dict) -> list[str]:
    dropped, recognized = stats["dropped_by_cleaner"], max(stats["recognized"], 1)
    if not dropped:
        return []
    share = f"{dropped:,} exact-duplicate rows ({dropped / recognized:.1%} of recognized {'records' if config.name == 'cloudtrail' else 'lines'})"
    if config.name == "cloudtrail":
        # raw_message carries the eventID, so a dropped row repeats a whole record.
        return [f"The shared HDFSDataCleaner removed {share}: records repeating the same eventID, time, caller, action, "
                "source IP and errorCode, i.e. duplicate deliveries of one event, so removing them is correct de-duplication."]
    return [f"The shared HDFSDataCleaner removed {share}. At 1-second resolution, identical lines repeated within the same "
            "second count as one event, so per-block event counts here understate bursts of repeated lines."]


def run(source: str) -> dict:
    config = get_source(source)
    if config.name == "cloudtrail":
        config.label_description = label_rules.WEAK_RULE_DESCRIPTION
    paths = Paths(source)
    paths.make()
    started = time.perf_counter()

    _log(source, "ingesting (parse -> clean -> extract) ...")
    ingested = ingest(config, paths)
    stats = ingested["stats"]
    _log(source, f"{stats['lines_or_records']:,} lines/records -> {stats['events_after_cleaning']:,} events, {stats['blocks']:,} blocks")
    # Per-block metadata and dataset stats for the dashboard's ?source= views.
    build_block_metadata(paths.cleaned, paths.processed / "block_metadata.csv", paths.processed / "dataset_stats.json")

    features = pd.read_csv(paths.features)
    label_of, label_info = build_labels(config, paths, features["block_id"], ingested["weak_labeler"])
    merged = features.assign(Label=features["block_id"].map(label_of).to_numpy())
    n_anomalies = int((merged["Label"] == POSITIVE).sum())
    if n_anomalies < 2:
        raise RuntimeError(f"{source}: only {n_anomalies} anomalous block(s) found; nothing to evaluate")
    _log(source, f"labels: {n_anomalies:,} Anomaly of {len(merged):,} ({config.label_kind})")

    train_idx, test_idx = split(merged["Label"].to_numpy(), config.protocol, TEST_SIZE)
    _log(source, "training V1 baseline + GA candidate ...")
    models = train_models(config, paths, merged, train_idx, test_idx)
    _log(source, f"V1 F1 {models['v1_baseline']['f1']:.4f} ({models['v1_baseline']['caught']}), "
                 f"GA F1 {models['ga_candidate']['f1']:.4f} ({models['ga_candidate']['caught']}), deployed v{models['deployed']['version']}")

    _log(source, "smart storage ...")
    storage = run_smart_storage(config, paths, merged, models["deployed"]["model_path"])
    _log(source, f"storage: saved {storage['holdout']['storage_saved_pct']:.1f}%, weighted retention {storage['holdout']['weighted_retention']:.3f}")

    _log(source, "early warning ...")
    early = run_early_warning(config, paths, label_of)
    _log(source, f"early warning best k = {early['trade_off']['best_k']}")

    report = {
        "source": source,
        "title": config.title,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_kind": config.data_kind,
        "label_kind": config.label_kind,
        "label_description": config.label_description,
        "synthetic": config.data_kind == "SYNTHETIC",
        "injected": "INJECTED" in config.data_kind,
        "weak_labels": config.label_kind.startswith("WEAK"),
        "warnings": config.warnings + _ingest_warnings(config, stats),
        "block_definition": config.block_description,
        "evaluation_protocol": ("in-sample: fit and scored on the same blocks (too few anomalies to hold any out)"
                                if config.protocol == "in_sample" else "stratified 70/30 train/test; all metrics on test"),
        "ingest": stats,
        "labels": {"blocks": int(len(merged)), "anomalies": n_anomalies, "anomaly_rate": n_anomalies / len(merged), **label_info},
        "split": {"train_blocks": int(len(train_idx)), "test_blocks": int(len(test_idx)),
                  "test_anomalies": int((merged["Label"].iloc[test_idx] == POSITIVE).sum())},
        "features": TRAINING_FEATURE_COLUMNS,
        "feature_separability": feature_separability(merged),
        "models": models,
        "smart_storage": storage,
        "early_warning": early,
        "paths": {"models": str(paths.models), "metrics": str(paths.evaluation), "storage": str(paths.storage),
                  "vocab": str(paths.vocab), "cleaned": str(paths.cleaned)},
        "wall_clock_seconds": round(time.perf_counter() - started, 1),
    }
    if config.name == "openstack":
        report["ground_truth_check"] = ground_truth_check(merged, models["deployed"]["model_path"], paths)
    paths.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(source, f"wrote {paths.report} in {report['wall_clock_seconds']:.0f}s")
    return report
