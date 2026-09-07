import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from detection.train import load_labeled_features
from optimization.ga_isolation_forest import TRAINING_FEATURE_COLUMNS, build_model, run_ga, stratified_subsample
from optimization.model_evolution import DEFAULT_CURRENT_VERSION_PATH, DEFAULT_VERSION_HISTORY_PATH, evaluate_and_promote

pipeline_bp = Blueprint("pipeline", __name__)

MODELS_DIR = Path("data/models")
EVALUATION_DIR = Path("data/evaluation")
BLOCK_METADATA_PATH = Path("data/processed/block_metadata.csv")
DATASET_STATS_PATH = Path("data/processed/dataset_stats.json")
GA_BEST_CONFIG_PATH = Path("data/optimization/ga_best_config.json")

CANDIDATE_MODEL_PATH = Path("data/models/isolation_forest_candidate.pkl")
CANDIDATE_METRICS_PATH = Path("data/evaluation/isolation_forest_candidate_metrics.json")
CANDIDATE_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_candidate_predictions.csv")

DEFAULT_CURRENT_MODEL_PATH = Path("data/models/isolation_forest_v1.pkl")
DEFAULT_CURRENT_METRICS_PATH = Path("data/evaluation/baseline_isolation_forest_metrics.json")

FITNESS_SAMPLE_SIZE = 40_000
POPULATION_SIZE = 16
NUM_GENERATIONS = 12
NUM_PARENTS_MATING = 8

TOP_N_COMPONENTS = 6


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_float(value):
    if value is None:
        return None
    value = float(value)
    return None if (math.isnan(value) or math.isinf(value)) else value


# ---------------------------------------------------------------------------
# Model / version helpers
# ---------------------------------------------------------------------------

def resolve_current_deployment():
    if not DEFAULT_CURRENT_VERSION_PATH.exists():
        return {
            "version": 1,
            "model_path": str(DEFAULT_CURRENT_MODEL_PATH),
            "metrics_path": str(DEFAULT_CURRENT_METRICS_PATH),
        }
    current = _read_json(DEFAULT_CURRENT_VERSION_PATH)
    return current


def _metrics_for(metrics_path: str) -> dict | None:
    return _read_json(Path(metrics_path))


# ---------------------------------------------------------------------------
# In-memory prediction cache (recomputed only when the deployed version changes)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache = {
    "version": None,
    "predictions": None,  # DataFrame: block_id, true_label, predicted_label, anomaly_score, severity
    "block_metadata": None,  # DataFrame
    "dataset_stats": None,
}


def _load_block_metadata() -> pd.DataFrame:
    if _cache["block_metadata"] is None:
        frame = pd.read_csv(BLOCK_METADATA_PATH)
        frame["first_seen"] = pd.to_datetime(frame["first_seen"])
        _cache["block_metadata"] = frame
    return _cache["block_metadata"]


def _load_dataset_stats() -> dict:
    if _cache["dataset_stats"] is None:
        _cache["dataset_stats"] = _read_json(DATASET_STATS_PATH) or {}
    return _cache["dataset_stats"]


def _compute_severity(scores: pd.Series, is_anomaly: pd.Series) -> pd.Series:
    """
    Buckets anomaly_score into critical/high/medium by tertile, computed
    over currently-predicted anomalies only (so buckets stay meaningful
    regardless of the deployed model's score range). Normal rows get
    "normal". Higher anomaly_score = more anomalous, so "critical" is the
    top tertile.
    """
    anomaly_scores = scores[is_anomaly]
    severity = pd.Series("normal", index=scores.index)
    if len(anomaly_scores) == 0:
        return severity

    high_cut = anomaly_scores.quantile(2 / 3)
    critical_cut = anomaly_scores.quantile(1 / 3)

    severity = severity.mask(is_anomaly, "medium")
    severity = severity.mask(is_anomaly & (scores >= critical_cut), "high")
    severity = severity.mask(is_anomaly & (scores >= high_cut), "critical")
    return severity


def get_current_predictions() -> pd.DataFrame:
    """
    Returns block-level predictions (block_id, true_label, predicted_label,
    anomaly_score, severity, first_seen, component, event_type, source_ip,
    destination_ip, block_size) for whichever model is currently deployed.
    Cached in memory; recomputed only when current_version.json's version
    changes (i.e. after a promotion).
    """
    current = resolve_current_deployment()
    version = current.get("version")

    with _cache_lock:
        if _cache["version"] == version and _cache["predictions"] is not None:
            return _cache["predictions"]

        model = joblib.load(current["model_path"])
        merged = load_labeled_features()
        feature_columns = list(model.feature_names_in_)
        predictions = predict(model, merged, output_path=CANDIDATE_PREDICTIONS_PATH.parent / "_live_predictions.csv", feature_columns=feature_columns)

        is_anomaly = predictions["predicted_label"] == "Anomaly"
        predictions["severity"] = _compute_severity(predictions["anomaly_score"], is_anomaly)

        block_metadata = _load_block_metadata()
        joined = predictions.merge(block_metadata, on="block_id", how="left")

        _cache["version"] = version
        _cache["predictions"] = joined
        return joined


def invalidate_prediction_cache():
    with _cache_lock:
        _cache["version"] = None
        _cache["predictions"] = None


# ---------------------------------------------------------------------------
# GET /api/dashboard
# ---------------------------------------------------------------------------

@pipeline_bp.get("/api/dashboard")
def dashboard():
    stats = _load_dataset_stats()
    predictions = get_current_predictions()
    current = resolve_current_deployment()
    current_metrics = _metrics_for(current["metrics_path"]) or {}

    total_events = stats.get("total_events", 0)
    total_blocks = len(predictions)
    anomalous = int((predictions["predicted_label"] == "Anomaly").sum())

    cm = current_metrics.get("confusion_matrix")
    fp_rate = None
    if cm:
        matrix = cm["matrix"]
        tn, fp = matrix[0][0], matrix[0][1]
        fp_rate = _clean_float(fp / (fp + tn)) if (fp + tn) else None

    events_per_day = stats.get("events_per_day", {})
    anomalies_by_day = (
        predictions.assign(day=predictions["first_seen"].dt.date.astype("string"))
        .groupby("day")["predicted_label"]
        .apply(lambda s: int((s == "Anomaly").sum()))
    )
    volume_by_day = [
        {
            "day": day,
            "logs": int(count),
            "anomalies": int(anomalies_by_day.get(day, 0)),
        }
        for day, count in sorted(events_per_day.items())
    ]

    severity_counts = predictions.loc[predictions["predicted_label"] == "Anomaly", "severity"].value_counts()
    severity_mix = [
        {"name": name.capitalize(), "value": int(severity_counts.get(name, 0))}
        for name in ["critical", "high", "medium"]
        if severity_counts.get(name, 0) > 0
    ]

    history = _read_json(DEFAULT_VERSION_HISTORY_PATH) or []
    model_evolution = []
    for version in range(1, current["version"] + 1):
        if version == 1:
            metrics = _read_json(DEFAULT_CURRENT_METRICS_PATH)
        else:
            metrics = _read_json(EVALUATION_DIR / f"isolation_forest_v{version}_metrics.json")
        model_evolution.append(
            {
                "version": f"V{version}",
                "accuracy_pct": _clean_float((metrics or {}).get("accuracy", 0) * 100),
                "active": version == current["version"],
            }
        )

    recent_anomalies = (
        predictions[predictions["predicted_label"] == "Anomaly"]
        .sort_values("anomaly_score", ascending=False)
        .head(5)
    )
    activity = [
        {
            "type": "anomaly",
            "text": f"Anomaly detected in **{row.component}** — block {row.block_id}, score {row.anomaly_score:.2f}",
            "timestamp": row.first_seen.isoformat(),
        }
        for row in recent_anomalies.itertuples()
    ]
    for entry in history[-5:]:
        activity.append(
            {
                "type": "promotion" if entry["promoted"] else "rejection",
                "text": (
                    f"Candidate **promoted** to V{entry['new_version']} ({entry['metric']}: "
                    f"{entry['candidate_metric_value']:.4f})"
                    if entry["promoted"]
                    else f"Candidate rejected — {entry['metric']} {entry['candidate_metric_value']:.4f} "
                    f"did not beat current {entry['current_metric_value']:.4f}"
                ),
                "timestamp": entry["timestamp"],
            }
        )
    activity.sort(key=lambda a: a["timestamp"], reverse=True)

    return jsonify(
        {
            "kpis": {
                "total_logs_processed": int(total_events),
                "total_blocks_analyzed": int(total_blocks),
                "anomalies_detected": anomalous,
                "anomaly_rate_pct": _clean_float((anomalous / total_blocks * 100) if total_blocks else 0),
                "current_version": current["version"],
                "false_positive_rate_pct": _clean_float(fp_rate * 100) if fp_rate is not None else None,
            },
            "volume_by_day": volume_by_day,
            "severity_mix": severity_mix,
            "model_evolution": model_evolution,
            "recent_activity": activity[:8],
            "date_range": stats.get("date_range"),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/detections, /api/detections/summary
# ---------------------------------------------------------------------------

@pipeline_bp.get("/api/detections/summary")
def detections_summary():
    predictions = get_current_predictions()
    current = resolve_current_deployment()
    model = joblib.load(current["model_path"])

    total = len(predictions)
    anomalous = int((predictions["predicted_label"] == "Anomaly").sum())
    normal = total - anomalous
    avg_score = _clean_float(predictions["anomaly_score"].mean())

    return jsonify(
        {
            "total_analyzed": total,
            "normal_count": normal,
            "anomalous_count": anomalous,
            "avg_anomaly_score": avg_score,
            "contamination_threshold": _clean_float(model.contamination),
        }
    )


@pipeline_bp.get("/api/detections")
def detections():
    predictions = get_current_predictions()

    search = (request.args.get("search") or "").strip().lower()
    severity = (request.args.get("severity") or "all").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    limit = min(200, max(1, int(request.args.get("limit", 25))))

    filtered = predictions
    if severity != "all":
        filtered = filtered[filtered["severity"] == severity]
    if search:
        mask = (
            filtered["block_id"].str.lower().str.contains(search, na=False)
            | filtered["component"].str.lower().str.contains(search, na=False)
            | filtered["event_type"].str.lower().str.contains(search, na=False)
            | filtered["source_ip"].str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]

    filtered = filtered.sort_values("anomaly_score", ascending=False)
    total = len(filtered)
    start = (page - 1) * limit
    page_rows = filtered.iloc[start : start + limit]

    items = [
        {
            "block_id": row.block_id,
            "timestamp": row.first_seen.isoformat() if pd.notna(row.first_seen) else None,
            "component": row.component,
            "event_type": row.event_type,
            "source_ip": row.source_ip,
            "destination_ip": row.destination_ip,
            "anomaly_score": _clean_float(row.anomaly_score),
            "severity": row.severity,
            "predicted_label": row.predicted_label,
            "true_label": row.true_label,
        }
        for row in page_rows.itertuples()
    ]

    return jsonify({"items": items, "total": total, "page": page, "limit": limit})


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

@pipeline_bp.get("/api/analytics")
def analytics():
    stats = _load_dataset_stats()
    predictions = get_current_predictions()

    def top_n(distribution: dict, n: int) -> list[dict]:
        ordered = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        return [{"name": name, "value": int(count)} for name, count in ordered[:n]]

    component_distribution = top_n(stats.get("component_distribution", {}), 9)
    event_type_distribution = top_n(stats.get("event_type_distribution", {}), 10)
    log_level_distribution = [
        {"name": name, "value": int(count)}
        for name, count in stats.get("log_level_distribution", {}).items()
    ]

    features_path = Path("data/features/features.csv")
    hour_counts = (
        pd.read_csv(features_path, usecols=["hour_of_day"])["hour_of_day"]
        .value_counts()
        .sort_index()
    )
    hour_of_day_distribution = [
        {"hour": int(hour), "count": int(count)} for hour, count in hour_counts.items()
    ]

    predictions = predictions.assign(day=predictions["first_seen"].dt.date.astype("string"))
    day_label_counts = (
        predictions.groupby(["day", "predicted_label"]).size().unstack(fill_value=0).reset_index()
    )
    day_label_counts = day_label_counts.rename(columns={"Anomaly": "anomalies", "Normal": "normal"})
    for column in ["anomalies", "normal"]:
        if column not in day_label_counts.columns:
            day_label_counts[column] = 0
    anomalies_by_day = day_label_counts[["day", "anomalies", "normal"]].to_dict("records")

    top_components = [c["name"] for c in component_distribution[:TOP_N_COMPONENTS]]
    anomaly_rows = predictions[predictions["predicted_label"] == "Anomaly"]
    by_day_component = (
        anomaly_rows[anomaly_rows["component"].isin(top_components)]
        .groupby(["day", "component"])
        .size()
        .unstack(fill_value=0)
    )
    anomalies_by_component_by_day = []
    for day, row in by_day_component.iterrows():
        record = {"day": day}
        record.update({component: int(row.get(component, 0)) for component in top_components})
        anomalies_by_component_by_day.append(record)

    return jsonify(
        {
            "component_distribution": component_distribution,
            "event_type_distribution": event_type_distribution,
            "log_level_distribution": log_level_distribution,
            "hour_of_day_distribution": hour_of_day_distribution,
            "anomalies_by_day": anomalies_by_day,
            "anomalies_by_component_by_day": anomalies_by_component_by_day,
            "top_components": top_components,
            "date_range": stats.get("date_range"),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

def _timeline_from_history(history: list[dict]) -> list[dict]:
    timeline = []
    for entry in reversed(history):
        if entry["promoted"]:
            timeline.append(
                {
                    "tone": "success",
                    "title": f"V{entry['new_version']} promoted from candidate",
                    "meta": entry["timestamp"],
                    "desc": (
                        f"{entry['metric']} improved {entry['current_metric_value']:.4f} -> "
                        f"{entry['candidate_metric_value']:.4f}"
                    ),
                }
            )
        else:
            timeline.append(
                {
                    "tone": "danger",
                    "title": f"Candidate rejected (current: V{entry['current_version']})",
                    "meta": entry["timestamp"],
                    "desc": (
                        f"{entry['metric']} {entry['candidate_metric_value']:.4f} did not beat "
                        f"current {entry['current_metric_value']:.4f}"
                    ),
                }
            )
    return timeline


@pipeline_bp.get("/api/models")
def models():
    current = resolve_current_deployment()
    current_metrics = _metrics_for(current["metrics_path"])
    history = _read_json(DEFAULT_VERSION_HISTORY_PATH) or []
    ga_best_config = _read_json(GA_BEST_CONFIG_PATH)

    version_rows = []
    for version in range(1, current["version"] + 1):
        metrics_path = DEFAULT_CURRENT_METRICS_PATH if version == 1 else EVALUATION_DIR / f"isolation_forest_v{version}_metrics.json"
        metrics = _read_json(metrics_path) or {}
        promoted_at = None
        for entry in history:
            if entry.get("new_version") == version:
                promoted_at = entry["timestamp"]
        version_rows.append(
            {
                "version": f"V{version}",
                "accuracy_pct": _clean_float(metrics.get("accuracy", 0) * 100),
                "f1": _clean_float(metrics.get("f1")),
                "promoted_at": promoted_at,
                "status": "active" if version == current["version"] else "retired",
            }
        )

    job = _latest_job()

    return jsonify(
        {
            "current": {
                "version": current["version"],
                "metrics": current_metrics,
                "promoted_at": current.get("promoted_at"),
                "model_path": current["model_path"],
            },
            "history_table": version_rows,
            "timeline": _timeline_from_history(history),
            "ga_best_config": ga_best_config,
            "job": job,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/models/retrain, GET /api/models/retrain/<job_id>
# ---------------------------------------------------------------------------

_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_active_job_id: str | None = None


def _latest_job():
    with _jobs_lock:
        if _active_job_id is None:
            return None
        return _jobs.get(_active_job_id)


def _run_retrain_job(job_id: str):
    with _jobs_lock:
        _jobs[job_id]["state"] = "running"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        current = resolve_current_deployment()
        current_model_path = current["model_path"]
        current_metrics_path = current["metrics_path"]

        merged = load_labeled_features()
        fitness_sample_size = min(FITNESS_SAMPLE_SIZE, len(merged))
        fitness_data = stratified_subsample(merged, fitness_sample_size)

        _, result = run_ga(
            X=fitness_data[TRAINING_FEATURE_COLUMNS],
            y=fitness_data["Label"],
            population_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
        )

        best = result.best_chromosome
        selected_features = best.selected_features
        model = build_model(best)
        model.fit(merged[selected_features])

        CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CANDIDATE_MODEL_PATH)

        candidate_predictions = predict(
            model, merged, output_path=CANDIDATE_PREDICTIONS_PATH, feature_columns=selected_features
        )
        candidate_metrics = evaluate(candidate_predictions, metrics_path=CANDIDATE_METRICS_PATH)
        print_metrics_table(candidate_metrics)

        decision = evaluate_and_promote(
            current_model_path=current_model_path,
            candidate_model_path=CANDIDATE_MODEL_PATH,
            current_metrics_path=current_metrics_path,
            candidate_metrics_path=CANDIDATE_METRICS_PATH,
            metric="f1",
        )

        invalidate_prediction_cache()

        with _jobs_lock:
            _jobs[job_id]["state"] = "done"
            _jobs[job_id]["result"] = decision
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the polling client
        with _jobs_lock:
            _jobs[job_id]["state"] = "error"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@pipeline_bp.post("/api/models/retrain")
def start_retrain():
    global _active_job_id
    with _jobs_lock:
        if _active_job_id is not None and _jobs[_active_job_id]["state"] == "running":
            return jsonify({"error": "A retrain job is already running.", "job_id": _active_job_id}), 409

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"job_id": job_id, "state": "queued", "result": None, "error": None}
        _active_job_id = job_id

    thread = threading.Thread(target=_run_retrain_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id}), 202


@pipeline_bp.get("/api/models/retrain/<job_id>")
def retrain_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(job)
