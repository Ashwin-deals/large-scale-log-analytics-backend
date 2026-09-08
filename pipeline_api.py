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

from auth import token_required

from detection.isolation_forest import save_model
from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from detection.train import DEFAULT_FEATURES_PATH, load_labeled_features
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


def _int_arg(name, default):
    """Read an int query param, falling back to the default when it isn't one.

    int() on a malformed value raises, which the error handler turns into a
    500; a bad page number should just fall back rather than break the page.
    """
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


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
    "data_key": None,  # mtimes of the files the predictions were built from
    "predictions": None,  # DataFrame: block_id, true_label, predicted_label, anomaly_score, severity
    "block_metadata": None,  # DataFrame
    "block_metadata_mtime": None,
    "dataset_stats": None,
    "dataset_stats_mtime": None,
}


def _mtime(path: Path):
    """Modification time, or None when the file isn't there yet."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _load_block_metadata() -> pd.DataFrame:
    # Keyed on mtime: the pipeline scripts rewrite these files, and caching on
    # first read alone served stale numbers until the server was restarted.
    mtime = _mtime(BLOCK_METADATA_PATH)
    if _cache["block_metadata"] is None or _cache["block_metadata_mtime"] != mtime:
        frame = pd.read_csv(BLOCK_METADATA_PATH)
        frame["first_seen"] = pd.to_datetime(frame["first_seen"])
        _cache["block_metadata"] = frame
        _cache["block_metadata_mtime"] = mtime
    return _cache["block_metadata"]


def _load_dataset_stats() -> dict:
    mtime = _mtime(DATASET_STATS_PATH)
    if _cache["dataset_stats"] is None or _cache["dataset_stats_mtime"] != mtime:
        # A missing file caches as {}, so without the mtime check the API kept
        # reporting zero logs processed after the stats were generated.
        _cache["dataset_stats"] = _read_json(DATASET_STATS_PATH) or {}
        _cache["dataset_stats_mtime"] = mtime
    return _cache["dataset_stats"]


# ---------------------------------------------------------------------------
# Uploaded blocks folded into the dashboard totals
# ---------------------------------------------------------------------------

UPLOAD_BLOCKS_DIR = Path("data/uploads/results")
UPLOAD_BLOCKS_GLOB = "*_blocks.csv"
UPLOAD_BLOCK_COLUMNS = [
    "block_id", "first_seen", "block_size", "component", "event_type",
    "source_ip", "destination_ip", "event_frequency", "predicted_label", "anomaly_score",
]


def _upload_blocks_key() -> tuple:
    """Cache key covering every processed upload's block record file, so a new
    upload shows up in the dashboard on the next request instead of waiting
    for a model promotion or a server restart."""
    try:
        paths = sorted(UPLOAD_BLOCKS_DIR.glob(UPLOAD_BLOCKS_GLOB))
    except OSError:
        return ()
    return tuple((str(path), _mtime(path)) for path in paths)


def _load_upload_blocks() -> pd.DataFrame:
    """Every uploaded block scored so far, one row per distinct block_id.

    Deduplicated across uploads: a block_id identifies one real HDFS block, so
    re-uploading a log that covers it must not make the cluster look bigger
    than it is. First occurrence wins; the model is deterministic, so a repeat
    scores identically anyway.
    """
    frames = []
    for path in sorted(UPLOAD_BLOCKS_DIR.glob(UPLOAD_BLOCKS_GLOB)):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            # A half-written or corrupt upload record must not take the whole
            # dashboard down; skip it and keep serving the rest.
            continue

    if not frames:
        return pd.DataFrame(columns=UPLOAD_BLOCK_COLUMNS)

    blocks = pd.concat(frames, ignore_index=True)
    missing = [column for column in UPLOAD_BLOCK_COLUMNS if column not in blocks.columns]
    if missing:
        return pd.DataFrame(columns=UPLOAD_BLOCK_COLUMNS)

    blocks = blocks[UPLOAD_BLOCK_COLUMNS].drop_duplicates(subset=["block_id"], keep="first")
    blocks["first_seen"] = pd.to_datetime(blocks["first_seen"], errors="coerce")
    return blocks.reset_index(drop=True)


HEALTHY_MIN_F1 = 0.40


def _model_status(current: dict, metrics: dict) -> dict:
    """Describe the deployed model's health from what is actually on disk.

    The dashboard used to render "Active / Healthy" as a literal, so it read
    the same whether the model was sound or scoring near zero.
    """
    model_path = Path(current.get("model_path", ""))
    if not model_path.exists():
        return {"state": "unavailable", "tone": "danger",
                "detail": f"Model file missing: {model_path}"}

    f1 = metrics.get("f1")
    if f1 is None:
        return {"state": "unverified", "tone": "warning",
                "detail": "No evaluation metrics recorded for the deployed model."}

    if f1 < HEALTHY_MIN_F1:
        return {"state": "degraded", "tone": "danger",
                "detail": f"F1 {f1:.4f} is below the {HEALTHY_MIN_F1:.2f} threshold."}

    return {"state": "healthy", "tone": "success",
            "detail": f"F1 {f1:.4f} on {metrics.get('support', {}).get('total', 0):,} blocks."}


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
    # Rebuilding features.csv or block_metadata.csv changes the predictions
    # even when the deployed version hasn't moved, so both count towards the
    # cache key alongside the version — as does any newly processed upload.
    data_key = (_mtime(DEFAULT_FEATURES_PATH), _mtime(BLOCK_METADATA_PATH), _upload_blocks_key())

    with _cache_lock:
        if _cache["version"] == version and _cache["data_key"] == data_key and _cache["predictions"] is not None:
            return _cache["predictions"]

        model = joblib.load(current["model_path"])
        merged = load_labeled_features()
        feature_columns = list(model.feature_names_in_)
        predictions = predict(model, merged, output_path=CANDIDATE_PREDICTIONS_PATH.parent / "_live_predictions.csv", feature_columns=feature_columns)

        block_metadata = _load_block_metadata()
        joined = predictions.merge(block_metadata, on="block_id", how="left")
        joined["source"] = "dataset"
        joined["event_frequency"] = pd.NA

        # Uploaded blocks the base dataset has never seen extend the totals;
        # ones it already contains are dropped rather than counted twice (the
        # same block scored by the same model gives the same answer either way).
        uploaded = _load_upload_blocks()
        if len(uploaded):
            new_blocks = uploaded[~uploaded["block_id"].isin(set(joined["block_id"]))].copy()
            if len(new_blocks):
                new_blocks["true_label"] = pd.NA  # uploads are unlabeled
                new_blocks["source"] = "upload"
                joined = pd.concat([joined, new_blocks[joined.columns]], ignore_index=True)

        # Severity is bucketed after the union so an uploaded anomaly is ranked
        # on the same scale as a base-dataset one.
        is_anomaly = joined["predicted_label"] == "Anomaly"
        joined["severity"] = _compute_severity(joined["anomaly_score"], is_anomaly)

        _cache["version"] = version
        _cache["data_key"] = data_key
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
@token_required
def dashboard():
    stats = _load_dataset_stats()
    predictions = get_current_predictions()
    current = resolve_current_deployment()
    current_metrics = _metrics_for(current["metrics_path"]) or {}

    # Uploads contribute their own raw log lines on top of the base dataset's,
    # counted per block via event_frequency (dataset rows carry no count — the
    # stats file already totals those).
    from_uploads = predictions["source"] == "upload"
    uploaded_events = int(pd.to_numeric(predictions.loc[from_uploads, "event_frequency"], errors="coerce").fillna(0).sum())
    uploaded_blocks = int(from_uploads.sum())

    total_events = stats.get("total_events", 0) + uploaded_events
    total_blocks = len(predictions)
    anomalous = int((predictions["predicted_label"] == "Anomaly").sum())

    cm = current_metrics.get("confusion_matrix")
    fp_rate = None
    if cm:
        matrix = cm["matrix"]
        tn, fp = matrix[0][0], matrix[0][1]
        fp_rate = _clean_float(fp / (fp + tn)) if (fp + tn) else None

    events_per_day = dict(stats.get("events_per_day", {}))
    if uploaded_blocks:
        # Uploaded lines land on the days their blocks were first seen, so a
        # day the base dataset never covered still shows up on the chart.
        upload_lines_by_day = (
            predictions.loc[from_uploads]
            .assign(
                day=predictions.loc[from_uploads, "first_seen"].dt.date.astype("string"),
                lines=pd.to_numeric(predictions.loc[from_uploads, "event_frequency"], errors="coerce").fillna(0),
            )
            .groupby("day")["lines"]
            .sum()
        )
        for day, lines in upload_lines_by_day.items():
            events_per_day[day] = events_per_day.get(day, 0) + int(lines)

    by_day = predictions.assign(day=predictions["first_seen"].dt.date.astype("string")).groupby("day")
    anomalies_by_day = by_day["predicted_label"].apply(lambda s: int((s == "Anomaly").sum()))
    # "logs" counts raw log lines while anomalies are counted per block, so the
    # block total goes out too — without it the two series share an axis while
    # having different denominators, and the chart reads as a false ratio.
    blocks_by_day = by_day.size()
    volume_by_day = [
        {
            "day": day,
            "logs": int(count),
            "blocks": int(blocks_by_day.get(day, 0)),
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
        # Refit entries record no candidate, so they carry no candidate metric.
        if entry.get("event") == "refit":
            changed = ", ".join(
                f"V{v['version']} {v['before']:.4f}->{v['after']:.4f}"
                for v in entry.get("versions", [])
            )
            text = f"Models **refitted** on rebuilt features ({entry['metric']}: {changed})"
            kind = "refit"
        elif entry["promoted"]:
            text = (
                f"Candidate **promoted** to V{entry['new_version']} ({entry['metric']}: "
                f"{entry['candidate_metric_value']:.4f})"
            )
            kind = "promotion"
        else:
            text = (
                f"Candidate rejected — {entry['metric']} {entry['candidate_metric_value']:.4f} "
                f"did not beat current {entry['current_metric_value']:.4f}"
            )
            kind = "rejection"

        activity.append({"type": kind, "text": text, "timestamp": entry["timestamp"]})
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
                "model_status": _model_status(current, current_metrics),
                # What uploads added on top of the base dataset. Reported so a
                # dashboard that hasn't moved after an upload can say why —
                # every block in that file was already in the dataset.
                "uploaded_blocks_added": uploaded_blocks,
                "uploaded_logs_added": uploaded_events,
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
@token_required
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
@token_required
def detections():
    predictions = get_current_predictions()

    search = (request.args.get("search") or "").strip().lower()
    severity = (request.args.get("severity") or "all").strip().lower()
    page = max(1, _int_arg("page", 1))
    limit = min(200, max(1, _int_arg("limit", 25)))

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
            # Uploaded blocks are unlabeled, so this is null for them rather
            # than a NaN that would serialize as invalid JSON.
            "true_label": row.true_label if pd.notna(row.true_label) else None,
            "source": row.source,
        }
        for row in page_rows.itertuples()
    ]

    return jsonify({"items": items, "total": total, "page": page, "limit": limit})


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

@pipeline_bp.get("/api/analytics")
@token_required
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
        # A refit re-fits the existing versions on rebuilt features; it isn't a
        # promotion, and without its own entry the scores recorded by earlier
        # events silently stop matching the metrics shown beside them.
        if entry.get("event") == "refit":
            changes = ", ".join(
                f"V{v['version']} {entry['metric']} {v['before']:.4f} -> {v['after']:.4f}"
                for v in entry.get("versions", [])
            )
            timeline.append(
                {
                    "tone": "primary",
                    "title": "Models refitted on rebuilt features",
                    "meta": entry["timestamp"],
                    "desc": changes or "Refitted against the current feature encoding.",
                }
            )
        elif entry["promoted"]:
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
@token_required
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

        # A fresh seed per run. Pinned to the module default, the GA and the
        # model are fully deterministic, so every retrain over unchanged data
        # reproduced a bit-identical candidate scoring exactly the incumbent's
        # F1 — and promotion requires strictly beating it, so nothing could
        # ever be promoted. Varying the seed lets each run explore differently.
        run_seed = uuid.uuid4().int % (2**31 - 1)
        with _jobs_lock:
            _jobs[job_id]["random_seed"] = run_seed

        fitness_data = stratified_subsample(merged, fitness_sample_size, random_state=run_seed)

        _, result = run_ga(
            X=fitness_data[TRAINING_FEATURE_COLUMNS],
            y=fitness_data["Label"],
            population_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            random_seed=run_seed,
        )

        best = result.best_chromosome
        selected_features = best.selected_features
        model = build_model(best, random_state=run_seed)
        model.fit(merged[selected_features])

        CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, CANDIDATE_MODEL_PATH)

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
@token_required
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
@token_required
def retrain_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(job)
