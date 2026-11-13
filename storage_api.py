"""Flask Blueprint for the smart-storage policy, every route behind token_required.

  GET  /api/storage/summary   holdout savings/retention, tier counts, baseline comparison
  GET  /api/storage/policy    the tuned policy (weights, thresholds, summary fraction)
  POST /api/storage/simulate  re-cost the policy with overridden parameters

The policy and report are produced offline by scripts/optimize_storage_policy.py;
these routes read what it wrote and 404 until it has been run. Simulation scores
blocks with the model the policy was tuned on (the tune-half refit named in
policy_best.json, not the deployed model), and runs against the labeled dataset
blocks only -- uploads carry no ground truth, so retention can't be computed.
"""

import json
import re
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from auth import token_required
from detection.train import DEFAULT_FEATURES_PATH, DEFAULT_LABELS_PATH
import joblib
import pandas as pd

from smart_storage.data import POLICY_PATH, REPORT_PATH, load_scored_blocks, resolve_deployed_model, score_blocks
from source_context import SourceContext, load_labeled, register_error_handlers, request_context
from smart_storage.optimizer import stratified_split
from smart_storage.policy import (
    PENALTY_WEIGHT,
    WARM_CREDIT,
    WARM_REF_FRACTION,
    WEIGHT_NAMES,
    BlockTable,
    Normalizer,
    StoragePolicy,
    evaluate_policy,
)

storage_bp = Blueprint("storage", __name__)
register_error_handlers(storage_bp)

SPLITS = ("holdout", "tune", "all")

_cache_lock = threading.Lock()
_caches: dict[str, dict] = {}


class StorageNotReady(Exception):
    pass


class BadRequest(Exception):
    pass


def _mtime(path: Path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def _rerun_hint(ctx: SourceContext | None) -> str:
    if ctx is None or ctx.is_hdfs:
        return "python3 scripts/optimize_storage_policy.py"
    return f"python3 scripts/run_source_pipeline.py --source {ctx.name}"


def _read_json(path: Path, ctx: SourceContext | None = None) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StorageNotReady(f"{path} not found. Run `{_rerun_hint(ctx)}` first.")
    except json.JSONDecodeError as exc:
        raise StorageNotReady(f"{path} is unreadable ({exc}). Re-run `{_rerun_hint(ctx)}`.")


def _paths(ctx: SourceContext) -> tuple[Path, Path]:
    """(policy, report) for a source. HDFS keeps the module-level paths so tests can patch them."""
    if ctx.is_hdfs:
        return Path(POLICY_PATH), Path(REPORT_PATH)
    return ctx.policy_path, ctx.storage_report_path


def _deployed_version(ctx: SourceContext) -> int | None:
    if ctx.is_hdfs:
        return resolve_deployed_model()["version"]
    if not ctx.current_version_path.exists():
        return 1
    return json.loads(ctx.current_version_path.read_text(encoding="utf-8")).get("version")


def _is_stale(policy_payload: dict, ctx: SourceContext) -> bool:
    version = policy_payload.get("model", {}).get("version")
    # Per-source policies record the model path rather than a version number.
    return version is not None and version != _deployed_version(ctx)


_VERSION_IN_PATH = re.compile(r"_v(\d+)\.pkl$")


def _model_for_display(model: dict) -> dict:
    """Per-source policies record which deployed model their hyperparameters
    came from but not its version number; read it from that file name."""
    if model.get("version") is not None:
        return model
    match = _VERSION_IN_PATH.search(str(model.get("hyperparameters_from", "")))
    return {**model, "version": int(match.group(1)) if match else None}


def _scored_frame(model_path: str, ctx: SourceContext | None = None):
    if ctx is None or ctx.is_hdfs:
        return load_scored_blocks(model_path)
    merged = load_labeled(ctx).reset_index(drop=True)
    return score_blocks(joblib.load(model_path), merged, vocab_path=ctx.vocab_path)


def _get_tables(policy_payload: dict, ctx: SourceContext | None = None) -> dict:
    """BlockTables for 'all' / 'tune' / 'holdout', built once per (scoring
    model, source data, policy file) and reused across simulate calls."""
    from source_context import get_context

    ctx = ctx or get_context("hdfs")
    model_path = policy_payload["model"]["model_path"]
    policy_path, _ = _paths(ctx)
    features_path = DEFAULT_FEATURES_PATH if ctx.is_hdfs else ctx.features_path
    labels_path = DEFAULT_LABELS_PATH if ctx.is_hdfs else ctx.labels_path
    key = (
        model_path,
        _mtime(model_path),
        _mtime(features_path),
        _mtime(labels_path),
        _mtime(policy_path),
    )
    cache = _caches.setdefault(ctx.name, {"key": None, "tables": None})
    with _cache_lock:
        if cache["key"] == key and cache["tables"] is not None:
            return cache["tables"]
        if not ctx.is_hdfs:
            ctx.require(features_path, labels_path)

        if _mtime(model_path) is None:
            raise StorageNotReady(f"{model_path} not found. Re-run `{_rerun_hint(ctx)}`.")
        frame = _scored_frame(model_path) if ctx.is_hdfs else _scored_frame(model_path, ctx)
        normalizer = Normalizer.from_dict(policy_payload["normalizer"])
        split = policy_payload["split"]
        full = BlockTable.from_frame(frame, normalizer)
        in_sample = split.get("protocol") == "in_sample" or str(policy_payload["model"].get("fitted_on", "")).startswith("all blocks")
        if in_sample:
            tables = {"all": full, "tune": full, "holdout": full}
        else:
            tune_idx, holdout_idx = stratified_split(frame["true_label"], split["test_size"], split["seed"])
            tables = {"all": full, "tune": full.subset(tune_idx), "holdout": full.subset(holdout_idx)}

        cache["key"], cache["tables"] = key, tables
        return tables


def _number(body: dict, name: str, default: float) -> float:
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(f"{name} must be a number")
    return float(value)


def _apply_overrides(base: StoragePolicy, body: dict) -> StoragePolicy:
    weights = body.get("weights", {})
    if not isinstance(weights, dict):
        raise BadRequest("weights must be an object keyed by " + ", ".join(WEIGHT_NAMES))
    unknown = set(weights) - set(WEIGHT_NAMES)
    if unknown:
        raise BadRequest(f"unknown weight(s): {', '.join(sorted(unknown))}")
    base_weights = dict(zip(WEIGHT_NAMES, base.weights))
    merged = {name: _number(weights, name, base_weights[name]) for name in WEIGHT_NAMES}

    policy = StoragePolicy(
        weight_anomaly_score=merged["anomaly_score"],
        weight_event_frequency=merged["event_frequency"],
        weight_block_size=merged["block_size"],
        t_hot=_number(body, "t_hot", base.t_hot),
        t_warm=_number(body, "t_warm", base.t_warm),
        summary_fraction=_number(body, "summary_fraction", base.summary_fraction),
    )
    try:
        return policy.validate()
    except ValueError as exc:
        raise BadRequest(str(exc))


@storage_bp.errorhandler(StorageNotReady)
def _not_ready(exc):
    return jsonify({"error": str(exc)}), 404


@storage_bp.errorhandler(BadRequest)
def _bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@storage_bp.route("/api/storage/policy", methods=["GET"])
@token_required
def get_policy():
    ctx = request_context()
    payload = _read_json(_paths(ctx)[0], ctx)
    return jsonify(
        {
            "policy": payload["policy"],
            "retention_target": payload["retention_target"],
            "model": _model_for_display(payload["model"]),
            "generated_at": payload["generated_at"],
            "stale": _is_stale(payload, ctx),
            "source": ctx.name,
        }
    )


@storage_bp.route("/api/storage/summary", methods=["GET"])
@token_required
def get_summary():
    ctx = request_context()
    report = _read_json(_paths(ctx)[1], ctx)
    for section in (report, report.get("deployed_model_scoring") or {}):
        history = section.pop("ga_history", {})
        section["ga_summary"] = {
            key: history.get(key)
            for key in ("population_size", "num_generations", "generations_completed", "wall_clock_seconds")
        }
    report["deployed_model_version"] = _deployed_version(ctx)
    # A policy is tuned against one model's score distribution; once a newer
    # version is promoted the numbers here describe a model no longer live.
    report["stale"] = _is_stale(report, ctx)
    report["source"] = {"name": ctx.name, "title": ctx.title, "label_kind": ctx.label_kind, "block_noun": ctx.block_noun}
    if "model" in report:
        report["model"] = _model_for_display(report["model"])
    return jsonify(report)


@storage_bp.route("/api/storage/simulate", methods=["POST"])
@token_required
def simulate():
    """Body (all fields optional; omitted ones keep the tuned policy's value):

        {"weights": {"anomaly_score": 0.8, "event_frequency": 0.1, "block_size": 0.1},
         "t_hot": 0.14, "t_warm": 0.10, "summary_fraction": 0.05,
         "retention_target": 0.95, "warm_credit": 0.5, "warm_ref_fraction": 0.25,
         "split": "holdout" | "tune" | "all"}
    """
    body = request.get_json(silent=True)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")

    split = body.get("split", "holdout")
    if split not in SPLITS:
        raise BadRequest(f"split must be one of {', '.join(SPLITS)}")

    ctx = request_context()
    policy_payload = _read_json(_paths(ctx)[0], ctx)
    target = _number(body, "retention_target", policy_payload["retention_target"])
    if not 0.0 <= target <= 1.0:
        raise BadRequest("retention_target must be between 0 and 1")
    warm_credit = _number(body, "warm_credit", policy_payload.get("warm_credit", WARM_CREDIT))
    if not 0.0 <= warm_credit <= 1.0:
        raise BadRequest("warm_credit must be between 0 and 1")
    warm_ref_fraction = _number(body, "warm_ref_fraction", policy_payload.get("warm_ref_fraction", WARM_REF_FRACTION))
    if not 0.0 < warm_ref_fraction <= 1.0:
        raise BadRequest("warm_ref_fraction must be greater than 0 and at most 1")

    policy = _apply_overrides(StoragePolicy.from_dict(policy_payload["policy"]), body)
    outcome = evaluate_policy(_get_tables(policy_payload, ctx)[split], policy, warm_credit, warm_ref_fraction)

    return jsonify(
        {
            "split": split,
            "policy": policy.to_dict(),
            "retention_target": target,
            "warm_credit": warm_credit,
            "warm_ref_fraction": warm_ref_fraction,
            "meets_retention_target": outcome.meets_target(target),
            "fitness": outcome.fitness(target, policy_payload.get("penalty_weight", PENALTY_WEIGHT)),
            "outcome": outcome.to_dict(),
            "stale": _is_stale(policy_payload, ctx),
            "source": ctx.name,
        }
    )
