"""Flask Blueprint for the early-warning detectors, behind token_required.

  GET /api/early-warning/summary   held-out metrics per prefix length k and
                                   the precision-vs-lead-time trade-off

Reads data/evaluation/early_warning_report.json, written offline by
scripts/build_early_warning.py; 404s with that hint until it has been run.
"""

import json
from pathlib import Path

from flask import Blueprint, jsonify

from auth import token_required
from source_context import register_error_handlers, request_context

early_warning_bp = Blueprint("early_warning", __name__)
register_error_handlers(early_warning_bp)

REPORT_PATH = Path("data/evaluation/early_warning_report.json")


def _summary_row(k: str, result: dict) -> dict:
    test = result["test"]
    return {
        "k": int(k),
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "false_alarm_rate": test["false_alarm_rate"],
        "mean_lead_over_true_alerts": test["lead_time"]["mean_over_true_alerts"],
        "expected_lead_per_alert": test["lead_time"]["expected_lead_per_alert"],
        "alerts": test["alerts"],
        "in_progress": {
            **{key: result["test_in_progress"][key] for key in ("blocks", "anomalies", "precision", "recall", "f1")},
            # Event-share lead, plus the same lead in log-time seconds, over correct
            # alerts on blocks still running when the alert fired.
            "mean_lead_over_true_alerts": result["test_in_progress"]["lead_time"]["mean_over_true_alerts"],
            "median_lead_seconds": result["test_in_progress"]["lead_time"].get("median_seconds_over_true_alerts"),
            "mean_lead_seconds": result["test_in_progress"]["lead_time"].get("mean_seconds_over_true_alerts"),
        } if "test_in_progress" in result else None,
    }


@early_warning_bp.route("/api/early-warning/summary", methods=["GET"])
@token_required
def get_summary():
    ctx = request_context()
    # HDFS has its own report; every other source keeps early warning inside
    # data/evaluation/<source>_report.json, in the same shape.
    path = REPORT_PATH if ctx.is_hdfs else ctx.source_report_path
    hint = "python3 scripts/build_early_warning.py" if ctx.is_hdfs else f"python3 scripts/run_source_pipeline.py --source {ctx.name}"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return jsonify({"error": f"{path} not found. Run `{hint}` first.", "not_ready": True}), 404
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"{path} is unreadable ({exc}). Re-run `{hint}`."}), 404
    if not ctx.is_hdfs:
        report = report.get("early_warning") or {}
        if not report.get("results"):
            return jsonify({"error": f"{path} has no early-warning results. Re-run `{hint}`.", "not_ready": True}), 404

    results = report.get("results", {})
    for result in results.values():
        result.pop("ga_history", None)
    report["table"] = sorted((_summary_row(k, r) for k, r in results.items()), key=lambda row: row["k"])
    report["source"] = {"name": ctx.name, "title": ctx.title, "label_kind": ctx.label_kind, "block_noun": ctx.block_noun}
    return jsonify(report)
