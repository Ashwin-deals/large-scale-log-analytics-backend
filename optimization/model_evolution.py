"""
Sprint 6: model evolution / deploy-if-better decision.

Compares a freshly trained "candidate" model against whatever is currently
deployed, on a single chosen metric (higher is better; F1 by default given
the class imbalance seen throughout this project). If the candidate wins,
it's promoted to be the new deployed "current" model; if not, current stays
deployed and the candidate is archived rather than discarded.

Versioning scheme
------------------
data/models/current_version.json is a pointer file: it always names
whichever model/metrics pair is actually deployed right now. Promoted
models are copied (never moved -- prior versions stay on disk) to
data/models/isolation_forest_v{N}.pkl / data/evaluation/isolation_forest_v{N}_metrics.json,
where N increments from whatever current_version.json currently says (or
from BOOTSTRAP_VERSION if this is the first evaluation ever).

data/models/version_history.json is append-only: every call to
evaluate_and_promote() adds one record, whether or not it promoted, so the
full decision history survives even when nothing changes.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_EVALUATION_DIR = Path("data/evaluation")
DEFAULT_ARCHIVE_DIR = DEFAULT_MODELS_DIR / "archive"
DEFAULT_CURRENT_VERSION_PATH = DEFAULT_MODELS_DIR / "current_version.json"
DEFAULT_VERSION_HISTORY_PATH = DEFAULT_MODELS_DIR / "version_history.json"

BOOTSTRAP_VERSION = 1


def _load_metric(metrics_path: str | Path, metric: str) -> float:
    with Path(metrics_path).open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if metric not in metrics:
        raise KeyError(
            f"Metric '{metric}' not found in {metrics_path} (available: {list(metrics.keys())})"
        )
    return float(metrics[metric])


def _load_current_version(current_version_path: Path) -> int:
    if not current_version_path.exists():
        return BOOTSTRAP_VERSION
    with current_version_path.open("r", encoding="utf-8") as version_file:
        return int(json.load(version_file)["version"])


def _write_current_version(
    current_version_path: Path,
    version: int,
    model_path: str | Path,
    metrics_path: str | Path,
    metric: str,
    metric_value: float,
    timestamp: str,
) -> None:
    current_version_path.parent.mkdir(parents=True, exist_ok=True)
    with current_version_path.open("w", encoding="utf-8") as version_file:
        json.dump(
            {
                "version": version,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "metric": metric,
                "metric_value": metric_value,
                "promoted_at": timestamp,
            },
            version_file,
            indent=2,
        )


def _append_version_history(version_history_path: Path, entry: dict) -> None:
    history = []
    if version_history_path.exists():
        with version_history_path.open("r", encoding="utf-8") as history_file:
            history = json.load(history_file)
    history.append(entry)
    version_history_path.parent.mkdir(parents=True, exist_ok=True)
    with version_history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


def evaluate_and_promote(
    current_model_path: str | Path,
    candidate_model_path: str | Path,
    current_metrics_path: str | Path,
    candidate_metrics_path: str | Path,
    metric: str = "f1",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
    archive_dir: str | Path = DEFAULT_ARCHIVE_DIR,
    current_version_path: str | Path = DEFAULT_CURRENT_VERSION_PATH,
    version_history_path: str | Path = DEFAULT_VERSION_HISTORY_PATH,
) -> dict:
    """
    Loads current_metrics_path and candidate_metrics_path, compares their
    `metric` values (candidate must be strictly better to promote), and:
      - promoted: copies the candidate model/metrics to the next version
        slot and repoints current_version.json at it.
      - not promoted: copies the candidate model into archive_dir (so it
        isn't silently lost/overwritten by the next run) and leaves the
        current deployment untouched.
    Either way, one record is appended to version_history.json.
    Returns the decision record that was appended.
    """
    models_dir = Path(models_dir)
    evaluation_dir = Path(evaluation_dir)
    archive_dir = Path(archive_dir)
    current_version_path = Path(current_version_path)
    version_history_path = Path(version_history_path)

    current_metric_value = _load_metric(current_metrics_path, metric)
    candidate_metric_value = _load_metric(candidate_metrics_path, metric)
    current_version = _load_current_version(current_version_path)
    timestamp = datetime.now(timezone.utc)
    timestamp_iso = timestamp.isoformat()

    promoted = candidate_metric_value > current_metric_value

    decision = {
        "timestamp": timestamp_iso,
        "metric": metric,
        "promoted": promoted,
        "current_version": current_version,
        "current_model_path": str(current_model_path),
        "current_metric_value": current_metric_value,
        "candidate_model_path": str(candidate_model_path),
        "candidate_metric_value": candidate_metric_value,
        "new_version": None,
        "promoted_model_path": None,
        "promoted_metrics_path": None,
        "archived_candidate_path": None,
    }

    if promoted:
        new_version = current_version + 1
        new_model_path = models_dir / f"isolation_forest_v{new_version}.pkl"
        new_metrics_path = evaluation_dir / f"isolation_forest_v{new_version}_metrics.json"

        models_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_model_path, new_model_path)
        shutil.copy2(candidate_metrics_path, new_metrics_path)

        _write_current_version(
            current_version_path,
            version=new_version,
            model_path=new_model_path,
            metrics_path=new_metrics_path,
            metric=metric,
            metric_value=candidate_metric_value,
            timestamp=timestamp_iso,
        )

        decision["new_version"] = new_version
        decision["promoted_model_path"] = str(new_model_path)
        decision["promoted_metrics_path"] = str(new_metrics_path)
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename_timestamp = timestamp.strftime("%Y%m%dT%H%M%S%f")
        archived_path = archive_dir / f"{Path(candidate_model_path).stem}_rejected_{filename_timestamp}.pkl"
        shutil.copy2(candidate_model_path, archived_path)

        if not current_version_path.exists():
            _write_current_version(
                current_version_path,
                version=current_version,
                model_path=current_model_path,
                metrics_path=current_metrics_path,
                metric=metric,
                metric_value=current_metric_value,
                timestamp=timestamp_iso,
            )

        decision["archived_candidate_path"] = str(archived_path)

    _append_version_history(version_history_path, decision)
    return decision
