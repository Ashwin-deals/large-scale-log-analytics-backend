"""Where each log source's files live, so every API can serve any source.

HDFS keeps the original flat layout (data/features/features.csv,
data/models/current_version.json, ...). The other sources use the per-source
layout written by multi_source/pipeline.py (data/features/<source>/,
data/models/<source>/, data/evaluation/<source>_metrics/, ...).

Every API takes ?source=<name> (default hdfs) and resolves it here. A source
whose files are not on this machine answers 404 with a hint saying what to run
(see SourceNotReady), instead of a 500.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from flask import request

SOURCES = ("hdfs", "openstack", "apache", "cloudtrail", "django")
DEFAULT_SOURCE = "hdfs"

SOURCE_TITLES = {
    "hdfs": "HDFS",
    "openstack": "OpenStack",
    "apache": "Apache",
    "cloudtrail": "AWS CloudTrail",
    "django": "Django",
}
# What a source's ground truth actually is -- shown next to every metric so a
# rule-based or synthetic label is never mistaken for a real one.
LABEL_KINDS = {
    "hdfs": "ground truth",
    "openstack": "by run (session-level)",
    "apache": "INJECTED",
    "cloudtrail": "WEAK rule-based",
    "django": "SYNTHETIC",
}
BLOCK_NOUNS = {
    "hdfs": "block",
    "openstack": "VM session",
    "apache": "60-second window",
    "cloudtrail": "identity-hour",
    "django": "user",
}


class UnknownSource(ValueError):
    pass


class SourceNotReady(Exception):
    """A file this request needs is not on this machine."""


@dataclass(frozen=True)
class SourceContext:
    name: str
    features_path: Path
    labels_path: Path
    vocab_path: Path
    cleaned_path: Path
    models_dir: Path
    evaluation_dir: Path
    current_version_path: Path
    version_history_path: Path
    baseline_model_path: Path
    baseline_metrics_path: Path
    block_metadata_path: Path
    dataset_stats_path: Path
    storage_dir: Path
    early_warning_report_path: Path
    source_report_path: Path | None

    @property
    def is_hdfs(self) -> bool:
        return self.name == "hdfs"

    @property
    def title(self) -> str:
        return SOURCE_TITLES[self.name]

    @property
    def label_kind(self) -> str:
        return LABEL_KINDS[self.name]

    @property
    def block_noun(self) -> str:
        return BLOCK_NOUNS[self.name]

    @property
    def candidate_model_path(self) -> Path:
        return self.models_dir / "isolation_forest_candidate.pkl"

    @property
    def candidate_metrics_path(self) -> Path:
        return self.evaluation_dir / "isolation_forest_candidate_metrics.json"

    @property
    def archive_dir(self) -> Path:
        return self.models_dir / "archive"

    @property
    def policy_path(self) -> Path:
        return self.storage_dir / "policy_best.json"

    @property
    def storage_report_path(self) -> Path:
        return self.storage_dir / "storage_report.json"

    def version_metrics_path(self, version: int) -> Path:
        if version == 1:
            return self.baseline_metrics_path
        return self.evaluation_dir / f"isolation_forest_v{version}_metrics.json"

    def source_report(self) -> dict | None:
        if self.source_report_path is None or not self.source_report_path.exists():
            return None
        return json.loads(self.source_report_path.read_text(encoding="utf-8"))

    def prepare_hint(self) -> str:
        if self.is_hdfs:
            return ("Download the data (python3 scripts/download_dataset.py --source hdfs --with-derived) "
                    "or build it (scripts/build_hdfs_features.py, then scripts/build_block_metadata.py).")
        return (f"Download it (python3 scripts/download_dataset.py --source {self.name} --with-derived), "
                f"then run python3 scripts/run_source_pipeline.py --source {self.name}.")

    def require(self, *paths: Path) -> None:
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise SourceNotReady(
                f"{self.title} data is not on this server ({', '.join(missing)} missing). {self.prepare_hint()}"
            )

    def availability(self) -> dict:
        """Which parts of this source can be served from this machine."""
        return {
            "detections": self.features_path.exists() and self.labels_path.exists(),
            "models": self.current_version_path.exists() or self.baseline_metrics_path.exists(),
            "storage": self.storage_report_path.exists(),
            "early_warning": self.early_warning_report_path.exists() or bool(self.source_report_path and self.source_report_path.exists()),
        }


def load_labeled(ctx: SourceContext):
    """features.csv joined with the source's labels, with a Label column.
    HDFS labels are anomaly_label.csv (BlockId, Label); every other source's
    labels.csv is (block_id, Label)."""
    import pandas as pd

    from detection.train import load_labeled_features

    if ctx.is_hdfs:
        return load_labeled_features(ctx.features_path, ctx.labels_path)
    features = pd.read_csv(ctx.features_path)
    labels = pd.read_csv(ctx.labels_path)
    if "BlockId" not in labels.columns:
        labels = labels.rename(columns={"block_id": "BlockId"})
    return features.merge(labels, left_on="block_id", right_on="BlockId", how="inner")


def get_context(name: str | None = None) -> SourceContext:
    name = (name or DEFAULT_SOURCE).strip().lower()
    if name not in SOURCES:
        raise UnknownSource(f"unknown source {name!r}; choose one of {', '.join(SOURCES)}")
    if name == "hdfs":
        return SourceContext(
            name="hdfs",
            features_path=Path("data/features/features.csv"),
            labels_path=Path("data/raw/hdfs/anomaly_label.csv"),
            vocab_path=Path("data/features/category_encoders.json"),
            cleaned_path=Path("data/processed/hdfs_cleaned.csv"),
            models_dir=Path("data/models"),
            evaluation_dir=Path("data/evaluation"),
            current_version_path=Path("data/models/current_version.json"),
            version_history_path=Path("data/models/version_history.json"),
            baseline_model_path=Path("data/models/isolation_forest_v1.pkl"),
            baseline_metrics_path=Path("data/evaluation/baseline_isolation_forest_metrics.json"),
            block_metadata_path=Path("data/processed/block_metadata.csv"),
            dataset_stats_path=Path("data/processed/dataset_stats.json"),
            storage_dir=Path("data/storage"),
            early_warning_report_path=Path("data/evaluation/early_warning_report.json"),
            source_report_path=None,
        )
    evaluation_dir = Path("data/evaluation") / f"{name}_metrics"
    return SourceContext(
        name=name,
        features_path=Path("data/features") / name / "features.csv",
        labels_path=Path("data/features") / name / "labels.csv",
        vocab_path=Path("data/features") / name / "category_encoders.json",
        cleaned_path=Path("data/processed") / name / "cleaned.csv",
        models_dir=Path("data/models") / name,
        evaluation_dir=evaluation_dir,
        current_version_path=Path("data/models") / name / "current_version.json",
        version_history_path=Path("data/models") / name / "version_history.json",
        baseline_model_path=Path("data/models") / name / "isolation_forest_v1.pkl",
        baseline_metrics_path=evaluation_dir / "isolation_forest_v1_metrics.json",
        block_metadata_path=Path("data/processed") / name / "block_metadata.csv",
        dataset_stats_path=Path("data/processed") / name / "dataset_stats.json",
        storage_dir=Path("data/storage") / name,
        # Per-source early-warning results live inside the source report.
        early_warning_report_path=Path("data/evaluation") / f"{name}_report.json",
        source_report_path=Path("data/evaluation") / f"{name}_report.json",
    )


def request_context() -> SourceContext:
    """The source named by ?source= (or a JSON body's "source"), default hdfs."""
    name = request.args.get("source")
    if name is None and request.is_json:
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            name = body.get("source")
    return get_context(name)


def register_error_handlers(blueprint) -> None:
    from flask import jsonify

    @blueprint.errorhandler(UnknownSource)
    def _unknown(exc):
        return jsonify({"error": str(exc), "sources": list(SOURCES)}), 400

    @blueprint.errorhandler(SourceNotReady)
    def _not_ready(exc):
        return jsonify({"error": str(exc), "not_ready": True}), 404
