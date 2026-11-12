"""Registry of the non-HDFS log sources: where the data is, how to parse it,
where its labels come from, and the caveats every report must carry."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

from parser.apache_parser import DEFAULT_WINDOW_SECONDS, ApacheParser
from parser.cloudtrail_parser import CloudTrailParser
from parser.django_parser import DjangoParser
from parser.openstack_parser import OpenStackParser


@dataclass
class SourceConfig:
    name: str
    title: str
    parser_factory: object
    inputs: object  # callable -> list[Path]
    data_kind: str  # "real" | "real, anomalies INJECTED" | "SYNTHETIC"
    label_kind: str  # "ground truth" | "INJECTED" | "WEAK rule-based" | "SYNTHETIC"
    label_description: str
    block_description: str
    protocol: str = "holdout"  # or "in_sample" when there are too few anomalies to hold any out
    chunked_by_time: bool = False  # stream + clean/extract in time-ordered chunks (blocks never straddle a chunk)
    warnings: list[str] = field(default_factory=list)
    prepare_hint: str = ""


def _files(*patterns):
    return lambda: sorted(Path(p) for pattern in patterns for p in glob.glob(pattern))


SOURCES = {
    "openstack": SourceConfig(
        name="openstack",
        title="OpenStack (nova)",
        parser_factory=OpenStackParser,
        inputs=_files("data/raw/openstack/openstack_normal1.log", "data/raw/openstack/openstack_normal2.log",
                      "data/raw/openstack/openstack_abnormal.log"),
        data_kind="real",
        label_kind="by source file (session-level)",
        label_description="A block (one instance-id session) is Anomaly if the instance appears in openstack_abnormal.log "
                          "(the LogHub abnormal run) and Normal if it appears in openstack_normal1/2.log. No instance appears in "
                          "more than one file.",
        block_description="VM instance UUID from '[instance: <uuid>]' (one session); lines without an instance id are dropped.",
        warnings=[
            "Labels mark which run a session came from, not which session had a fault: the dataset's own "
            "anomaly_labels.txt lists only 4 of the abnormal run's instances as injected with faults. The other abnormal-run "
            "sessions are labeled Anomaly here although the dataset calls them normal. See ground_truth_check for the 4.",
            "The abnormal run is a different day (2017-05-14) from the normal runs (2017-05-16/17), so a model can separate "
            "runs by time-of-day or workload differences rather than faults; see feature_separability (single-feature AUC).",
        ],
        prepare_hint="Copy the OpenStack dataset into data/raw/openstack/ (see data/raw/README.md).",
    ),
    "apache": SourceConfig(
        name="apache",
        title="Apache error log (INJECTED anomalies)",
        parser_factory=lambda: ApacheParser(DEFAULT_WINDOW_SECONDS),
        inputs=_files("data/raw/apache/injected/Apache_INJECTED.log"),
        data_kind="real, anomalies INJECTED",
        label_kind="INJECTED",
        label_description="Labels come from data/raw/apache/injected/Apache_INJECTED_labels.csv: windows where "
                          "scripts/inject_apache_anomalies.py (seed 42) injected an error burst or never-seen templates are "
                          "Anomaly; all other windows are Normal.",
        block_description=f"Fixed {DEFAULT_WINDOW_SECONDS}-second time window (no block id in the log).",
        warnings=[
            "Anomalies are INJECTED, not observed: the detector is measured on synthetic error bursts and unseen templates "
            "added to the real log, not on real Apache incidents.",
            "Windows that were not injected are labeled Normal, but the real log may contain genuine, unlabeled anomalies; "
            "flagging those counts as a false positive here.",
        ],
        prepare_hint="python3 scripts/inject_apache_anomalies.py",
    ),
    "cloudtrail": SourceConfig(
        name="cloudtrail",
        title="AWS CloudTrail (flAWS.cloud, WEAK labels)",
        parser_factory=CloudTrailParser,
        inputs=_files("data/raw/cloudtrail/*.json.gz"),
        data_kind="real",
        label_kind="WEAK rule-based",
        label_description="",  # filled from labels.WEAK_RULE_DESCRIPTION
        block_description="(identity, 1-hour window): the caller's ARN/key/principal, or anonymous@<ip>, per clock hour.",
        chunked_by_time=True,
        warnings=[
            "Labels are WEAK and rule-based, NOT ground truth (see label_description). Precision/recall measure agreement "
            "with the rule, not with real attacks.",
            "The rule keys on authorization errors and new IPs; the model's features (dominant event/component/IP codes, "
            "event count, hour) only partly overlap with it, so disagreement can mean the rule is wrong as often as the model.",
        ],
        prepare_hint="Copy flaws_cloudtrail*.json.gz into data/raw/cloudtrail/ (see data/raw/README.md).",
    ),
    "django": SourceConfig(
        name="django",
        title="Django (SYNTHETIC)",
        parser_factory=DjangoParser,
        inputs=_files("data/raw/django/django_SYNTHETIC.log"),
        data_kind="SYNTHETIC",
        label_kind="SYNTHETIC",
        label_description="Labels come from data/raw/django/django_SYNTHETIC_labels.csv, written by "
                          "scripts/generate_django_logs.py (seed 42): users given an injected fault (500 burst, slow requests, "
                          "DB errors) are Anomaly.",
        block_description="One user ('user-<id>') across the whole generated period.",
        warnings=[
            "All data is SYNTHETIC, generated by scripts/generate_django_logs.py. The faults are designed to be detectable, "
            "so these numbers say the pipeline runs end to end, not how it would do on real Django traffic.",
        ],
        prepare_hint="python3 scripts/generate_django_logs.py",
    ),
}


def get_source(name: str) -> SourceConfig:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; choose from {', '.join(SOURCES)}") from None
