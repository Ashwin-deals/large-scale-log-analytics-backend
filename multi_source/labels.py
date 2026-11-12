"""Block labels for each non-HDFS source.

Only OpenStack has ground truth, and only 4 instances of it. The rest are
INJECTED (Apache), SYNTHETIC (Django) or a WEAK rule (CloudTrail), and every
report says which.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def openstack_anomalous_instances(path: str | Path) -> set[str]:
    """anomaly_labels.txt: a sentence header, then one instance UUID per line (CRLF endings)."""
    return set(UUID_PATTERN.findall(Path(path).read_text(encoding="utf-8", errors="replace")))


def instances_in(path: str | Path) -> set[str]:
    """Every instance UUID mentioned as "[instance: <uuid>]" in one OpenStack log file."""
    pattern = re.compile(r"\[instance:\s*(" + UUID_PATTERN.pattern + r")\]")
    return set(pattern.findall(Path(path).read_text(encoding="utf-8", errors="replace")))


def labels_from_csv(path: str | Path) -> pd.Series:
    """block_id,label[,...] -> Series of "Anomaly"/"Normal" indexed by block_id."""
    frame = pd.read_csv(path, dtype={"block_id": "string"})
    return frame.set_index("block_id")["label"]


# --- CloudTrail weak labels ------------------------------------------------
#
# NOT GROUND TRUTH. flAWS.cloud traffic is real but unlabeled; this rule marks
# an (identity, 1-hour) block as Anomaly when either holds:
#
#   R1 auth-error burst: at least AUTH_ERROR_BURST events in the block whose
#      errorCode matches AccessDenied / Unauthorized (AccessDenied,
#      Client.UnauthorizedOperation, UnauthorizedOperation,
#      AccessDeniedException, ...).
#   R2 new IP on a sensitive action: the block contains a sensitive IAM/S3
#      call (SENSITIVE_IAM_PREFIXES / SENSITIVE_S3_ACTIONS) from a source IP
#      whose first appearance anywhere in the trace falls in that same hour,
#      i.e. a never-before-seen address changing permissions or bucket policy.
#
# Both are judgement calls with fixed thresholds; they capture "looks like
# credential misuse" but will both miss real attacks and flag benign bursts.

AUTH_ERROR_PATTERN = re.compile(r"accessdenied|unauthorized", re.IGNORECASE)
AUTH_ERROR_BURST = 10
SENSITIVE_IAM_PREFIXES = ("Create", "Delete", "Attach", "Detach", "Put", "Update", "Add", "Remove", "Upload", "Set", "Change")
SENSITIVE_S3_ACTIONS = frozenset({
    "PutBucketPolicy", "DeleteBucketPolicy", "PutBucketAcl", "PutObjectAcl", "DeleteBucket",
    "PutBucketPublicAccessBlock", "DeletePublicAccessBlock", "PutBucketCors", "PutBucketWebsite",
    "DeleteBucketWebsite", "PutBucketLogging", "PutBucketVersioning", "PutEncryptionConfiguration",
})
WEAK_RULE_DESCRIPTION = (
    f"WEAK, rule-based labels (not ground truth). An (identity, 1-hour) block is Anomaly if R1: >= {AUTH_ERROR_BURST} "
    "events with an AccessDenied/Unauthorized errorCode, or R2: a sensitive IAM write (Create*/Delete*/Attach*/"
    "Detach*/Put*/Update*/Add*/Remove*/Upload*/Set*/Change*) or S3 permission/policy change from a source IP first "
    "seen in the trace during that same hour."
)


def is_sensitive(component: str, event_name: str) -> bool:
    if component == "iam.amazonaws.com":
        return event_name.startswith(SENSITIVE_IAM_PREFIXES)
    if component == "s3.amazonaws.com":
        return event_name in SENSITIVE_S3_ACTIONS
    return False


class CloudTrailWeakLabeler:
    """Fed cleaned CloudTrail rows in chronological order, chunk by chunk."""

    def __init__(self):
        self.auth_errors = defaultdict(int)
        self.sensitive_new_ip = defaultdict(int)
        self.first_seen_hour: dict[str, str] = {}
        self.blocks: set[str] = set()

    def update(self, chunk: pd.DataFrame):
        chunk = chunk.sort_values("datetime", kind="stable")
        block_ids = chunk["block_id"].astype(str)
        self.blocks.update(block_ids.unique())
        error_code = chunk["raw_message"].astype(str).str.extract(r"errorCode=(\S+)")[0].fillna("-")
        auth = error_code.str.contains(AUTH_ERROR_PATTERN)
        for block_id, count in block_ids[auth].value_counts().items():
            self.auth_errors[block_id] += int(count)

        hours = block_ids.str.rsplit("|", n=1).str[-1]
        for ip, hour in zip(chunk["source_ip"].astype(str), hours):
            self.first_seen_hour.setdefault(ip, hour)
        sensitive = [is_sensitive(c, e) for c, e in zip(chunk["component"].astype(str), chunk["event_type"].astype(str))]
        for block_id, ip, hour, flag in zip(block_ids, chunk["source_ip"].astype(str), hours, sensitive):
            if flag and self.first_seen_hour.get(ip) == hour:
                self.sensitive_new_ip[block_id] += 1

    def labels(self) -> pd.DataFrame:
        rows = []
        for block_id in sorted(self.blocks):
            r1 = self.auth_errors.get(block_id, 0) >= AUTH_ERROR_BURST
            r2 = self.sensitive_new_ip.get(block_id, 0) > 0
            rows.append({
                "block_id": block_id,
                "label": "Anomaly" if (r1 or r2) else "Normal",
                "rule_r1_auth_error_burst": r1,
                "rule_r2_new_ip_sensitive_action": r2,
                "auth_errors": self.auth_errors.get(block_id, 0),
            })
        return pd.DataFrame(rows)
