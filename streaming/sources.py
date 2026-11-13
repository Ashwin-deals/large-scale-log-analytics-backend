"""Per-source plumbing for the stream: topic, raw input, line parser, block key.

Every source streams through the same processor; what differs is

  topic      logs.raw for HDFS (unchanged), logs.raw.<source> for the others
  messages   one raw log line per message; CloudTrail, whose raw data is
             gzipped JSON, sends one JSON record per message
  parser     the source's batch parser, so a streamed event equals the batch
             row (CloudTrail: record_to_row on the decoded record)
  key        the block id the parser assigns, so a block's events stay on one
             partition, in order
  models     the source's deployed model, early-warning models, vocabulary and
             Smart Storage policy (source_context)
"""

from __future__ import annotations

import json
from pathlib import Path

from streaming.config import TOPIC_LOGS_RAW

STREAM_SOURCES = ("hdfs", "openstack", "apache", "cloudtrail", "django")
HDFS_LOG = Path("data/raw/hdfs/HDFS.log")


def topic_for(source: str) -> str:
    if source not in STREAM_SOURCES:
        raise ValueError(f"unknown source {source!r}; choose one of {', '.join(STREAM_SOURCES)}")
    return TOPIC_LOGS_RAW if source == "hdfs" else f"{TOPIC_LOGS_RAW}.{source}"


class _CloudTrailLineParser:
    """parse_line() over one JSON-encoded CloudTrail record."""

    def __init__(self):
        from parser.cloudtrail_parser import CloudTrailParser

        self._parser = CloudTrailParser()

    def parse_line(self, line: str) -> dict:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return {}
        return self._parser.record_to_row(record) if isinstance(record, dict) else {}


def line_parser(source: str):
    """An object with parse_line(str) -> row dict in the common schema."""
    if source == "hdfs":
        from parser.hdfs_parser import HDFSParser

        return HDFSParser()
    if source == "cloudtrail":
        return _CloudTrailLineParser()
    from multi_source.sources import get_source

    return get_source(source).parser_factory()


def input_files(source: str) -> list[Path]:
    if source == "hdfs":
        return [HDFS_LOG]
    from multi_source.sources import get_source

    return list(get_source(source).inputs())


def iter_messages(source: str, limit: int | None = None, log: Path | None = None):
    """Raw messages in replay order: lines, or JSON records for CloudTrail."""
    files = [log] if log is not None else input_files(source)
    missing = [str(f) for f in files if not Path(f).exists()]
    if not files or missing:
        raise FileNotFoundError(
            f"No raw {source} input found ({', '.join(missing) or 'no files matched'}). "
            f"Fetch it with: python3 scripts/download_dataset.py --source {source}"
        )
    sent = 0
    for path in files:
        if source == "cloudtrail":
            from parser.cloudtrail_parser import iter_records

            stream = (json.dumps(record, sort_keys=True) for record in iter_records(path))
        else:
            stream = _lines(path)
        for message in stream:
            if limit is not None and sent >= limit:
                return
            sent += 1
            yield message


def _lines(path: Path):
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n\r")


def message_key(parser, message: str) -> bytes | None:
    try:
        block_id = parser.parse_line(message).get("block_id")
    except Exception:  # noqa: BLE001 - an unparsable line is sent unkeyed; the processor drops it
        return None
    return str(block_id).encode() if block_id else None


# Log-time silence after which a block counts as finished, matched to each
# source's block definition. None = never by timeout, only at end of stream:
# a Django block is one user over the whole period, so any timeout would cut
# users into pieces (late events). Apache blocks are 60-second windows and
# CloudTrail blocks identity-hours, so they end soon after their window does.
IDLE_TIMEOUTS = {"hdfs": 3600, "openstack": 3600, "apache": 120, "cloudtrail": 3600, "django": None}


def best_k(source: str) -> int:
    """The early-warning prefix length the source's report picked as the best
    trade-off on still-running blocks (HDFS: 20)."""
    from source_context import get_context

    ctx = get_context(source)
    report_path = ctx.early_warning_report_path
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 20
    if not ctx.is_hdfs:
        report = report.get("early_warning") or {}
    trade = report.get("trade_off_in_progress") or report.get("trade_off") or {}
    return int(trade.get("best_k") or 20)
