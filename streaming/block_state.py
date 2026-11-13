"""Line -> event, and per-block running state.

Parity with the batch pipeline is the point of this module:

  - parse_event() keeps exactly the lines HDFSDataCleaner keeps (every
    required field present, parsable date/time) and normalizes them the same
    way (stripped strings, missing IPs -> "UNKNOWN", missing block_size -> 0).
  - BlockState.add() drops a line identical to one already seen for that
    block, which is what the cleaner's drop_duplicates() does (identical rows
    always share a block_id).
  - prefix_features() at the moment a block reaches k events equals
    early_warning.prefix_features for that k: the running state *is* the
    prefix then.
  - full_features() equals HDFSFeatureExtractor's row for the block.

tests/test_streaming.py checks both against the batch code.
"""

from __future__ import annotations

import calendar
import copy
from collections import Counter
from datetime import datetime

from early_warning.prefix_features import ERROR_LEVELS, WARN_LEVELS
from feature_engineering.category_encoder import DEFAULT_VOCAB_PATH, UNKNOWN, CategoryEncoder

REQUIRED_FIELDS = ("date", "time", "pid", "log_level", "component", "event_type", "block_id", "raw_message")
OPTIONAL_IP_FIELDS = ("source_ip", "destination_ip")


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != "<NA>" else None


def parse_event(parser, line: str) -> dict | None:
    """One raw line -> a cleaned event dict, or None where the cleaner would drop it."""
    row = parser.parse_line(line)
    event = {field: _text(row.get(field)) for field in REQUIRED_FIELDS}
    if any(value is None for value in event.values()):
        return None
    try:
        moment = datetime.fromisoformat(f"{event['date']} {event['time']}")
        event["pid"] = int(float(event["pid"]))
    except ValueError:
        return None
    event["seconds"] = calendar.timegm(moment.timetuple())  # naive log time, as the batch reads it
    event["hour"] = moment.hour
    for field in OPTIONAL_IP_FIELDS:
        event[field] = _text(row.get(field)) or UNKNOWN
    try:
        event["block_size"] = float(row.get("block_size")) if row.get("block_size") is not None else 0.0
    except (TypeError, ValueError):
        event["block_size"] = 0.0
    return event


class StreamEncoder:
    """Read-only view of the persisted CategoryEncoder vocabulary.

    A value the vocabulary has never seen gets the next free code in memory,
    the same append rule CategoryEncoder.encode() uses, but is never written
    back. The stream must not mutate the vocabulary the batch models were
    trained against. Unseen values are counted so status can report them.
    """

    def __init__(self, vocab_path=DEFAULT_VOCAB_PATH):
        self.vocab = copy.deepcopy(CategoryEncoder.load(vocab_path).vocab)
        self.unseen = Counter()

    def code(self, column: str, value: str | None) -> int:
        value = _text(value) or UNKNOWN
        mapping = self.vocab.setdefault(column, {})
        if value not in mapping:
            mapping[value] = max(mapping.values(), default=-1) + 1
            self.unseen[column] += 1
        return mapping[value]

    def event_type_names(self) -> list[str]:
        return [name for name, _ in sorted(self.vocab.get("event_type", {}).items(), key=lambda item: item[1])]


def _dominant(counts: Counter) -> int:
    # Most frequent code, ties toward the smallest: HDFSFeatureExtractor's rule.
    top = max(counts.values())
    return min(code for code, count in counts.items() if count == top)


class BlockState:
    __slots__ = (
        "block_id", "n", "t_min", "t_max", "hour_at_t_min", "warn", "error", "is_error", "block_size_max",
        "first_component", "first_event_type", "event_types", "components", "source_ips", "destination_ips",
        "seen_rows", "last_event_seconds", "early_alert_k", "dedupe",
    )

    def __init__(self, block_id: str, dedupe: bool = True):
        self.block_id = block_id
        self.dedupe = dedupe
        self.n = 0
        self.t_min = self.t_max = None
        self.hour_at_t_min = None
        self.warn = self.error = 0
        self.is_error = 0
        self.block_size_max = 0.0
        self.first_component = self.first_event_type = None
        self.event_types, self.components = Counter(), Counter()
        self.source_ips, self.destination_ips = Counter(), Counter()
        self.seen_rows: set = set()
        self.last_event_seconds = None
        self.early_alert_k = None

    def add(self, event: dict, encoder: StreamEncoder) -> bool:
        """Fold one event in. Returns False for an exact duplicate (dropped)."""
        row_key = (
            event["date"], event["time"], event["pid"], event["log_level"], event["component"], event["event_type"],
            event["source_ip"], event["destination_ip"], event["block_size"], event["raw_message"],
        )
        if self.dedupe:
            # Same rule as the batch cleaner for this source.
            if row_key in self.seen_rows:
                return False
            self.seen_rows.add(row_key)

        component = encoder.code("component", event["component"])
        event_type = encoder.code("event_type", event["event_type"])
        if self.n == 0:
            self.first_component, self.first_event_type = component, event_type
        self.n += 1
        self.event_types[event_type] += 1
        self.components[component] += 1
        self.source_ips[encoder.code("source_ip", event["source_ip"])] += 1
        self.destination_ips[encoder.code("destination_ip", event["destination_ip"])] += 1

        level = event["log_level"].upper()
        self.warn += level in WARN_LEVELS
        self.error += level in ERROR_LEVELS
        self.is_error = max(self.is_error, int(level == "ERROR"))  # the extractor's is_error rule
        self.block_size_max = max(self.block_size_max, event["block_size"])

        seconds = event["seconds"]
        if self.t_min is None or seconds < self.t_min:
            self.t_min, self.hour_at_t_min = seconds, event["hour"]
        self.t_max = seconds if self.t_max is None else max(self.t_max, seconds)
        self.last_event_seconds = seconds
        return True

    def prefix_features(self, event_type_names: list[str], encoder: StreamEncoder) -> dict:
        features = {
            "n_events": self.n,
            "elapsed_seconds": self.t_max - self.t_min,
            "distinct_event_types": len(self.event_types),
            "distinct_components": len(self.components),
            "warn_count": self.warn,
            "error_count": self.error,
            "block_size_max": self.block_size_max,
            "first_component_encoded": self.first_component,
            "first_event_type_encoded": self.first_event_type,
            "dominant_component_encoded": _dominant(self.components),
            "dominant_event_type_encoded": _dominant(self.event_types),
        }
        for name in event_type_names:
            features[f"count_{name}"] = self.event_types.get(encoder.vocab["event_type"][name], 0)
        return features

    def full_features(self) -> dict:
        return {
            "hour_of_day": self.hour_at_t_min,
            "is_error": self.is_error,
            "component_encoded": _dominant(self.components),
            "event_type_encoded": _dominant(self.event_types),
            "block_size": self.block_size_max,
            "event_frequency": self.n,
            "source_ip_encoded": _dominant(self.source_ips),
            "destination_ip_encoded": _dominant(self.destination_ips),
        }
