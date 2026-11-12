"""AWS CloudTrail logs ({"Records": [...]} JSON, optionally .gz) -> COMMON_LOG_SCHEMA.

Records are streamed one at a time with json.JSONDecoder.raw_decode over a
rolling text buffer, so a 100k-record file is never materialized whole
(iter_records). parse() exists for small files such as uploads.

Field mapping:
    component   = eventSource            event_type = eventName
    source_ip   = sourceIPAddress        log_level  = ERROR if errorCode else INFO
    block_id    = "<identity>|<YYYYmmddTHH>"  -- one identity's activity in a 1-hour window

identity is the caller's ARN, else its access key / principal id, else
"<type>:<account>"; anonymous callers are split per source IP
("anonymous@<ip>"), or every anonymous request in an hour would be one block.
raw_message keeps eventID and errorCode for the weak-label rule
(multi_source/labels.py).
"""

import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .base_parser import BaseParser

READ_CHUNK = 1 << 20
WINDOW_FORMAT = "%Y%m%dT%H"


def _open_text(path: Path):
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gzipped else open(path, "r", encoding="utf-8")


def iter_records(path):
    """Yield each record of a CloudTrail file without loading the whole array."""
    decoder = json.JSONDecoder()
    with _open_text(Path(path)) as handle:
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            chunk = handle.read(READ_CHUNK)
            eof = not chunk
            buffer += chunk

        while '"Records"' not in buffer and not eof:
            fill()
        start = buffer.find('"Records"')
        if start < 0:
            return
        while "[" not in buffer[start:] and not eof:
            fill()
        position = buffer.index("[", start) + 1
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position >= len(buffer):
                if eof:
                    return
                buffer, position = buffer[position:], 0
                fill()
                continue
            if buffer[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer, position = buffer[position:], 0
                fill()
                continue
            yield record
            position = end
            if position > READ_CHUNK:  # drop what has been consumed
                buffer, position = buffer[position:], 0


def identity_of(record: dict) -> str:
    who = record.get("userIdentity") or {}
    kind = who.get("type") or "Unknown"
    if kind in ("AWSAccount", "Anonymous") and not who.get("arn") and not who.get("principalId"):
        return f"anonymous@{record.get('sourceIPAddress') or 'unknown'}"
    return who.get("arn") or who.get("accessKeyId") or who.get("principalId") or f"{kind}:{who.get('accountId') or 'unknown'}"


class CloudTrailParser(BaseParser):
    def parse(self, file_path: str) -> pd.DataFrame:
        rows = list(self.iter_rows(file_path))
        return pd.DataFrame(rows, columns=self.schema) if rows else self.empty_frame()

    def iter_rows(self, file_path):
        for record in iter_records(file_path):
            yield self.record_to_row(record)

    def record_to_row(self, record: dict) -> dict:
        try:
            moment = datetime.strptime(record.get("eventTime", ""), "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return {**dict.fromkeys(self.schema), "event_type": "OTHER", "raw_message": json.dumps(record)[:200]}
        error_code = record.get("errorCode")
        return {
            "date": moment.date().isoformat(),
            "time": moment.time().isoformat(),
            "pid": 0,  # CloudTrail has no process id; the cleaner requires a value
            "log_level": "ERROR" if error_code else "INFO",
            "component": record.get("eventSource"),
            "event_type": record.get("eventName"),
            "block_id": f"{identity_of(record)}|{moment:{WINDOW_FORMAT}}",
            "source_ip": record.get("sourceIPAddress"),
            "destination_ip": None,
            "block_size": None,
            "raw_message": f"eventID={record.get('eventID')} errorCode={error_code or '-'}",
        }
