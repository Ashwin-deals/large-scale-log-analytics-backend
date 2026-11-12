"""OpenStack (nova) logs -> COMMON_LOG_SCHEMA.

Line format:
    <source file> <date> <time.ms> <pid> <LEVEL> <component> [req-<id> <user> <tenant> - - -] <message>
    nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-...] 10.11.10.1 "GET ..." status: 200 len: 1893 time: 0.24

block_id is the VM instance UUID from "[instance: <uuid>]": a block is one
instance's session. A line with no instance id has no block (block_id None,
dropped by the cleaner); OpenStackParser(fallback_to_request_id=True) instead
uses the line's request id ("req-<uuid>") as its block. event_type is the message normalized into a template
(parser/templates.py). Time is truncated to whole seconds, the resolution
the rest of the pipeline (and HDFS) works at.
"""

import re

import pandas as pd

from .base_parser import BaseParser
from .templates import normalize_template

LINE_PATTERN = re.compile(
    r"^(?P<file>\S+)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})(?:\.\d+)?\s+"
    r"(?P<pid>\d+)\s+(?P<log_level>[A-Z]+)\s+(?P<component>\S+)\s+\[(?P<context>[^\]]*)\]\s?(?P<message>.*)$"
)
INSTANCE_PATTERN = re.compile(r"\[instance:\s*(?P<uuid>[0-9a-fA-F-]{36})\]")
REQUEST_PATTERN = re.compile(r"^(?P<req>req-[0-9a-fA-F-]{36})")
IP_PATTERN = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
LENGTH_PATTERN = re.compile(r"\blen:\s*(\d+)")


class OpenStackParser(BaseParser):
    def __init__(self, fallback_to_request_id: bool = False):
        self.fallback_to_request_id = fallback_to_request_id

    def parse(self, file_path: str) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            rows = [self.parse_line(line) for line in handle]
        return pd.DataFrame(rows, columns=self.schema) if rows else self.empty_frame()

    def parse_line(self, line: str) -> dict:
        raw = line.rstrip("\n\r")
        match = LINE_PATTERN.match(raw)
        if not match:
            return {**dict.fromkeys(self.schema), "event_type": "OTHER", "raw_message": raw}

        message = match.group("message")
        instance = INSTANCE_PATTERN.search(message)
        request = REQUEST_PATTERN.match(match.group("context").strip()) if self.fallback_to_request_id else None
        block_id = instance.group("uuid") if instance else (request.group("req") if request else None)
        template_source = INSTANCE_PATTERN.sub("", message)
        ip = IP_PATTERN.search(message)
        length = LENGTH_PATTERN.search(message)
        return {
            "date": match.group("date"),
            "time": match.group("time"),
            "pid": int(match.group("pid")),
            "log_level": match.group("log_level"),
            "component": match.group("component"),
            "event_type": normalize_template(template_source),
            "block_id": block_id,
            "source_ip": ip.group(1) if ip else None,
            "destination_ip": None,
            "block_size": int(length.group(1)) if length else None,
            "raw_message": message,
        }
