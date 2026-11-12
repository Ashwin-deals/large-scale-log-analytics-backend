"""Django request/error logs -> COMMON_LOG_SCHEMA.

The format is the one scripts/generate_django_logs.py writes (SYNTHETIC data;
there is no real Django log in this project yet):

    [2026-03-02 14:05:33,120] INFO django.request user=1234 ip=10.1.2.3 "GET /api/orders/551 HTTP/1.1" 200 1532 45ms
    [2026-03-02 14:05:34,002] ERROR django.request user=88 ip=10.1.9.4 "POST /api/checkout HTTP/1.1" 500 89 1203ms Internal Server Error: /api/checkout
    [2026-03-02 14:05:35,410] WARNING app.performance user=88 ip=10.1.9.4 "GET /api/reports/12 HTTP/1.1" 200 20431 5230ms Slow request

block_id = "user-<id>" (one user's activity). event_type = "<METHOD> <route
template> <status>" (ids in the path -> <n>), with " SLOW" appended for a
slow-request warning. block_size = response bytes. Milliseconds are dropped
(the pipeline works at 1-second resolution).
"""

import re

import pandas as pd

from .base_parser import BaseParser
from .templates import normalize_template

LINE_PATTERN = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2})(?:,\d+)?\]\s+(?P<log_level>[A-Z]+)\s+(?P<component>[\w.]+)\s+"
    r"user=(?P<user>\S+)\s+ip=(?P<ip>\S+)\s+\"(?P<method>[A-Z]+) (?P<path>\S+) [^\"]*\"\s+(?P<status>\d{3})\s+(?P<bytes>\d+)\s+(?P<ms>\d+)ms\s*(?P<detail>.*)$"
)
ID_SEGMENT = re.compile(r"/\d+(?=/|$)")


class DjangoParser(BaseParser):
    def parse(self, file_path: str) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            rows = [self.parse_line(line) for line in handle]
        return pd.DataFrame(rows, columns=self.schema) if rows else self.empty_frame()

    def parse_line(self, line: str) -> dict:
        raw = line.rstrip("\n\r")
        match = LINE_PATTERN.match(raw)
        if not match:
            return {**dict.fromkeys(self.schema), "event_type": "OTHER", "raw_message": raw}
        route = ID_SEGMENT.sub("/<n>", match.group("path").split("?")[0])
        event_type = f"{match.group('method')} {route} {match.group('status')}"
        if "slow request" in match.group("detail").lower():
            event_type += " SLOW"
        return {
            "date": match.group("date"),
            "time": match.group("time"),
            "pid": 0,
            "log_level": match.group("log_level"),
            "component": match.group("component"),
            "event_type": normalize_template(event_type) if len(event_type) > 120 else event_type,
            "block_id": f"user-{match.group('user')}",
            "source_ip": match.group("ip"),
            "destination_ip": None,
            "block_size": int(match.group("bytes")),
            "raw_message": raw[raw.index("]") + 2:],
        }
