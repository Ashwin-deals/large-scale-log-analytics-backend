"""Apache error log -> COMMON_LOG_SCHEMA.

Line format:
    [Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK
    [Fri Jun 10 03:37:39 2005] [error] [client 194.116.250.2] File does not exist: /var/www/html/sumthin

There is no block id, so a block is a fixed time window: every line whose
timestamp falls in the same `window_seconds` slot (default 60) shares
block_id "win-<YYYYmmddTHHMMSS of the window start>". The window size must
match between training and any upload scored against the trained model.

component: "client" for a "[client <ip>]" request line, else the leading
module token ("jk2_init()" -> "jk2_init", "mod_jk" -> "mod_jk", "LDAP:" ->
"LDAP"). event_type: the message (without the client prefix) normalized into a
template. Untimestamped continuation lines ("script not found or unable to
stat") have no date and are dropped by the cleaner.
"""

import re
from datetime import datetime, timedelta

import pandas as pd

from .base_parser import BaseParser
from .templates import normalize_template

DEFAULT_WINDOW_SECONDS = 60

LINE_PATTERN = re.compile(r"^\[(?P<stamp>[A-Z][a-z]{2} [A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2} \d{4})\]\s+\[(?P<level>[a-z]+)\]\s*(?P<message>.*)$")
CLIENT_PATTERN = re.compile(r"^\[client (?P<ip>[^\]\s]+)\]\s*")
MODULE_PATTERN = re.compile(r"^(?P<module>[A-Za-z_][\w.]*?)(?:\(\)|:|\s|$)")
EPOCH = datetime(1970, 1, 1)


def window_block_id(moment: datetime, window_seconds: int) -> str:
    seconds = int((moment - EPOCH).total_seconds())
    start = EPOCH + timedelta(seconds=seconds - seconds % window_seconds)
    return f"win-{start:%Y%m%dT%H%M%S}"


class ApacheParser(BaseParser):
    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = int(window_seconds)

    def parse(self, file_path: str) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            rows = [self.parse_line(line) for line in handle]
        return pd.DataFrame(rows, columns=self.schema) if rows else self.empty_frame()

    def parse_line(self, line: str) -> dict:
        raw = line.rstrip("\n\r")
        match = LINE_PATTERN.match(raw)
        if not match:
            return {**dict.fromkeys(self.schema), "event_type": "OTHER", "raw_message": raw}
        moment = datetime.strptime(" ".join(match.group("stamp").split()), "%a %b %d %H:%M:%S %Y")
        message = match.group("message")
        client = CLIENT_PATTERN.match(message)
        body = message[client.end():] if client else message
        module = MODULE_PATTERN.match(body)
        return {
            "date": moment.date().isoformat(),
            "time": moment.time().isoformat(),
            "pid": 0,  # the error log carries none; the cleaner requires a value
            "log_level": match.group("level").upper(),
            "component": "client" if client else (module.group("module") if module else "core"),
            "event_type": normalize_template(body),
            "block_id": window_block_id(moment, self.window_seconds),
            "source_ip": client.group("ip") if client else None,
            "destination_ip": None,
            "block_size": None,
            "raw_message": message,
        }
