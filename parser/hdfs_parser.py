import re
from datetime import datetime

import pandas as pd
from .base_parser import BaseParser


LINE_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<log_level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s+"
    r"(?P<message>.*)$"
)

BLOCK_ID_PATTERN = re.compile(r"\bblk_-?\d+\b")
BLOCK_SIZE_PATTERN = re.compile(r"\b(?:of\s+)?size\s+(?P<size>\d+)\b", re.IGNORECASE)
IP_PATTERN = r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3})"

SOURCE_PATTERNS = [
    re.compile(rf"\bsrc:\s*/?{IP_PATTERN}(?::\d+)?"),
    re.compile(rf"\bfrom\s+/?{IP_PATTERN}(?::\d+)?"),
    re.compile(rf"^/?{IP_PATTERN}(?::\d+)?(?::)?"),
    re.compile(rf"\bask\s+/?{IP_PATTERN}(?::\d+)?\s+to\s+replicate\b"),
    re.compile(rf"\bblockMap updated:\s+/?{IP_PATTERN}(?::\d+)?\s+is added\b"),
]

DESTINATION_PATTERNS = [
    re.compile(rf"\bdest:\s*/?{IP_PATTERN}(?::\d+)?"),
    re.compile(rf"\bto\s+/?{IP_PATTERN}(?::\d+)?"),
    re.compile(rf"\bto datanode\(s\)\s+/?{IP_PATTERN}(?::\d+)?"),
]

EVENT_PATTERNS = [
    (re.compile(r"\bReceiving block\b"), "RECEIVE_BLOCK"),
    (re.compile(r"\bReceived block\b"), "RECEIVED_BLOCK"),
    (re.compile(r"\bPacketResponder\b.*\bterminating\b"), "PACKET_TERMINATED"),
    (re.compile(r"\bNameSystem\.allocateBlock\b"), "ALLOCATE_BLOCK"),
    (re.compile(r"\bNameSystem\.addStoredBlock\b"), "STORE_BLOCK"),
    (re.compile(r"\bServed block\b"), "SERVE_BLOCK"),
    (re.compile(r"\bStarting thread to transfer block\b"), "START_TRANSFER"),
    (re.compile(r"\bTransmitted block\b"), "TRANSMIT_BLOCK"),
    (re.compile(r"\breplicate\b", re.IGNORECASE), "REPLICATE_BLOCK"),
]


class HDFSParser(BaseParser):
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parses an HDFS log file and returns a pandas DataFrame.
        """
        rows = []

        with open(file_path, "r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                raw_line = line.rstrip("\n\r")
                rows.append(self._parse_line(raw_line))

        if not rows:
            return self.empty_frame()

        return pd.DataFrame(rows, columns=self.schema)

    def classify_event(self, message: str) -> str:
        """
        Maps a raw HDFS log message to the common event taxonomy.
        """
        for pattern, event_type in EVENT_PATTERNS:
            if pattern.search(message):
                return event_type
        return "OTHER"

    def _parse_line(self, raw_line: str) -> dict:
        match = LINE_PATTERN.match(raw_line)

        if not match:
            return {
                "date": None,
                "time": None,
                "pid": None,
                "log_level": None,
                "component": None,
                "event_type": "OTHER",
                "block_id": self._extract_block_id(raw_line),
                "source_ip": self._extract_first_ip(raw_line, SOURCE_PATTERNS),
                "destination_ip": self._extract_first_ip(raw_line, DESTINATION_PATTERNS),
                "block_size": self._extract_block_size(raw_line),
                "raw_message": raw_line,
            }

        message = match.group("message")
        return {
            "date": self._normalize_date(match.group("date")),
            "time": self._normalize_time(match.group("time")),
            "pid": int(match.group("pid")),
            "log_level": match.group("log_level"),
            "component": match.group("component"),
            "event_type": self.classify_event(message),
            "block_id": self._extract_block_id(message),
            "source_ip": self._extract_first_ip(message, SOURCE_PATTERNS),
            "destination_ip": self._extract_first_ip(message, DESTINATION_PATTERNS),
            "block_size": self._extract_block_size(message),
            "raw_message": message,
        }

    @staticmethod
    def _normalize_date(value: str) -> str:
        return datetime.strptime(value, "%y%m%d").date().isoformat()

    @staticmethod
    def _normalize_time(value: str) -> str:
        return datetime.strptime(value, "%H%M%S").time().isoformat()

    @staticmethod
    def _extract_block_id(message: str) -> str | None:
        match = BLOCK_ID_PATTERN.search(message)
        return match.group(0) if match else None

    @staticmethod
    def _extract_block_size(message: str) -> int | None:
        match = BLOCK_SIZE_PATTERN.search(message)
        return int(match.group("size")) if match else None

    @staticmethod
    def _extract_first_ip(message: str, patterns: list[re.Pattern]) -> str | None:
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                return match.group("ip")
        return None
