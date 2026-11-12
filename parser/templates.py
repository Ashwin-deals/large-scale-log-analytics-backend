"""Message text -> event template, for log sources without a fixed event taxonomy.

Variable parts are replaced by placeholders so lines that differ only in an
id, address, path or number share one event_type:

    "File does not exist: /var/www/html/sumthin"        -> "File does not exist: <path>"
    "jk2_init() Found child 30770 in scoreboard slot 9" -> "jk2_init() Found child <n> in scoreboard slot <n>"
"""

import re

MAX_TEMPLATE_LENGTH = 120

_SUBSTITUTIONS = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"(?:(?<=\s)|^)(?:[a-zA-Z]+://)?[\w.-]*(?:/[^\s\"',;)\]]*)+"), "<path>"),
    (re.compile(r"\b(?:0x)?[0-9a-fA-F]*\d[0-9a-fA-F]*\b(?=[^a-zA-Z]|$)"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def normalize_template(message: str) -> str:
    text = message or ""
    for pattern, placeholder in _SUBSTITUTIONS:
        text = pattern.sub(placeholder, text)
    text = text.strip()
    return (text[:MAX_TEMPLATE_LENGTH] if text else "<empty>")
