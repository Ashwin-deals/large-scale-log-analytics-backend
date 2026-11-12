"""Write an INJECTED copy of the Apache error log, with labeled anomalous windows.

The real Apache.log has no labels. For evaluation this picks a fixed-seed
sample of the log's existing time windows and injects one of:

    error_burst      30-120 extra "[error] [client <ip>] ..." lines, reusing the
                     log's own commonest error templates with fresh paths/IPs
    unseen_template  5-20 lines of error/crit messages that never occur in the log

Injected lines get timestamps inside their window and are merged in time
order. The original data/raw/apache/Apache.log is never modified. Windows not
chosen are labeled Normal, though the real log may contain genuine,
unlabeled anomalies. Those would count as false positives, so the metrics
are a lower bound on detector quality for those windows.

Writes (both marked INJECTED in their names):
    data/raw/apache/injected/Apache_INJECTED.log
    data/raw/apache/injected/Apache_INJECTED_labels.csv   block_id,label,injection,window_seconds

Usage:
    python3 scripts/inject_apache_anomalies.py
    python3 scripts/inject_apache_anomalies.py --rate 0.03 --window-seconds 60 --seed 42
"""

import argparse
import csv
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from parser.apache_parser import DEFAULT_WINDOW_SECONDS, LINE_PATTERN, window_block_id  # noqa: E402

SOURCE_LOG = Path("data/raw/apache/Apache.log")
OUT_DIR = Path("data/raw/apache/injected")
LOG_PATH = OUT_DIR / "Apache_INJECTED.log"
LABELS_PATH = OUT_DIR / "Apache_INJECTED_labels.csv"
STAMP_FORMAT = "%a %b %d %H:%M:%S %Y"

BURST_TEMPLATES = [
    "File does not exist: /var/www/html/{word}/{n}.php",
    "Directory index forbidden by rule: /var/www/html/{word}/",
    "client denied by server configuration: /var/www/html/{word}",
]
UNSEEN_MESSAGES = [
    ("crit", "(28)No space left on device: mod_rewrite: could not create rewrite_log_lock"),
    ("error", "mod_security: Access denied with code 403 (phase 2). Pattern match \"union select\" at ARGS:id [id \"{n}\"]"),
    ("emerg", "(13)Permission denied: make_sock: could not bind to address 0.0.0.0:{n}"),
    ("error", "server reached MaxClients setting, consider raising the MaxClients setting"),
    ("crit", "[client {ip}] configuration error: couldn't perform authentication. AuthType not set!: /admin/{word}"),
]
WORDS = ["admin", "phpmyadmin", "wp-login", "cgi-bin", "backup", "shell", "config", "xmlrpc"]


def random_ip(rng):
    return f"{rng.randint(11, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def stamp_line(moment, level, message):
    return f"[{moment.strftime(STAMP_FORMAT)}] [{level}] {message}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rate", type=float, default=0.03, help="share of windows to inject into")
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    original = SOURCE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    timed, windows = [], {}
    for index, text in enumerate(original):
        match = LINE_PATTERN.match(text)
        moment = datetime.strptime(" ".join(match.group("stamp").split()), STAMP_FORMAT) if match else None
        timed.append((moment, index, text))
        if moment is not None:
            windows.setdefault(window_block_id(moment, args.window_seconds), moment.replace(second=0) - timedelta(
                seconds=(moment.minute * 60) % args.window_seconds if args.window_seconds > 60 else 0))

    window_ids = sorted(windows)
    chosen = sorted(rng.sample(window_ids, max(1, round(len(window_ids) * args.rate))))
    kinds = {w: rng.choice(["error_burst", "unseen_template"]) for w in chosen}

    injected = []
    for window in chosen:
        start = datetime.strptime(window[len("win-"):], "%Y%m%dT%H%M%S")
        at = lambda: start + timedelta(seconds=rng.randrange(args.window_seconds))
        if kinds[window] == "error_burst":
            for _ in range(rng.randint(30, 120)):
                message = rng.choice(BURST_TEMPLATES).format(word=rng.choice(WORDS), n=rng.randint(1, 9999))
                injected.append((at(), stamp_line(at(), "error", f"[client {random_ip(rng)}] {message}")))
        else:
            for _ in range(rng.randint(5, 20)):
                level, message = rng.choice(UNSEEN_MESSAGES)
                injected.append((at(), stamp_line(at(), level, message.format(n=rng.randint(1, 65535), ip=random_ip(rng), word=rng.choice(WORDS)))))

    # Merge in time order. A continuation line (no timestamp) stays right after
    # the line it continues: it inherits the preceding line's time.
    merged, last_time = [], datetime.min
    for moment, index, text in timed:
        last_time = moment or last_time
        merged.append((last_time, 0, index, text))
    merged += [(moment, 1, i, text) for i, (moment, text) in enumerate(injected)]
    merged.sort(key=lambda item: (item[0], item[1], item[2]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(text for *_, text in merged) + "\n", encoding="utf-8")
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_id", "label", "injection", "window_seconds"])
        for window in window_ids:
            writer.writerow([window, "Anomaly" if window in kinds else "Normal", kinds.get(window, ""), args.window_seconds])
    print(f"INJECTED Apache log: {len(original):,} original + {len(injected):,} injected lines; "
          f"{len(chosen)} of {len(window_ids):,} windows injected ({Counter(kinds.values())}), seed {args.seed}")
    print(f"Wrote {LOG_PATH}\nWrote {LABELS_PATH}")


if __name__ == "__main__":
    main()
