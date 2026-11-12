"""Replay HDFS log lines into logs.raw at a fixed rate.

Each message is one raw line, keyed by its block id (the first blk_... in
the line) so all of a block's lines land on one partition, in order. Lines
with no block id are sent unkeyed; the processor drops them, as the cleaner
does.

Usage:
    python3 -m streaming.producer --limit 50000 --rate 5000
    python3 -m streaming.producer --log data/raw/hdfs/HDFS.log --rate 0     # as fast as possible
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parser.hdfs_parser import BLOCK_ID_PATTERN  # noqa: E402
from streaming.config import BOOTSTRAP_SERVERS, TOPIC_LOGS_RAW  # noqa: E402

DEFAULT_LOG = Path("data/raw/hdfs/HDFS.log")


def block_key(line: str) -> bytes | None:
    match = BLOCK_ID_PATTERN.search(line)
    return match.group(0).encode() if match else None


class RateLimiter:
    """Paces sends to `rate` per second against a fixed start time, so
    sleep jitter does not accumulate. rate <= 0 means unthrottled."""

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.clock, self.sleep = rate, clock, sleep
        self.start = clock()

    def wait(self, sent: int):
        if self.rate <= 0:
            return
        ahead = sent / self.rate - (self.clock() - self.start)
        if ahead > 0:
            self.sleep(ahead)


def read_lines(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                return
            yield line.rstrip("\n\r")


def main(argv=None):
    from confluent_kafka import Producer

    parser = argparse.ArgumentParser(description="Replay HDFS log lines into Kafka.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--rate", type=float, default=1000.0, help="events per second (0 = unthrottled)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N lines")
    parser.add_argument("--bootstrap", default=BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=TOPIC_LOGS_RAW)
    args = parser.parse_args(argv)

    if not args.log.exists():
        sys.exit(f"{args.log} not found. Fetch it with: python3 scripts/download_dataset.py")

    failed = {"n": 0}

    def on_delivery(err, _msg):
        if err is not None:
            failed["n"] += 1

    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 20, "batch.num.messages": 10000})
    limiter = RateLimiter(args.rate)
    sent = 0
    print(f"[producer] {args.log} -> {args.topic} at {'max' if args.rate <= 0 else f'{args.rate:,.0f}/s'}"
          f"{f', limit {args.limit:,}' if args.limit else ''}", flush=True)
    for line in read_lines(args.log, args.limit):
        while True:
            try:
                producer.produce(args.topic, key=block_key(line), value=line.encode("utf-8"), on_delivery=on_delivery)
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.1)
        sent += 1
        producer.poll(0)
        if sent % 1000 == 0:
            limiter.wait(sent)
        if sent % 10000 == 0:
            print(f"[producer] {sent:,} sent ({sent / max(time.monotonic() - limiter.start, 1e-9):,.0f}/s)", flush=True)
    remaining = producer.flush(30)
    elapsed = time.monotonic() - limiter.start
    print(f"[producer] done: {sent:,} lines in {elapsed:.1f}s ({sent / max(elapsed, 1e-9):,.0f}/s), "
          f"{failed['n']} failed, {remaining} undelivered", flush=True)
    if failed["n"] or remaining:
        sys.exit(1)


if __name__ == "__main__":
    main()
