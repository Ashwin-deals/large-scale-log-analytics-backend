"""Syslog (RFC 5424 / RFC 3164) over UDP -> a source's Kafka topic.

Syslog is the transport, not a log format: each datagram's MSG part is one
log line of the chosen source (HDFS, OpenStack, Apache, Django ...), so a
host that ships its application log over syslog feeds the same stream
processor as a replay. The header (priority, timestamp, hostname, app name)
is stripped; the line is keyed by its block id like the producer does.

    python3 -m streaming.syslog_receiver --source hdfs --port 5514
    logger -n 127.0.0.1 -P 5514 -d "081109 203615 148 INFO dfs.DataNode\\$PacketResponder: ..."   # test

Then run the processor for that source (python3 -m streaming.processor --source hdfs).
"""

from __future__ import annotations

import argparse
import re
import signal
import socket
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streaming.config import BOOTSTRAP_SERVERS  # noqa: E402
from streaming.sources import STREAM_SOURCES, line_parser, message_key, topic_for  # noqa: E402

# <PRI>1 TIMESTAMP HOST APP PROCID MSGID [SD] MSG   (structured data is "-" or [...] blocks)
RFC5424 = re.compile(r"^<\d{1,3}>1 \S+ \S+ \S+ \S+ \S+ (?:-|(?:\[[^\]]*\])+) ?(?P<msg>.*)$", re.DOTALL)
# <PRI>Mmm dd hh:mm:ss HOST TAG[pid]: MSG
RFC3164 = re.compile(r"^<\d{1,3}>[A-Z][a-z]{2} [ \d]\d \d\d:\d\d:\d\d \S+ [^:\s]+:? ?(?P<msg>.*)$", re.DOTALL)
BOM = "\ufeff"


def extract_message(datagram: bytes) -> str | None:
    """The MSG part of one syslog datagram, or None when it isn't syslog."""
    text = datagram.decode("utf-8", errors="replace").rstrip("\r\n\x00")
    for pattern in (RFC5424, RFC3164):
        match = pattern.match(text)
        if match:
            message = match.group("msg").lstrip(BOM).strip("\r\n")
            return message or None
    return None


def main(argv=None):
    from confluent_kafka import Producer

    parser = argparse.ArgumentParser(description="Receive syslog over UDP and publish each message to a source's topic.")
    parser.add_argument("--source", default="hdfs", choices=STREAM_SOURCES, help="log format the syslog messages carry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5514, help="UDP port (514 needs root; default 5514)")
    parser.add_argument("--bootstrap", default=BOOTSTRAP_SERVERS)
    args = parser.parse_args(argv)

    topic = topic_for(args.source)
    keyer = line_parser(args.source)
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 20})
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)
    stopping = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))

    counts = {"received": 0, "published": 0, "not_syslog": 0}
    print(f"[syslog] udp://{args.host}:{args.port} ({args.source}) -> {topic}", flush=True)
    while not stopping["flag"]:
        try:
            datagram, _ = sock.recvfrom(65535)
        except socket.timeout:
            producer.poll(0)
            continue
        counts["received"] += 1
        message = extract_message(datagram)
        if message is None:
            counts["not_syslog"] += 1
            continue
        producer.produce(topic, key=message_key(keyer, message), value=message.encode("utf-8"))
        counts["published"] += 1
        producer.poll(0)
        if counts["received"] % 1000 == 0:
            print(f"[syslog] {counts}", flush=True)
    producer.flush(10)
    sock.close()
    print(f"[syslog] stopped. {counts}", flush=True)


if __name__ == "__main__":
    main()
