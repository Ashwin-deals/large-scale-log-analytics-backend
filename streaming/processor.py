"""Stream processor: logs.raw -> per-block state -> alerts + storage decisions.

For every line on logs.raw:
  1. parse it with parser/hdfs_parser.py and keep or drop it by the cleaner's
     rules (streaming.block_state.parse_event)
  2. fold it into that block's running state
  3. when a block reaches one of the configured k's (default 20, the
     early-warning report's best trade-off), score its first-k features with
     that k's early-warning model; at or above the GA-tuned threshold, raise
     an `early_warning` alert (at most once per block)
  4. when a block goes quiet for --idle-timeout seconds of *log* time (the
     newest event time seen is the clock, so a fast replay times blocks out
     the same way real time would), finalize it: deployed-model score (an
     Anomaly verdict raises a `block_anomaly` alert) and Smart Storage tier

Alerts go to the `alerts` topic and the MongoDB collection stream_alerts;
storage decisions go to `storage.decisions`. A status document (events/sec,
blocks in flight, alert counts, storage saved so far) is upserted into
stream_status every couple of seconds for GET /api/stream/status.

HDFS has no explicit "block finished" event, so completion means "quiet for
the timeout", and when the topic itself goes quiet for --idle-flush seconds of
wall time every block still in flight is finalized with reason
"end_of_stream". A line for a block that was already finalized is counted
as late and ignored, because the decision has been published. State is in
memory, so a restarted processor starts its blocks fresh.

Usage:
    python3 -m streaming.processor
    python3 -m streaming.processor --k 5 10 20 --idle-timeout 3600 --stop-after-idle 30
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parser.hdfs_parser import HDFSParser  # noqa: E402
from streaming.block_state import BlockState, StreamEncoder, parse_event  # noqa: E402
from streaming.config import (  # noqa: E402
    BOOTSTRAP_SERVERS,
    CONSUMER_GROUP,
    TOPIC_ALERTS,
    TOPIC_LOGS_RAW,
    TOPIC_STORAGE_DECISIONS,
)

DEFAULT_KS = (20,)
DEFAULT_IDLE_TIMEOUT_SECONDS = 3600  # log time
DEFAULT_IDLE_FLUSH_SECONDS = 10.0  # wall time
RATE_WINDOW_SECONDS = 5.0
STATUS_EVERY_SECONDS = 2.0


def _iso(seconds: int | None) -> str | None:
    return None if seconds is None else datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class StreamProcessor:
    """Kafka-free core: feed it lines, get back alerts and storage decisions."""

    def __init__(self, scorer, ks=DEFAULT_KS, idle_timeout_seconds=DEFAULT_IDLE_TIMEOUT_SECONDS,
                 run_id: str | None = None, encoder: StreamEncoder | None = None, clock=time.monotonic,
                 source: str = "hdfs", parser=None):
        self.scorer = scorer
        self.ks = sorted(set(int(k) for k in ks))
        missing = [k for k in self.ks if k not in scorer.early]
        if missing:
            raise ValueError(f"no early-warning model loaded for k = {missing}")
        self.idle_timeout_seconds = idle_timeout_seconds
        self.run_id = run_id or _now_utc().strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        self.encoder = encoder or StreamEncoder()
        self.event_type_names = self.encoder.event_type_names()
        self.source = source
        self.parser = parser or HDFSParser()
        if source == "hdfs":
            self.dedupe = True
        else:
            from multi_source.sources import get_source

            self.dedupe = get_source(source).dedupe
        self.clock = clock

        self.blocks: dict[str, BlockState] = {}
        self.finalized: set[str] = set()
        self.watermark: int | None = None  # newest log time seen
        self.started_at = _now_utc()
        self._rate_window: deque = deque()
        self.counts = {
            "lines_received": 0, "events": 0, "lines_dropped_unparsed": 0, "duplicate_lines": 0, "late_events": 0,
            "early_warning_alerts": 0, "block_anomaly_alerts": 0, "blocks_finalized": 0,
            "lines_total": 0, "lines_kept": 0,
        }
        self.tier_counts = {"HOT": 0, "WARM": 0, "COLD": 0}
        self.finalize_reasons = {"timeout": 0, "end_of_stream": 0}

    # -- ingestion ---------------------------------------------------------

    def process(self, lines) -> tuple[list[dict], list[dict]]:
        """Returns (alerts, storage_decisions) produced by this batch of lines."""
        # Prefix features are snapshotted at the exact event where a block hits
        # k -- later lines in the same batch must not leak into them -- and
        # scored together after the batch.
        reached: dict[int, list[tuple[BlockState, dict, int]]] = {k: [] for k in self.ks}
        n_lines = 0
        for line in lines:
            n_lines += 1
            event = parse_event(self.parser, line)
            if event is None:
                self.counts["lines_dropped_unparsed"] += 1
                continue
            block_id = event["block_id"]
            if block_id in self.finalized:
                self.counts["late_events"] += 1
                continue
            state = self.blocks.get(block_id)
            if state is None:
                state = self.blocks[block_id] = BlockState(block_id, dedupe=self.dedupe)
            if not state.add(event, self.encoder):
                self.counts["duplicate_lines"] += 1
                continue
            self.counts["events"] += 1
            self.watermark = event["seconds"] if self.watermark is None else max(self.watermark, event["seconds"])
            if state.n in reached and state.early_alert_k is None:
                features = state.prefix_features(self.event_type_names, self.encoder)
                reached[state.n].append((state, features, event["seconds"]))

        self.counts["lines_received"] += n_lines
        self._rate_window.append((self.clock(), n_lines))

        alerts = self._early_warnings(reached)
        more_alerts, decisions = self.expire()
        return alerts + more_alerts, decisions

    def _early_warnings(self, reached: dict[int, list[tuple[BlockState, dict, int]]]) -> list[dict]:
        alerts = []
        for k, snapshots in reached.items():  # ascending k
            snapshots = [snap for snap in snapshots if snap[0].early_alert_k is None]  # once per block, first k that fires
            if not snapshots:
                continue
            ew = self.scorer.early[k]
            scores = ew.scores([features for _, features, _ in snapshots])
            for (state, _, seconds), score in zip(snapshots, scores):
                if score >= ew.threshold:
                    state.early_alert_k = k
                    self.counts["early_warning_alerts"] += 1
                    alerts.append(self._alert("early_warning", state, float(score), ew.threshold,
                                              events_seen=k, event_time=_iso(seconds), k=k))
        return alerts

    def _alert(self, kind: str, state: BlockState, score: float, threshold: float | None, **extra) -> dict:
        # extra may override events_seen/event_time (an early warning reports
        # the k-th event, not where the block has got to since).
        return {
            "_id": f"{self.run_id}:{kind}:{state.block_id}",
            "run_id": self.run_id,
            "source": self.source,
            "type": kind,
            "block_id": state.block_id,
            "score": score,
            "threshold": threshold,
            "events_seen": state.n,
            "event_time": _iso(state.last_event_seconds),
            "detected_at": _now_utc().isoformat(),
            **extra,
        }

    # -- completion --------------------------------------------------------

    def expire(self) -> tuple[list[dict], list[dict]]:
        """Finalize blocks quiet for longer than the timeout in log time
        (never, when idle_timeout_seconds is None: end of stream only)."""
        if self.watermark is None or self.idle_timeout_seconds is None:
            return [], []
        cutoff = self.watermark - self.idle_timeout_seconds
        quiet = [s for s in self.blocks.values() if s.last_event_seconds < cutoff]
        return self._finalize(quiet, "timeout")

    def flush_all(self, reason: str = "end_of_stream") -> tuple[list[dict], list[dict]]:
        return self._finalize(list(self.blocks.values()), reason)

    def _finalize(self, states: list[BlockState], reason: str) -> tuple[list[dict], list[dict]]:
        if not states:
            return [], []
        verdict = self.scorer.full([s.full_features() for s in states])
        alerts, decisions = [], []
        for i, state in enumerate(states):
            label, score = str(verdict["predicted_label"][i]), float(verdict["anomaly_score"][i])
            tier, kept = verdict["tier"][i], int(verdict["lines_kept"][i])
            if label == "Anomaly":
                self.counts["block_anomaly_alerts"] += 1
                # finalize_reason "end_of_stream" means the block was forced
                # out when the topic went quiet, possibly before it finished:
                # a short replay cuts most blocks off mid-life, and a cut-off
                # block looks anomalous to a whole-block model.
                alerts.append(self._alert("block_anomaly", state, score, None, early_warning_k=state.early_alert_k,
                                          finalize_reason=reason))
            decisions.append({
                "run_id": self.run_id,
                "source": self.source,
                "block_id": state.block_id,
                "reason": reason,
                "events": state.n,
                "first_event_time": _iso(state.t_min),
                "last_event_time": _iso(state.t_max),
                "predicted_label": label,
                "anomaly_score": score,
                "importance": float(verdict["importance"][i]),
                "tier": tier,
                "lines_kept": kept,
                "early_warning_k": state.early_alert_k,
                # share of the block still to come when the early warning fired
                "early_warning_lead": None if state.early_alert_k is None else (state.n - state.early_alert_k) / state.n,
                "decided_at": _now_utc().isoformat(),
            })
            self.counts["blocks_finalized"] += 1
            self.counts["lines_total"] += state.n
            self.counts["lines_kept"] += kept
            self.tier_counts[tier] += 1
            self.finalize_reasons[reason] = self.finalize_reasons.get(reason, 0) + 1
            self.finalized.add(state.block_id)
            del self.blocks[state.block_id]
        return alerts, decisions

    # -- status ------------------------------------------------------------

    def events_per_second(self) -> float:
        now = self.clock()
        while self._rate_window and self._rate_window[0][0] < now - RATE_WINDOW_SECONDS:
            self._rate_window.popleft()
        if not self._rate_window:
            return 0.0
        span = max(now - self._rate_window[0][0], 1.0)
        return sum(n for _, n in self._rate_window) / span

    def status(self) -> dict:
        lines_total = self.counts["lines_total"]
        return {
            "_id": self.run_id,
            "run_id": self.run_id,
            "source": self.source,
            "started_at": self.started_at.isoformat(),
            "updated_at": _now_utc().isoformat(),
            "events_per_sec": round(self.events_per_second(), 1),
            "blocks_in_flight": len(self.blocks),
            "alerts": {
                "early_warning": self.counts["early_warning_alerts"],
                "block_anomaly": self.counts["block_anomaly_alerts"],
                "total": self.counts["early_warning_alerts"] + self.counts["block_anomaly_alerts"],
            },
            "storage": {
                "blocks_decided": self.counts["blocks_finalized"],
                "lines_seen": lines_total,
                "lines_kept": self.counts["lines_kept"],
                "storage_saved_pct": 100.0 * (1 - self.counts["lines_kept"] / lines_total) if lines_total else 0.0,
                "tier_counts": dict(self.tier_counts),
            },
            "counts": dict(self.counts),
            "finalize_reasons": dict(self.finalize_reasons),
            "log_time_watermark": _iso(self.watermark),
            "ks": self.ks,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "unseen_vocabulary_values": dict(self.encoder.unseen),
            "deployed_model_version": getattr(self.scorer, "deployed_version", None),
        }


class MongoSink:
    """stream_alerts (one document per alert) and stream_status (one per run)."""

    def __init__(self, alerts_collection, status_collection):
        from pymongo import DESCENDING

        self.alerts, self.status_collection = alerts_collection, status_collection
        self.alerts.create_index([("run_id", 1), ("detected_at", DESCENDING)])
        self.status_collection.create_index([("updated_at", DESCENDING)])

    def save_alerts(self, alerts: list[dict]):
        from pymongo.errors import BulkWriteError

        if not alerts:
            return
        try:
            self.alerts.insert_many([dict(a) for a in alerts], ordered=False)
        except BulkWriteError as exc:
            # _id is run:type:block, so a duplicate is the same alert written twice.
            if any(err.get("code") != 11000 for err in exc.details.get("writeErrors", [])):
                raise

    def record_leads(self, decisions: list[dict]):
        """An early warning's lead time is only known once its block is
        finalized: write it onto that alert document when the decision lands."""
        from pymongo import UpdateOne

        updates = [
            UpdateOne({"_id": f"{d['run_id']}:early_warning:{d['block_id']}"},
                      {"$set": {"lead_fraction": d["early_warning_lead"], "events_total": d["events"],
                                "block_finalized": d["reason"]}})
            for d in decisions if d.get("early_warning_k") is not None
        ]
        if updates:
            self.alerts.bulk_write(updates, ordered=False)

    def save_status(self, status: dict):
        self.status_collection.replace_one({"_id": status["_id"]}, status, upsert=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Consume logs.raw, raise early warnings, publish storage decisions.")
    parser.add_argument("--bootstrap", default=BOOTSTRAP_SERVERS)
    parser.add_argument("--group", default=CONSUMER_GROUP)
    parser.add_argument("--source", default="hdfs", choices=("hdfs", "openstack", "apache", "cloudtrail", "django"),
                        help="log source to consume (its own topic, parser, models and policy)")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="prefix lengths to alert at (default: the source's best trade-off k; models must exist)")
    parser.add_argument("--idle-timeout", type=int, default=None,
                        help="finalize a block after this many seconds of log time without an event "
                             "(default per source: hdfs/openstack/cloudtrail 3600, apache 120, django never; 0 = never)")
    parser.add_argument("--idle-flush", type=float, default=DEFAULT_IDLE_FLUSH_SECONDS,
                        help="when the topic is quiet this many wall seconds, finalize every block in flight")
    parser.add_argument("--stop-after-idle", type=float, default=None,
                        help="exit after the topic has been quiet this many wall seconds (after flushing)")
    parser.add_argument("--from-beginning", action="store_true", help="new group id, read logs.raw from offset 0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None):
    from confluent_kafka import Consumer, KafkaException, Producer

    from db import stream_alerts, stream_status
    from source_context import get_context
    from streaming.scoring import scorer_for
    from streaming.sources import best_k, line_parser, topic_for

    args = parse_args(argv)
    ks = args.k or [best_k(args.source)]
    from streaming.sources import IDLE_TIMEOUTS

    idle_timeout = IDLE_TIMEOUTS[args.source] if args.idle_timeout is None else (args.idle_timeout or None)
    ctx = get_context(args.source)
    encoder = StreamEncoder() if ctx.is_hdfs else StreamEncoder(ctx.vocab_path)
    processor = StreamProcessor(scorer_for(args.source, ks), ks=ks, idle_timeout_seconds=idle_timeout,
                                run_id=args.run_id, encoder=encoder, source=args.source, parser=line_parser(args.source))
    topic = topic_for(args.source)
    sink = MongoSink(stream_alerts, stream_status)
    group = f"{args.group}-{processor.run_id}" if args.from_beginning else args.group
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 50})
    consumer.subscribe([topic])

    stopping = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))

    def publish(alerts, decisions):
        for alert in alerts:
            producer.produce(TOPIC_ALERTS, key=alert["block_id"], value=json.dumps(alert))
        for decision in decisions:
            producer.produce(TOPIC_STORAGE_DECISIONS, key=decision["block_id"], value=json.dumps(decision))
        producer.poll(0)
        sink.save_alerts(alerts)
        sink.record_leads(decisions)

    print(f"[processor] run {processor.run_id} ({args.source}): consuming {topic} from {args.bootstrap} (group {group}), "
          f"k = {processor.ks}, idle timeout {f'{idle_timeout}s log time' if idle_timeout else 'none (end of stream)'}", flush=True)
    last_message = last_status = time.monotonic()
    flushed_since_last_message = True
    try:
        while not stopping["flag"]:
            messages = consumer.consume(num_messages=args.batch_size, timeout=1.0)
            lines = []
            for message in messages:
                if message.error():
                    raise KafkaException(message.error())
                lines.append(message.value().decode("utf-8", errors="replace"))
            now = time.monotonic()
            if lines:
                last_message, flushed_since_last_message = now, False
                publish(*processor.process(lines))
            elif not flushed_since_last_message and now - last_message >= args.idle_flush:
                alerts, decisions = processor.flush_all()
                publish(alerts, decisions)
                flushed_since_last_message = True
                print(f"[processor] topic quiet {args.idle_flush:.0f}s: finalized {len(decisions):,} blocks in flight", flush=True)
            if now - last_status >= STATUS_EVERY_SECONDS:
                status = processor.status()
                sink.save_status(status)
                last_status = now
                print(f"[processor] {status['counts']['lines_received']:,} lines, {status['events_per_sec']:,.0f}/s, "
                      f"{status['blocks_in_flight']:,} in flight, alerts {status['alerts']}, "
                      f"saved {status['storage']['storage_saved_pct']:.1f}%", flush=True)
            received_any = processor.counts["lines_received"] > 0  # a processor started before the producer waits
            if (args.stop_after_idle is not None and received_any and flushed_since_last_message
                    and now - last_message >= args.stop_after_idle):
                break
    finally:
        alerts, decisions = processor.flush_all("end_of_stream") if processor.blocks else ([], [])
        publish(alerts, decisions)
        producer.flush(10)
        consumer.close()
        status = processor.status()
        status["stopped_at"] = _now_utc().isoformat()
        sink.save_status(status)
        print(f"[processor] stopped. {json.dumps({k: status[k] for k in ('alerts', 'storage', 'counts')})}", flush=True)


if __name__ == "__main__":
    main()
