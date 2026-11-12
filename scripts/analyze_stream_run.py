"""Score one streaming replay against the ground-truth labels.

A replay of the first N lines of HDFS.log cuts many blocks off mid-life, and a
cut-off block looks anomalous to a whole-block model. So results are split by
whether each block was actually complete when it was decided:

  complete   every line the full HDFS.log has for the block was inside the
             replayed prefix (checked by counting the block's lines in the
             whole file, not by trusting the processor's timeout)
  cut off    the block has more lines after the replayed prefix

and, for reference, by the processor's own finalize reason (timeout vs the
end-of-stream flush). Precision and lead time are reported on complete blocks.

Reads the run's messages back from the Kafka topics storage.decisions and
alerts (a throwaway consumer group, so nothing else is disturbed).

Writes data/evaluation/stream_run_report.json.

Usage:
    python3 scripts/analyze_stream_run.py --run-id <id> --replayed-lines 1000000
"""

import argparse
import json
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from streaming.config import BOOTSTRAP_SERVERS, TOPIC_ALERTS, TOPIC_STORAGE_DECISIONS

RAW_LOG = Path("data/raw/hdfs/HDFS.log")
LABELS = Path("data/raw/hdfs/anomaly_label.csv")
OUTPUT = Path("data/evaluation/stream_run_report.json")
BLOCK_ID = re.compile(r"\bblk_-?\d+\b")


def read_topic(topic: str, run_id: str, bootstrap: str, quiet_seconds: float = 6.0) -> list[dict]:
    from confluent_kafka import Consumer

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"stream-analysis-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])
    rows, last = [], time.monotonic()
    while time.monotonic() - last < quiet_seconds:
        for msg in consumer.consume(num_messages=5000, timeout=1.0):
            if msg.error():
                continue
            last = time.monotonic()
            value = json.loads(msg.value())
            if value.get("run_id") == run_id:
                rows.append(value)
    consumer.close()
    return rows


def count_lines(blocks: set, replayed: int) -> tuple[Counter, Counter]:
    """Lines per block inside the replayed prefix and in the whole file.
    The producer keys by the first blk_ id in a line; so does this."""
    prefix, total = Counter(), Counter()
    with RAW_LOG.open("r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            match = BLOCK_ID.search(line)
            if match is None:
                continue
            block = match.group(0)
            if block in blocks:
                total[block] += 1
                if i < replayed:
                    prefix[block] += 1
    return prefix, total


def rate(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def detection(frame: pd.DataFrame, flagged: pd.Series) -> dict:
    anomalous = frame["is_anomaly"]
    tp = int((flagged & anomalous).sum())
    fp = int((flagged & ~anomalous).sum())
    return {
        "blocks": int(len(frame)),
        "anomalies": int(anomalous.sum()),
        "alerts": int(flagged.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "precision": rate(tp, int(flagged.sum())),
        "recall": rate(tp, int(anomalous.sum())),
        # Independent of the anomaly share, which differs a lot between groups.
        "false_alarm_rate": rate(fp, int((~anomalous).sum())),
        "anomaly_share": rate(int(anomalous.sum()), int(len(frame))),
    }


def lead_stats(values: pd.Series) -> dict | None:
    values = values.dropna()
    if values.empty:
        return None
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(values.median())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replayed-lines", type=int, required=True, help="the producer's --limit")
    parser.add_argument("--bootstrap", default=BOOTSTRAP_SERVERS)
    parser.add_argument("--k", type=int, default=20, help="the processor's early-warning prefix length")
    args = parser.parse_args()

    print(f"Reading run {args.run_id} back from Kafka ...")
    decisions = pd.DataFrame(read_topic(TOPIC_STORAGE_DECISIONS, args.run_id, args.bootstrap))
    alerts = pd.DataFrame(read_topic(TOPIC_ALERTS, args.run_id, args.bootstrap))
    if decisions.empty:
        sys.exit(f"No storage decisions for run {args.run_id}")
    decisions = decisions.drop_duplicates("block_id", keep="last").set_index("block_id")
    print(f"  {len(decisions):,} block decisions, {len(alerts):,} alerts")

    labels = pd.read_csv(LABELS).set_index("BlockId")["Label"]
    decisions["is_anomaly"] = decisions.index.map(labels).fillna("Normal") == "Anomaly"
    decisions["unlabeled"] = ~decisions.index.isin(labels.index)

    print(f"Counting each block's lines in the full {RAW_LOG} (one pass) ...")
    prefix, total = count_lines(set(decisions.index), args.replayed_lines)
    decisions["lines_in_replay"] = decisions.index.map(prefix).fillna(0).astype(int)
    decisions["lines_in_file"] = decisions.index.map(total).fillna(0).astype(int)
    decisions["complete"] = decisions["lines_in_replay"] >= decisions["lines_in_file"]

    early = alerts[alerts["type"] == "early_warning"].set_index("block_id") if not alerts.empty else pd.DataFrame()
    decisions["early_alert"] = decisions.index.isin(early.index)
    decisions["block_alert"] = decisions["predicted_label"] == "Anomaly"
    # Seconds of log time from the early warning (k-th event) to the block's last line seen.
    if not early.empty:
        alert_time = early["event_time"].reindex(decisions.index)
        last = pd.to_datetime(decisions["last_event_time"])
        decisions["lead_seconds"] = (last - pd.to_datetime(alert_time)).dt.total_seconds()
    else:
        decisions["lead_seconds"] = np.nan

    def section(frame: pd.DataFrame) -> dict:
        true_early = frame[frame["early_alert"] & frame["is_anomaly"]]
        return {
            "block_anomaly": detection(frame, frame["block_alert"]),
            "early_warning": {
                **detection(frame, frame["early_alert"]),
                "lead_fraction_true_alerts": lead_stats(true_early["early_warning_lead"]),
                "lead_seconds_true_alerts": lead_stats(true_early["lead_seconds"]),
                "true_alerts_before_block_end": int((true_early["early_warning_lead"] > 0).sum()),
                # An early warning can only fire once a block reaches k events.
                "on_blocks_reaching_k": detection(frame[frame["events"] >= args.k],
                                                  frame.loc[frame["events"] >= args.k, "early_alert"]),
            },
            "lines_per_block": {
                "mean": float(frame["events"].mean()) if len(frame) else None,
                "median": float(frame["events"].median()) if len(frame) else None,
            },
        }

    complete, cut_off = decisions[decisions["complete"]], decisions[~decisions["complete"]]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "replayed_lines": args.replayed_lines,
        "definitions": {
            "complete": "all of the block's lines in the full HDFS.log fall inside the replayed prefix",
            "cut_off": "the block has lines after the replayed prefix, so it was decided on part of its life",
            "lead_fraction": "share of the block's events still to come when the early warning fired",
            "lead_seconds": "log seconds from the early warning (k-th event) to the block's last line in the replay",
        },
        "blocks": {
            "total": int(len(decisions)),
            "complete": int(len(complete)),
            "cut_off": int(len(cut_off)),
            "by_finalize_reason": decisions["reason"].value_counts().to_dict(),
            "timeout_but_cut_off": int(((decisions["reason"] == "timeout") & ~decisions["complete"]).sum()),
            "unlabeled": int(decisions["unlabeled"].sum()),
        },
        "complete": section(complete),
        "cut_off": section(cut_off),
        "all": section(decisions),
        "by_finalize_reason": {reason: section(group) for reason, group in decisions.groupby("reason")},
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"\nBlocks: {len(decisions):,} = {len(complete):,} complete + {len(cut_off):,} cut off "
          f"(finalize reasons {report['blocks']['by_finalize_reason']}; "
          f"{report['blocks']['timeout_but_cut_off']:,} timed out but had more lines later)")
    for name in ("complete", "cut_off"):
        for kind in ("block_anomaly", "early_warning"):
            r = report[name][kind]
            p = "—" if r["precision"] is None else f"{r['precision']:.3f}"
            rc = "—" if r["recall"] is None else f"{r['recall']:.3f}"
            print(f"  {name:<9} {kind:<14} blocks {r['blocks']:>7,}  anomalies {r['anomalies']:>5,}  "
                  f"alerts {r['alerts']:>5,}  true {r['true_alerts']:>5,}  precision {p}  recall {rc}  "
                  f"false-alarm rate {r['false_alarm_rate']:.4f}")
        e = report[name]["early_warning"]["on_blocks_reaching_k"]
        print(f"  {name:<9} {'EW, >= k events':<14} blocks {e['blocks']:>7,}  anomalies {e['anomalies']:>5,}  "
              f"alerts {e['alerts']:>5,}  true {e['true_alerts']:>5,}  precision {e['precision'] or 0:.3f}  "
              f"recall {e['recall'] or 0:.3f}  false-alarm rate {e['false_alarm_rate'] or 0:.4f}")
    lead = report["complete"]["early_warning"]
    if lead["lead_seconds_true_alerts"]:
        f, s = lead["lead_fraction_true_alerts"], lead["lead_seconds_true_alerts"]
        print(f"  complete-block early-warning lead: {f['mean']:.3f} of the block (median {f['median']:.3f}); "
              f"{s['mean'] / 3600:.2f} h mean, {s['median'] / 3600:.2f} h median before the block's last line")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
