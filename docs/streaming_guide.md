# Streaming guide

The real-time layer replays HDFS log lines through Kafka and scores them with the
models the batch pipeline already trained. It does not change the batch pipeline.

```
HDFS.log --producer--> logs.raw --processor--> alerts             (Kafka)
                                     |      \-> storage.decisions  (Kafka)
                                     \-> MongoDB stream_alerts / stream_status
                                              \-> GET /api/stream/status, /api/stream/alerts
```

## Prerequisites

- Docker Desktop running.
- The backend set up as in `setup_guide.md` (venv, `pip install -r requirements.txt`, `.env` with `MONGO_URI`).
- Models on disk. The processor loads:
  - `data/models/early_warning/isolation_forest_k20.pkl`, from `python3 scripts/build_early_warning.py`
  - the deployed model in `data/models/current_version.json`
  - `data/storage/policy_best.json` and its tune-half model, from `python3 scripts/optimize_storage_policy.py`
- `data/raw/hdfs/HDFS.log`, from `python3 scripts/download_dataset.py`.

Run every command from `large-scale-log-analytics-backend/` with the venv active.

## 1. Start Kafka

```bash
docker compose up -d                 # single broker, KRaft mode, localhost:9092
docker compose logs kafka-init       # should list: alerts, logs.raw, storage.decisions
```

## 2. Start the API (optional, for the status endpoints)

```bash
python3 app.py
```

## 3. Start the processor

```bash
python3 -m streaming.processor --from-beginning --idle-flush 8 --stop-after-idle 20
```

- `--from-beginning` uses a fresh consumer group and reads `logs.raw` from offset 0.
- `--k 20` is the default prefix length. It was the best trade-off in
  `data/evaluation/early_warning_report.json`, and `--k 5 10 20` alerts at the first k that fires.
- `--idle-timeout` finalizes a block after that many seconds of *log* time with no events. The
  default depends on the source (`streaming/sources.py` `IDLE_TIMEOUTS`): HDFS 43,663 s, OpenStack and
  CloudTrail 3,600 s, Apache 120 s, Django never (end of stream only). `0` means never.
- **Why 43,663 s for HDFS.** It is the smallest timeout that never closes 99% of blocks early.
  `scripts/measure_block_gaps.py` measured the longest log-time gap between two consecutive lines of
  each of the 575,061 blocks in the full `HDFS.log` (`data/evaluation/hdfs_block_gap_stats.json`):

  | Percentile of blocks | p50 | p95 | p99 | p99.9 | max |
  |---|---:|---:|---:|---:|---:|
  | Longest quiet gap | 6,848 s (1.9 h) | 38,227 s (10.6 h) | **43,663 s (12.1 h)** | 46,261 s (12.9 h) | 53,611 s (14.9 h) |

  An HDFS block is usually written, then read or deleted hours later, so the old 3,600 s default
  covered only 39.3% of blocks.
- `--idle-flush 8` finalizes every block still in flight once the topic has been quiet for 8 wall seconds.
- `--stop-after-idle 20` exits 20 s after the last message. Leave it off to keep running.
- `--no-mongo` publishes alerts and decisions to Kafka only and never connects to MongoDB. Use it for
  offline replays scored with `scripts/analyze_stream_run.py`, which reads Kafka.

## 4. Replay log lines

In a second terminal:

```bash
python3 -m streaming.producer --limit 50000 --rate 5000     # 50,000 real lines at 5,000/s (~10 s)
python3 -m streaming.producer --limit 1000000 --rate 0     # the B3 verification replay (~3 s to send)
```

`--rate 0` sends as fast as possible, and leaving out `--limit` replays the whole file.

## 5. Watch it

```bash
TOKEN=...   # a JWT from POST /api/auth/login
curl -H "Authorization: Bearer $TOKEN" localhost:5000/api/stream/status
curl -H "Authorization: Bearer $TOKEN" "localhost:5000/api/stream/alerts?limit=20&type=early_warning"

# raw topics
docker exec lsla-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:19092 \
  --topic alerts --from-beginning --max-messages 5
```

`/api/stream/status` returns events/sec over the last 5 s, blocks in flight, alert counts,
storage saved so far, and tier counts. `live` turns false 10 s after the processor stops reporting.

## Reading the numbers

- **Alert types.** There are two:
  - `early_warning` fires when a block reaches k events and its first-k score is at or above the
    GA-tuned threshold, at most once per block. `events_seen` is k and `event_time` is the k-th event's time.
  - `block_anomaly` fires when the deployed model flags a finalized block.
- **Block completion is a timeout.** HDFS has no "block finished" event, so a block is finalized
  after `--idle-timeout` of log-time silence. A line for an already-finalized block is counted as
  `late_events` and ignored.
- **A replay cuts blocks off mid-life, and so does the idle timeout.** A truncated block looks anomalous
  to a whole-block model. Alerts carry `finalize_reason`, and `end_of_stream` marks a forced finalization.
  To score a run against the labels, split by whether each block was really complete:

  ```bash
  python3 scripts/analyze_stream_run.py --run-id <run id> --replayed-lines <producer --limit>
  ```

  **Verification replay (B3).** The first 1,000,000 HDFS lines (about 3.5 hours of log time) were replayed at full speed, run id `b3-1m-v3`, with V3 live and the 43,663 s HDFS timeout. `scripts/analyze_stream_run.py` read every decision and alert back from Kafka and joined them with `anomaly_label.csv`. A block counts as **complete** only if all of its lines in the full `HDFS.log` fell inside the replay (`data/evaluation/stream_run_report.json`).

  | Blocks (66,167 decided) | Blocks | Anomaly share | Block-anomaly precision | Recall | False-alarm rate |
  |---|---:|---:|---:|---:|---:|
  | Complete | 2,528 | 51.5% | **0.937** | **0.962** | **6.85%** |
  | Cut off | 63,639 | 2.6% | 0.027 | 0.477 | 46.6% |

  | Early warning (k = 20) | Blocks reaching 20 events | Alerts | True | Precision | Recall | False-alarm rate |
  |---|---:|---:|---:|---:|---:|---:|
  | Complete blocks | 88 | 22 | 21 | **0.955** | 0.247 | — (3 normal blocks) |
  | Cut-off blocks | 2,172 | 142 | 131 | **0.923** | 0.728 | 0.55% |

  - **No block was closed early.** The replay spans about 3.5 h of log time, less than the timeout, so all 66,167 blocks were finalized by the end-of-stream flush: 0 timed out, 0 late events. With the old 3,600 s timeout, 34,321 blocks timed out and then got more lines.
  - **Lead time on complete blocks:** a true early warning fired with a median 28.6% of the block still to come, a median 285 s (4.8 min) before the block's last line. For cut-off blocks the end is outside the replay; the median of 1,777 s to the last line *seen* is only a lower bound.
  - **Complete blocks are mostly anomalous** (51.5% against 2.9% overall), because failing blocks end early. That inflates precision, so the false-alarm rate is the fairer number: 6.85% of complete normal blocks, against 0.58% for batch V3 on all blocks.
  - **The whole-block verdict is unreliable on partial blocks.** That covers 29,680 of the 31,015 block-anomaly alerts, nearly all of them false. These are replay artifacts: the end-of-stream flush scores 63,639 blocks that still had lines to come. A live stream has no such flush, so those blocks would stay open. Early warning does not have this problem: it only scores a block's first 20 events, which a partial block has.
  - **What changed from the earlier run.** The earlier run `b3-1m` had V2 live and a 3,600 s timeout: complete-block precision 0.986, recall 0.934, false-alarm rate 1.39%. A control replay with V3 and the 3,600 s timeout (run `b3-1m-v3-t3600`, `data/evaluation/stream_run_report_v3_timeout3600.json`) gives exactly the new complete-block numbers. So that change comes from the model (V3 flags more of the short complete blocks), not from the timeout. The timeout only stops blocks being closed early: 34,315 in the control run, 0 now. Early warning is the same in all three runs.
- **State is in memory.** A restarted processor starts its blocks fresh.
- **The vocabulary is read-only.** Category codes come from `data/features/category_encoders.json`.
  A value it has never seen gets a temporary in-memory code and is never written back, and status
  reports the count under `unseen_vocabulary_values`.

## Stop

```bash
docker compose down        # keep the broker's data
docker compose down -v     # also delete it
```
