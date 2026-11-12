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
- `--idle-timeout 3600` finalizes a block after an hour of *log* time with no events.
- `--idle-flush 8` finalizes every block still in flight once the topic has been quiet for 8 wall seconds.
- `--stop-after-idle 20` exits 20 s after the last message. Leave it off to keep running.

## 4. Replay log lines

In a second terminal:

```bash
python3 -m streaming.producer --limit 50000 --rate 5000     # 50,000 real lines at 5,000/s (~10 s)
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
- **A short replay cuts blocks off mid-life.** 50,000 lines is only about 12 minutes of log time.
  In the verification run, 4,048 of the 4,087 blocks were forced out by the end-of-stream flush before they
  had finished (median 13 of 25 events), and a truncated block looks anomalous to a whole-block model.
  Of the 39 blocks that were complete, all 39 were flagged and all 39 are true anomalies. Of the 4,048 cut-off ones,
  585 were flagged but only 16 are true anomalies. Alerts carry `finalize_reason`, and `end_of_stream`
  marks a forced finalization. Replay more of the log (e.g. `--limit 1000000 --rate 0`) for
  timeout-based completion.
- **State is in memory.** A restarted processor starts its blocks fresh.
- **The vocabulary is read-only.** Category codes come from `data/features/category_encoders.json`.
  A value it has never seen gets a temporary in-memory code and is never written back, and status
  reports the count under `unseen_vocabulary_values`.

## Stop

```bash
docker compose down        # keep the broker's data
docker compose down -v     # also delete it
```
