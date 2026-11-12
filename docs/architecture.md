# MorphGuard — System Architecture Documentation

---

## 1. Overview

MorphGuard is an evolutionary real-time log analytics and anomaly detection system for cloud environments. It implements a layered SIEM architecture that ingests raw distributed logs, extracts per-block features, detects behavioral anomalies via Isolation Forests, and evolves detection parameters using Genetic Algorithms. On top of that core it adds:

- **Smart Storage**: a GA-tuned policy that keeps every line of suspicious blocks and summarizes or drops the rest.
- **Early warning**: per-k models that score a block from its first k events.
- **A Kafka streaming layer**: runs those models on logs as they arrive.
- **OpenStack**: a second labeled log source, kept separate from HDFS.

---

## 2. High-Level Architecture Diagram

Two paths share one set of models. The **batch path** parses, cleans and featurizes a whole log,
scores every block with the GA-optimised Isolation Forest, and passes the anomaly score to **Smart
Storage**, which decides how much of each block to keep. **Early warning** models are trained on the
same cleaned log, using only each block's first k events. The **real-time path** replays log lines
through **Kafka**, and the stream processor applies those batch-trained models while blocks are still
arriving.

```
                   +---------------------------------------------------------+
                   |                       LOG SOURCES                       |
                   |  HDFS.log (LogHub);  OpenStack nova logs (LogHub)       |
                   |  files uploaded through POST /api/sources/upload        |
                   +--------------+---------------------------+--------------+
                                  |                           |
                     BATCH PATH   |                           |   REAL-TIME PATH
                                  v                           v
+-------------------------------------------+   +-------------------------------------------+
| PARSING  (parser/)                        |   | KAFKA LAYER  (docker-compose.yml, KRaft)  |
| hdfs_parser.py, openstack_parser.py       |   | streaming/producer.py replays log lines   |
| one row per log line, common schema       |   | into topic logs.raw, keyed by block id    |
+---------------------+---------------------+   +---------------------+---------------------+
                      |                                               |
                      v                                               v
+-------------------------------------------+   +-------------------------------------------+
| CLEANING + FEATURE EXTRACTION             |   | STREAM PROCESSOR (streaming/processor.py) |
| data_cleaner.py -> feature_extractor.py   |   | parse each line, keep per-block running   |
| one feature row per block                 |   | state (same features as the batch path)   |
+----------+-------------------+------------+   +----------+----------------------+---------+
           |                   |                           |                      |
           |                   v                           | block reaches        | block quiet for
           |   +-------------------------------+           | k events             | the timeout, or
           |   | EARLY WARNING (early_warning/)|           v                      v end of stream
           |   | features from first k events  |   +-------------------+  +-----------------------+
           |   | Isolation Forest per k        |-->| EARLY WARNING     |  | FULL ANOMALY SCORE    |
           |   | GA-tuned alert threshold      |   | alert if score >= |  | then SMART STORAGE    |
           |   +-------------------------------+   | threshold         |  | TIER (HOT/WARM/COLD)  |
           v                                       +---------+---------+  +-----------+-----------+
+-------------------------------------------+                |                        |
| ANOMALY SCORE  (detection/)               |                v                        v
| Isolation Forest, GA-optimised            |      +-------------------+  +-----------------------+
| (optimization/ga_isolation_forest.py)     |      | topic: alerts     |  | topic:                |
+---------------------+---------------------+      | MongoDB:          |  | storage.decisions     |
                      |                            | stream_alerts     |  +-----------------------+
                      v                            +---------+---------+
+-------------------------------------------+                |
| SMART STORAGE  (smart_storage/)           |                |
| importance = weighted anomaly score,      |                |
| event frequency, block size, error flag   |                |
| -> HOT (keep all) / WARM (summary) /      |                |
|    COLD (1 metadata line)                 |                |
| GA-tuned weights and thresholds           |                |
+---------------------+---------------------+                |
                      |                                      |
                      v                                      v
+-------------------------------------------------------------------------------------------+
| FLASK API  (app.py + blueprints)  ->  REACT DASHBOARD                                     |
| /api/dashboard  /api/storage/*  /api/early-warning/summary  /api/stream/*  /api/sources/* |
+-------------------------------------------------------------------------------------------+
```

The stream processor loads the batch-trained models from disk: the early-warning model for k, the deployed Isolation Forest for the full score, and the Smart Storage policy and its model.

---

## 3. Subsystem Breakdown

### 3.1 Ingestion & Parsing (`parser/`)
- `base_parser.py`: Abstract Base Class enforcing the `parse(filepath: str) -> pd.DataFrame` contract.
- `hdfs_parser.py`: High-throughput regex extraction for block IDs, timestamps, log levels, and event templates.
- `openstack_parser.py`: OpenStack (nova) logs; one block = one VM instance session (`[instance: <uuid>]`). See §3.9.
- `django_parser.py`: Parser for application-level web server event streams.

### 3.2 Feature Engineering (`feature_engineering/`)
- `data_cleaner.py`: Deduplication, missing value imputation, and timestamp alignment.
- `feature_extractor.py`: Aggregates per `block_id` into event frequency vectors (E1–E29), sequence count, and lifecycle duration.

### 3.3 Detection Engine (`detection/`)
- `isolation_forest.py`: Unsupervised tree ensemble wrapper for scoring and thresholding.
- `train.py` & `predict.py`: Standalone CLI and API hooks for model training and real-time inference.
- `evaluate.py`: Computes Precision, Recall, Macro-F1, and Confusion Matrix against labeled ground truth.

### 3.4 Evolutionary Optimization (`optimization/`)
- `ga_isolation_forest.py`: Implements Genetic Algorithm searching over:
  - Feature selection bitmask
  - Estimator count ($n \in [50, 200]$)
  - Sample ratio ($s \in [0.2, 1.0]$)
  - Contamination threshold ($c \in [0.01, 0.30]$)
- `model_evolution.py`: Orchestrates promotion from Candidate to Champion models using historical F1 thresholds.

### 3.5 API Service & Authentication (`app.py`, `auth.py`, `db.py`)
- Flask 2.0 RESTful interface with Bearer token authentication.
- Password hashing with bcrypt.
- Connection pooling to MongoDB Atlas via TLS.

### 3.6 Smart Storage (`smart_storage/`, `storage_api.py`)
Smart Storage sits after the anomaly-score step. Each block gets an importance score: a weighted sum of four parts, each normalized on the tuning half.
- **Parts:** the Isolation Forest anomaly score, `event_frequency`, the `block_size` percentile, and a fixed-weight error flag.
- **Tiers:** the score places the block in **HOT** (keep every line), **WARM** (keep one metadata line plus a `round(f x event_frequency)`-line summary) or **COLD** (keep one metadata line). WARM therefore never costs less than COLD.
- **What the GA tunes:** the three weights, both thresholds, and `f`.
- **Fitness:** storage saved, minus a penalty when weighted anomaly retention falls below the target. Weighted retention counts HOT = 1, COLD = 0, and each WARM anomaly earns `warm_credit x min(1, summary_lines / (warm_ref_fraction x event_frequency))`: full credit (default 0.5) only when the summary keeps at least `warm_ref_fraction` (default 0.25) of the block, proportionally less below that.
- **Clean evaluation:** the deployed model's hyperparameters are refitted on a stratified tune half, so the held-out half is scored by a model that never saw it.

Results (`data/storage/storage_report.json`). These are the numbers from before the WARM fix (flat credit), still in the committed report until the GA is rerun on the full 575,061 blocks. On a 150,000-block representative sample the fixed rule gives 68.42% saved, full retention 0.9504, weighted 0.9517 and coverage 0.9668, and still beats every baseline.
- **Split:** 287,530 tuning blocks and 287,531 held-out blocks, each with 8,419 anomalies.
- **Learned policy:** weights 0.750 (anomaly score), 0.073 (event frequency) and 0.177 (block size); HOT threshold 0.1723; WARM threshold 0.0493; summary fraction 0.0565.
- **On the held-out half:** it keeps 1,745,197 of 5,587,093 lines, saving **68.76%**. Weighted retention is **0.9676** against a 0.95 target; full (HOT-only) retention is 0.9356; coverage is 0.9996.
- **Comparison:** it beats every baseline on the same penalised fitness.
- **Fixed: WARM tier collapse.** Under the old rule (WARM cost `max(1, round(f x lines))`, flat credit), WARM held 74.9% of held-out blocks and 99.98% of them kept exactly one line: the COLD cost, but with free credit. Charging the metadata line separately and making credit proportional to the summary removes that free credit. On the 150k-block sample, COLD is a real tier again (54% of held-out blocks), WARM keeps 1.33 lines per block on average, and the 0.95 target is met by full (HOT-only) retention of 0.9504 rather than by WARM credit.

### 3.7 Early Warning (`early_warning/`, `early_warning_api.py`)
Early warning flags a block from its first k events, for k in {3, 5, 10, 20}.
- **Features:** `prefix_features.py` builds them in one chunked pass over `hdfs_cleaned.csv`, keeping per-block running totals rather than the events. The features are event and type counts, elapsed time, distinct event types and components, WARN/ERROR counts, the running block size, and encoded first/dominant categories.
- **Model:** one Isolation Forest per k, fitted on a stratified 70% training split. A GA tunes its alert threshold, as a quantile of training scores, for F1.
- **Lead time:** the fraction of a block's events not yet seen when the alert fires, and the log-time seconds from the alert to the block's last line. Lead in seconds is log time from the alert (the k-th event) to the block's last log line (`lead_time_seconds` in the report). In HDFS a block's last line is often a much later read or delete, so this is time before the block's last line, not time before anyone would notice a problem.

Results (`data/evaluation/early_warning_report.json`, 172,519 test blocks with 5,051 anomalies):

| k | Precision | Recall | F1 | False-alarm rate | Mean lead (true alerts) | Mean lead, seconds | Median lead, seconds | Still-running blocks: F1 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.1834 | 0.1966 | 0.1898 | 2.64% | 0.067 | 1,734 s | 0 s | 0.0197 |
| 5 | 0.8833 | 0.3671 | 0.5186 | 0.15% | 0.012 | 283 s | 0 s | 0.0160 |
| 10 | 0.8812 | 0.3979 | 0.5483 | 0.16% | 0.049 | 1,592 s | 0 s | 0.0980 |
| 20 | 0.8490 | 0.5722 | 0.6836 | 0.31% | 0.124 | 5,165 s (1.4 h) | 0 s | 0.5996 |

The precision-vs-lead-time Pareto front is k = 5, 10 and 20. The best trade-off is **k = 20**, for all test blocks and for still-running ones.

At k ≤ 10, most correct alerts are short blocks that had already ended (1,826 of 1,854 at k = 5). Measured on blocks still running when the alert fires, only k = 20 warns early in practice: precision 0.6961, recall 0.5266, mean lead 0.339 of the block, median 2.9 h before the block's last line.

### 3.8 Real-Time Streaming Layer (`streaming/`, `stream_api.py`, `docker-compose.yml`)
- **Broker:** a single Kafka broker in KRaft mode with three topics: `logs.raw`, `alerts` and `storage.decisions`.
- **Producer:** `producer.py` replays log lines at a fixed rate, keyed by block id so each block's lines stay on one partition, in order.
- **Processor:** `processor.py` parses each line with the HDFS parser and drops lines the way the batch cleaner would. It keeps per-block running state whose features equal the batch prefix and full-block features; a test checks this parity.
- **On the processor's side:**
  - When a block reaches k events, it scores the first-k snapshot. An `early_warning` alert fires if the score is at or above the GA-tuned threshold.
  - When a block goes quiet for the idle timeout (in log time), or the stream ends, it computes the deployed model's full score and the Smart Storage tier. An Anomaly verdict raises a `block_anomaly` alert.
- **Outputs:**
  - Alerts go to the `alerts` topic and the MongoDB collection `stream_alerts`.
  - Tier decisions go to `storage.decisions`.
  - A status document is upserted every 2 s and served by `GET /api/stream/status`.
  - An early warning's lead time is written back onto its alert once the block finishes.
- **Limitations:**
  - HDFS has no "block finished" event, so completion means a timeout. In the 1,000,000-line verification replay, 34,321 of 36,279 timed-out blocks had more lines later, and whole-block verdicts on those partial blocks were mostly false (precision 0.145). On truly complete blocks precision was 0.986 and the false-alarm rate 1.39%. Early warning stayed precise in both cases (0.92 to 0.96). See `docs/streaming_guide.md` and `data/evaluation/stream_run_report.json`.
  - Per-block state is in memory, so it does not survive a restart.

### 3.9 OpenStack Source (`parser/openstack_parser.py`, `multi_source/`)
OpenStack (LogHub nova logs) is a second labeled source. It reuses the HDFS cleaner, feature extractor, Isolation Forest and GA, with its own vocabulary, models, versions and metrics under `data/models/openstack/` and `data/evaluation/openstack_*`. The HDFS V2 model is never touched.
- **Blocks:** one block is one VM instance session. Lines without an instance id are dropped: 156,083 of 207,820 lines, leaving 49,675 events in 2,069 sessions.
- **Labels:** a session is labeled by the file it came from. Sessions from `openstack_abnormal.log` are Anomaly: 198 of 2,069, or 9.57%.

Results on the held-out test set (`data/evaluation/openstack_report.json`; 621 sessions, 59 anomalies):

| Model | Precision | Recall | F1 | Accuracy | Caught |
|---|---|---|---|---|---|
| V1 baseline Isolation Forest | 0.0759 | 0.2034 | 0.1106 | 0.6892 | 12 of 59 |
| GA-optimised (promoted to OpenStack V2) | 0.1043 | 0.2034 | 0.1379 | 0.7585 | 12 of 59 |

Two caveats:
- **The labels mark the run, not the fault.** The dataset's own `anomaly_labels.txt` lists only 4 faulty instances, and the deployed model catches 0 of those 4.
- **The runs are separable by time of day.** `hour_of_day` alone separates them with AUC 0.849, while every other feature sits near 0.50. The abnormal run is a different day from the normal runs.

---

## 4. Data Flow Pipeline

```
Raw Log Stream
      │
      ▼
Regex Tokenization & Parsing  ──>  Structured DataFrame
                                             │
                                             ▼
                                   Block Aggregation & Scaling
                                             │
                                             ▼
                                   Feature Matrix (X)
                                       │            │
                                       ▼            ▼
                                 Active Model   GA Evaluator
                                 (Predict API)  (Candidate Refinement)
                                       │            │
                                       ▼            ▼
                                   Alert Stream  Model Registry
```

---

## 5. Security & Deployment

- **Containerization**: Stateless Flask containers scalable behind an NGINX reverse proxy.
- **Model Registry**: Atomic updates with JSON metadata tracking commit hash, F1 score, and active feature masks.
- **Secrets Management**: Environment-variable based credentials (`JWT_SECRET`, `MONGO_URI`).
