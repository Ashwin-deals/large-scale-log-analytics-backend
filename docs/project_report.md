# MorphGuard — Large-Scale Log Analytics for Cloud Security

## Project Report

---

### Executive Overview

| Attribute | Details |
|---|---|
| **System** | MorphGuard Anomaly Detection Platform |
| **Target Logs** | HDFS (primary), OpenStack nova logs (second labeled source) |
| **Core Detection** | Isolation Forest with Genetic Algorithm Optimization |
| **Storage Policy** | Smart Storage: GA-tuned HOT / WARM / COLD retention |
| **Early Warning** | Per-k Isolation Forests on each block's first k events |
| **Streaming** | Apache Kafka (single KRaft broker) + stream processor |
| **API** | Flask RESTful + JWT Authentication |
| **Database** | MongoDB / MongoDB Atlas |
| **Repository** | `Ashwin-deals/large-scale-log-analytics-backend` |

---

## 1. Abstract

MorphGuard is an evolutionary real-time log analytics backend engineered for cloud security monitoring and behavioral anomaly detection. Combining high-throughput log ingestion, automated feature extraction, and machine learning, MorphGuard identifies operational and security anomalies within distributed infrastructure logs. A Genetic Algorithm (GA) continuously optimizes feature subsets and Isolation Forest hyperparameters, enabling the detection engine to adapt to shifting log profiles.

---

## 2. Architecture & Components

```
Log Sources (HDFS / Cloud) ──> Ingestion Engine ──> Feature Engineering
                                                          │
Alerting Dashboard <── REST API (Flask/JWT) <── Detection Model (Isolation Forest)
                                                          │
Database (MongoDB) <────────── GA Evolutionary Optimizer ─┘
```

The system comprises five core subsystems:
1. **Parser Subsystem (`parser/`)**: Extends `BaseParser` for structured log normalization.
2. **Feature Engineering (`feature_engineering/`)**: Computes block-level frequency counts, transitions, and duration metrics.
3. **Detection Engine (`detection/`)**: Encapsulates Isolation Forest training, inference, and F1 evaluation.
4. **Genetic Algorithm Optimizer (`optimization/`)**: Executes evolutionary search over chromosome configurations.
5. **API & Persistence Layer (`app.py`, `auth.py`, `db.py`)**: Secured with JWT access tokens and role-based policies.

---

## 3. Machine Learning & Evolutionary Optimization

### 3.1 Feature Representation
Logs are aggregated per unique `block_id` to form feature vectors spanning:
- Event code frequencies (e.g., `E1` through `E29`)
- Operational sequence indicators (block allocations, terminations)
- Lifecycle duration and message count ratios

### 3.2 Detection Model: Isolation Forest
Unsupervised tree ensembles isolate anomalies based on path lengths:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Points with shorter average path lengths require fewer partitions and are flagged as anomalous.

### 3.3 Genetic Algorithm Formulation
Candidate solutions are represented as binary and continuous chromosomes:

| Component | Encoding | Search Range |
|---|---|---|
| Feature Mask | Binary string (N genes) | `{0, 1}` |
| `n_estimators` | Integer gene | `[50, 200]` |
| `max_samples` | Float / Ratio | `[0.2, 1.0]` |
| `contamination` | Continuous float | `[0.01, 0.30]` |

**Fitness Function**:
$$\text{Fitness} = \text{Macro F1-Score} - \lambda \cdot (\text{Feature Ratio})$$

Selection is performed via tournament selection ($k=3$), followed by uniform crossover ($p_c=0.8$) and bit-flip / Gaussian mutation ($p_m=0.15$).

---

## 4. Dataset & Benchmarking

The primary evaluation uses the HDFS benchmark from the LogHub dataset:
- **Total Log Lines**: 11,175,629
- **Unique Blocks**: 575,061
- **Ground Truth**: 16,838 anomaly blocks (~2.9% contamination rate)
- **Train/Test Split**: 80/20 stratified split

### Performance Progression

| Model Phase | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|
| Baseline Isolation Forest | 0.812 | 0.745 | 0.777 | Default sklearn params |
| Hand-tuned Model | 0.864 | 0.810 | 0.836 | 15 selected features |
| GA-Optimized Model | **0.932** | **0.918** | **0.925** | Evolved chromosome |

The GA optimization improved F1 score by **+14.8%** over baseline while reducing feature dimensionality by 35%.

---

## 5. System Architecture (current)

The diagram shows Smart Storage after the anomaly-score step, the Kafka streaming layer, and early warning. See `docs/architecture.md` §3.6–3.9 for each subsystem.

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

---

## 6. Smart Storage

**Goal.** Keep the logs that matter and drop the rest. The importance score, tiers and GA search are defined in `docs/architecture.md` §3.6. Evaluation uses a 50/50 stratified split: the deployed model's hyperparameters are refitted on the tuning half, so held-out blocks are scored by a model that never saw them. Every baseline is scored on the held-out half with the same penalised fitness.

**Configuration** (`data/storage/storage_report.json`):
- Retention target 0.95, WARM credit 0.5, penalty weight 1000, fixed error weight 0.2.
- GA population 80, 80 generations, seed 42.
- Split: 287,530 tuning blocks and 287,531 held-out blocks, each with 8,419 anomalies.

**Learned policy:**
- Weights 0.750 (anomaly score), 0.073 (event frequency), 0.177 (block size).
- HOT threshold 0.1723, WARM threshold 0.0493, summary fraction 0.0565.

**Held-out results.** The "clean" column uses the tune-half refit. The "deployed" column scores with the deployed model, which was fitted on all blocks; it's kept for comparison.

| Policy | Saved % (clean) | Weighted retention (clean) | Fitness (clean) | Saved % (deployed) | Weighted retention (deployed) |
|---|---|---|---|---|---|
| **GA policy** | **68.76** | **0.9676** | **68.76** | 64.19 | 0.9662 |
| Keep all | 0.00 | 1.0000 | 0.00 | 0.00 | 1.0000 |
| Keep only model-flagged blocks | 93.00 | 0.4559 | -401.13 | 92.74 | 0.4691 |
| Top-k% by score, k tuned on tune half (k = 38.74%) | 58.49 | 0.9461 | 54.57 | 55.76 | 0.9408 |
| Top-k% by score at the GA's saving | 68.76 | 0.9425 | 61.27 | 64.19 | 0.7215 |
| Random sampling at the GA's saving | 68.76 | 0.2745 | -606.77 | 64.19 | 0.3223 |

**Held-out tiers.**

| Tier | Blocks | Lines kept | Anomalies |
|---|---|---|---|
| HOT | 62,961 | 1,520,577 | 7,877 |
| WARM | 215,399 | 215,449 | 539 |
| COLD | 9,171 | 9,171 | 3 |

That is 1,745,197 of 5,587,093 lines kept. Full (HOT-only) retention is 0.9356 and coverage (HOT + WARM) is 0.9996.

**Limitation.** WARM holds 74.9% of held-out blocks, but 99.98% of them keep exactly one line. That is the same cost as COLD, with partial credit. Because WARM credit does not depend on summary size, the GA learns the smallest summary fraction that still rounds to one line.

---

## 7. Early Warning

**Goal.** Flag a risky block before it finishes.
- **Features:** for k ∈ {3, 5, 10, 20}, features are built from each block's first k events only, in one chunked pass over the cleaned log. The pass covered 11,173,511 rows and 575,061 blocks in 23.3 s, with 0 rows out of time order.
- **Model:** one Isolation Forest per k, fitted on a stratified 70% training split (402,542 blocks, 11,787 anomalies). A GA tunes its alert threshold for F1 on that training split.
- **Evaluation:** the untouched 30% test split (172,519 blocks, 5,051 anomalies).
- **Lead time:** the fraction of a block's events not yet seen when the alert fires.

**Results on test blocks** (`data/evaluation/early_warning_report.json`):

| k | Alert quantile | Precision | Recall | F1 | False-alarm rate | Mean lead, true alerts | Expected lead per alert |
|---|---|---|---|---|---|---|---|
| 3 | 0.9698 | 0.1834 | 0.1966 | 0.1898 | 2.64% | 0.067 | 0.012 |
| 5 | 0.9874 | 0.8833 | 0.3671 | 0.5186 | 0.15% | 0.012 | 0.011 |
| 10 | 0.9867 | 0.8812 | 0.3979 | 0.5483 | 0.16% | 0.049 | 0.043 |
| 20 | 0.9801 | 0.8490 | 0.5722 | 0.6836 | 0.31% | 0.124 | 0.105 |

**Results on blocks still running when the alert fires** (more than k events):

| k | Blocks | Anomalies | Precision | Recall | F1 | Mean lead |
|---|---|---|---|---|---|---|
| 3 | 171,604 | 4,136 | 0.0189 | 0.0206 | 0.0197 | 0.779 |
| 5 | 170,693 | 3,225 | 0.1026 | 0.0087 | 0.0160 | 0.797 |
| 10 | 170,689 | 3,221 | 0.3991 | 0.0559 | 0.0980 | 0.551 |
| 20 | 42,339 | 2,009 | 0.6961 | 0.5266 | 0.5996 | 0.339 |

**Trade-off.** The Pareto front on precision versus mean lead time is k = 5, 10 and 20. The best trade-off is **k = 20**, on both views. At small k, the headline precision comes from short anomalous blocks that have already ended; only k = 20 warns early on blocks still in progress.

---

## 8. Real-Time Streaming Layer (Apache Kafka)

- **Broker:** `docker-compose.yml` runs a single Kafka broker in KRaft mode (`apache/kafka`) with the topics `logs.raw`, `alerts` and `storage.decisions`.
- **Producer:** `streaming/producer.py` replays HDFS log lines at a configurable `--rate` and `--limit`, keyed by block id.
- **Processor:** `streaming/processor.py` (confluent-kafka) parses every line with `parser/hdfs_parser.py` and keeps per-block running state. Its features are tested to equal the batch builders exactly.
  - When a block reaches k events, it raises an `early_warning` alert if the early-warning model scores the first-k snapshot at or above the GA-tuned threshold. The default is k = 20, the best trade-off in §7.
  - When a block completes or times out, it computes the deployed model's full score (an Anomaly raises a `block_anomaly` alert) and the Smart Storage tier.
  - Alerts are published to `alerts` and saved to the MongoDB collection `stream_alerts`. Decisions are published to `storage.decisions`.
- **API:** `GET /api/stream/status` serves live counters (events/sec, blocks in flight, alert counts, storage saved so far) and `GET /api/stream/alerts` serves the alerts. The dashboard's Live Stream page polls both.
- **Setup:** `docs/streaming_guide.md` has the exact start-up commands.

---

## 9. OpenStack as a Second Labeled Source

- **Data:** OpenStack nova logs from LogHub: two normal runs and one abnormal run.
- **Parser:** `parser/openstack_parser.py` implements `BaseParser` and outputs the same cleaned columns as the HDFS parser. `block_id` is the VM instance UUID; lines without one are dropped.
- **Labels:** each instance session is labeled by the file it came from: Anomaly if it appears in the abnormal run's file.
- **Models:** the common feature extractor, a baseline Isolation Forest and the GA-optimised model are trained under `data/models/openstack/` with their own vocabulary, versions and metrics. The HDFS V2 model is untouched.
- **Uploads:** `POST /api/sources/upload` accepts `source=openstack`.

**Ingest.** 207,820 lines in total:
- 184 did not match the line format.
- 156,083 carried no instance id.
- 1,878 were exact duplicates.

That leaves 49,675 events in 2,069 sessions, of which 198 (9.57%) are from the abnormal run. The split is stratified 70/30: 1,448 training and 621 test sessions, 59 of them anomalous.

**Results on the 621 test sessions:**

| Model | Precision | Recall | F1 | Accuracy | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|
| V1 baseline | 0.0759 | 0.2034 | 0.1106 | 0.6892 | [[416, 146], [47, 12]] |
| GA-optimised, promoted to OpenStack V2 | 0.1043 | 0.2034 | 0.1379 | 0.7585 | [[459, 103], [47, 12]] |

**GA chromosome.**
- Features: `hour_of_day`, `block_size`, `event_frequency`, `source_ip_encoded`, `destination_ip_encoded`.
- Hyperparameters: `n_estimators` 175, `max_samples` 0.344, `contamination` 0.187.

**Smart Storage on OpenStack.** On 1,035 held-out sessions (99 anomalies) it saves 31.42% of lines. Weighted retention is 0.9293, below the 0.95 target; full retention is 0.8687 and coverage 0.9899.

**Early warning on OpenStack.** The test set has 59 anomalous sessions.
- At k = 3, 5 and 10, the tuned threshold alerts on every session: precision 0.095 (the base rate), recall 1.0, F1 0.1735.
- At k = 20: precision 0.1054, recall 0.5254, F1 0.1756.

**Why these numbers are low:**
- **The labels mark the run, not the fault.** The dataset's own `anomaly_labels.txt` names only 4 faulty instances; the deployed model catches 0 of those 4 while flagging 380 sessions overall.
- **The runs are separable by time of day.** `hour_of_day` alone separates the runs with AUC 0.849, while every other feature sits at about 0.50.

---

## 10. Security & Threat Mitigation

MorphGuard adheres to defense-in-depth principles:
- **Authentication**: Stateless JWT with HMAC-SHA256 signatures and token expiry.
- **Password Protection**: bcrypt key derivation with salted cost factor.
- **Transport Security**: TLS 1.3 encryption across Atlas cluster connections.
- **Input Sanitization**: Path normalization and regex validation on all ingestion routes.

---

## 11. API Reference Summary

Key endpoints provided by the Flask application:
- `POST /api/v1/auth/login`: Issue bearer token.
- `POST /api/v1/ingest`: Stream new log entries into feature store.
- `POST /api/v1/detect`: Run real-time inference on feature vectors.
- `POST /api/v1/optimize/run`: Trigger asynchronous GA generation cycle.
- `GET /api/v1/models/history`: Fetch version lineage and evaluation metrics.

---

## 12. Future Directions

Planned enhancements include:
1. **Streaming Pipeline**: Apache Kafka integration is now implemented (§8). Remaining work: durable per-block state across restarts, and an explicit block-completion signal in place of the idle timeout.
2. **Deep Learning Detectors**: Autoencoder and Graph Neural Network (GNN) modules.
3. **Multi-Tenant Isolation**: Namespace partitioning for enterprise cloud tenants.
4. **Automated Rollback**: Canary verification with automatic model rollback.

---

## 13. References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE ICDM.
2. He, P., et al. (2020). *Loghub: A large collection of system log datasets towards automated log analytics*. IEEE SRDS.
3. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
4. Du, M., et al. (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. ACM CCS.
