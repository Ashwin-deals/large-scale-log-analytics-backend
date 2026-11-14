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
Logs are grouped by `block_id` (for HDFS, one storage block). Each block has 8 numeric features (`feature_engineering/feature_extractor.py`):
- `hour_of_day`: hour of the block's first event
- `is_error`: 1 if any line is ERROR. It is always 0 in HDFS, which has no ERROR lines.
- `component_encoded`, `event_type_encoded`, `source_ip_encoded`, `destination_ip_encoded`: the block's most frequent value of each, as a stable category code
- `block_size`: largest size seen
- `event_frequency`: number of log lines

Category codes come from a persisted vocabulary and are never renumbered. Every model is stamped with it, and scoring refuses to run if codes have changed. The GA chooses which of the 7 training features a model uses; the live V3 uses `block_size` and `event_frequency`; V2 used `component`, `event_frequency` and `destination_ip`. Features are not scaled: Isolation Forest splits on raw values. The Autoencoder comparison (§12) standardizes its inputs.

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

The primary evaluation uses the HDFS_v1 benchmark from LogHub:
- **Raw log lines**: 11,175,629 (11,173,511 events after cleaning)
- **Unique blocks**: 575,061
- **Ground truth**: 16,838 anomalous blocks (2.93%)

### Performance progression (in-sample)

Every model is scored on all 575,061 blocks, the same blocks it was fitted on (`data/evaluation/*_metrics.json`).

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| V1 baseline Isolation Forest | 0.0656 | 0.6514 | 0.1191 | Default sklearn settings, all 7 features |
| V2 GA-optimized | 0.6037 | 0.4705 | 0.5288 | 3 of 7 features, 150 trees, contamination 0.0235 |
| **V3 GA-optimized (deployed)** | **0.7069** | **0.4666** | **0.5622** | 2 of 7 features, 75 trees, contamination 0.0209. Promoted 9 Sep, rolled back by hand the same day, re-promoted 14 Nov |

### Held-out performance

`scripts/evaluate_holdout.py` refits each version's hyperparameters on a stratified 70% split of all 575,061 blocks and scores the unseen 30% (172,519 blocks, 5,051 anomalies), repeated over five splits (`data/evaluation/holdout_comparison.json`).

| Model | Train F1 | Test F1 | Test F1 over 5 splits |
|---|---|---|---|
| V1 | 0.1364 | 0.1370 | 0.128 ± 0.008 |
| V2 | 0.5255 | 0.5224 | 0.528 ± 0.014 |
| **V3 (deployed)** | 0.5610 | **0.5539** | **0.555 ± 0.009** |

The GA improves F1 over the baseline 4.7x in-sample (0.1191 to 0.5622) and 4.3x held out (0.128 to 0.555), while using 2 of the 7 features. Train and test F1 are level, so the models do not overfit. **V3 is the best on unseen data**: higher F1 than V2 on all five splits (0.555 against 0.528), with higher precision at about the same recall. It was rolled back to V2 by hand on 9 Sep with no recorded reason; nobody on the team could recall one, so on 14 Nov it was re-promoted on the held-out evidence and Smart Storage was re-tuned for it. The full story is in `data/evaluation/v1_v2_v3_comparison.md`.

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

**Learned policy** (scored by V3):
- Weights 0.812 (anomaly score), 0.118 (event frequency), 0.069 (block size).
- HOT threshold 0.1259, WARM threshold 0.1175, summary fraction 0.049 (WARM keeps 1 metadata line plus about 5% of the block).

**Held-out results.** The "clean" column uses the tune-half refit. The "deployed" column scores with the deployed model, which was fitted on all blocks; it's kept for comparison.

| Policy | Saved % (clean) | Weighted retention (clean) | Fitness (clean) | Saved % (deployed) | Weighted retention (deployed) |
|---|---|---|---|---|---|
| **GA policy** | **68.71** | **0.9555** | **68.71** | 68.65 | 0.9478 |
| Keep all | 0.00 | 1.0000 | 0.00 | 0.00 | 1.0000 |
| Keep only model-flagged blocks | 93.44 | 0.4667 | -389.87 | 93.49 | 0.4638 |
| Top-k% by score, k tuned on tune half (k = 23.31%) | 68.27 | 0.9291 | 47.36 | 68.36 | 0.9311 |
| Top-k% by score at the GA's saving | 68.71 | 0.8876 | 6.35 | 68.65 | 0.8979 |
| Random sampling at the GA's saving | 68.71 | 0.2749 | -606.37 | 68.65 | 0.2755 |

**Held-out tiers.**

| Tier | Blocks | Lines kept | Anomalies |
|---|---|---|---|
| HOT | 63,858 | 1,496,958 | 8,026 |
| WARM | 27,349 | 54,698 | 211 |
| COLD | 196,324 | 196,324 | 182 |

That is 1,747,980 of 5,587,093 lines kept. Full (HOT-only) retention is 0.953, weighted 0.955 and coverage (HOT + WARM) 0.978, so the policy meets the 0.95 target on these unseen blocks (the tuning half gives 0.959) and beats every baseline on the penalised fitness. With V2 scoring, the same search reached only 0.948 held out: V3's sharper anomaly scores let more anomalies go HOT for the same saving.

**The WARM fix behind these numbers.** Under the old rule, WARM held 74.9% of held-out blocks and 99.98% of them kept exactly one line, the cost of COLD but with a flat 0.5 credit; that gave 68.76% saved and weighted retention 0.968 with only 0.936 kept in full. WARM now keeps one metadata line plus `round(f x lines)` summary lines, and a WARM anomaly earns `0.5 x min(1, summary_lines / (0.25 x lines))`. Free credit gone, the GA keeps more anomalies in full (0.953 under V3) and no WARM block is reduced to the one-line COLD cost. Under V3 WARM is a small tier (9.5% of blocks, about 2 of 23.5 lines each); the retention comes from HOT.

---

## 7. Early Warning

**Goal.** Flag a risky block before it finishes.
- **Features:** for k ∈ {3, 5, 10, 20}, features are built from each block's first k events only, in one chunked pass over the cleaned log. The pass covered 11,173,511 rows and 575,061 blocks in 23.3 s, with 0 rows out of time order.
- **Model:** one Isolation Forest per k, fitted on a stratified 70% training split (402,542 blocks, 11,787 anomalies). A GA tunes its alert threshold for F1 on that training split.
- **Evaluation:** the untouched 30% test split (172,519 blocks, 5,051 anomalies).
- **Lead time:** the fraction of a block's events not yet seen when the alert fires, and the log-time seconds from the alert to the block's last line. Lead in seconds is log time from the alert (the k-th event) to the block's last log line (`lead_time_seconds` in the report). In HDFS a block's last line is often a much later read or delete, so this is time before the block's last line, not time before anyone would notice a problem.

**Results on test blocks** (`data/evaluation/early_warning_report.json`):

| k | Alert quantile | Precision | Recall | F1 | False-alarm rate | Mean lead, true alerts | Expected lead per alert | Mean lead, seconds | Median lead, seconds |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.9698 | 0.1834 | 0.1966 | 0.1898 | 2.64% | 0.067 | 0.012 | 1,734 s | 0 s |
| 5 | 0.9874 | 0.8833 | 0.3671 | 0.5186 | 0.15% | 0.012 | 0.011 | 283 s | 0 s |
| 10 | 0.9867 | 0.8812 | 0.3979 | 0.5483 | 0.16% | 0.049 | 0.043 | 1,592 s | 0 s |
| 20 | 0.9801 | 0.8490 | 0.5722 | 0.6836 | 0.31% | 0.124 | 0.105 | 5,165 s (1.4 h) | 0 s |

**Results on blocks still running when the alert fires** (more than k events):

| k | Blocks | Anomalies | Precision | Recall | F1 | Mean lead | Mean lead, seconds | Median lead, seconds |
|---|---|---|---|---|---|---|---|---|
| 3 | 171,604 | 4,136 | 0.0189 | 0.0206 | 0.0197 | 0.779 | 20,258 s (5.6 h) | 8,908 s (2.5 h) |
| 5 | 170,693 | 3,225 | 0.1026 | 0.0087 | 0.0160 | 0.797 | 18,709 s (5.2 h) | 8,138 s (2.3 h) |
| 10 | 170,689 | 3,221 | 0.3991 | 0.0559 | 0.0980 | 0.551 | 17,776 s (4.9 h) | 8,062 s (2.2 h) |
| 20 | 42,339 | 2,009 | 0.6961 | 0.5266 | 0.5996 | 0.339 | 14,110 s (3.9 h) | 10,466 s (2.9 h) |

**Trade-off.** The Pareto front on precision versus mean lead time is k = 5, 10 and 20. The best trade-off is **k = 20**, on both views. At small k, the headline precision comes from short anomalous blocks that have already ended; only k = 20 warns early on blocks still in progress.

---

## 8. Real-Time Streaming Layer (Apache Kafka)

Every source streams: `producer.py` and `processor.py` take `--source`. Each source has its own topic, parser, models, vocabulary, storage policy and idle timeout. The timeout matches the block definition: Django users end only at end of stream, and Apache windows close after 120 s. A full Django replay matches batch exactly: 161 anomalous users and 0 late events. Syslog over UDP feeds any source's topic through `streaming/syslog_receiver.py`.

- **Why a Python stream processor and not Apache Flink (decision D3).** The original diagram had a Kafka/Flink box. It is now "Kafka + Python stream processor", for three reasons:
  1. The stream must score with the batch pipeline's own Python feature code and scikit-learn models. `tests/test_streaming.py` checks that streamed features equal batch features exactly. PyFlink would run that same Python inside a JVM cluster and add nothing to the result.
  2. One consumer process handled the 1,000,000-line verification replay on a laptop, and the demo VM has 8 GB.
  3. Kafka's keyed partitions already give per-block ordering, which is the property the per-block state needs.

  What Flink would add is checkpointed, exactly-once state that survives restarts. That is listed as future work.

- **Broker:** `docker-compose.yml` runs a single Kafka broker in KRaft mode (`apache/kafka`) with the topics `logs.raw`, `alerts` and `storage.decisions`.
- **Producer:** `streaming/producer.py` replays HDFS log lines at a configurable `--rate` and `--limit`, keyed by block id.
- **Processor:** `streaming/processor.py` (confluent-kafka) parses every line with `parser/hdfs_parser.py` and keeps per-block running state. Its features are tested to equal the batch builders exactly.
  - When a block reaches k events, it raises an `early_warning` alert if the early-warning model scores the first-k snapshot at or above the GA-tuned threshold. The default is k = 20, the best trade-off in §7.
  - When a block completes or times out, it computes the deployed model's full score (an Anomaly raises a `block_anomaly` alert) and the Smart Storage tier.
  - Alerts are published to `alerts` and saved to the MongoDB collection `stream_alerts`. Decisions are published to `storage.decisions`.
- **API:** `GET /api/stream/status` serves live counters (events/sec, blocks in flight, alert counts, storage saved so far) and `GET /api/stream/alerts` serves the alerts. The dashboard's Live Stream page polls both.
- **Setup:** `docs/streaming_guide.md` has the exact start-up commands.

**Verification replay (B3).** The first 1,000,000 HDFS lines (about 3.5 hours of log time) were replayed at full speed, run id `b3-1m`, while V2 was the live model. `scripts/analyze_stream_run.py` read every decision and alert back from Kafka and joined them with `anomaly_label.csv`. A block counts as **complete** only if all of its lines in the full `HDFS.log` fell inside the replay (`data/evaluation/stream_run_report.json`).

| Blocks (66,167 decided) | Blocks | Anomaly share | Block-anomaly precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|---:|
| Complete | 2,528 | 51.5% | **0.986** | **0.934** | **1.39%** |
| Cut off | 63,639 | 2.6% | 0.049 | 0.180 | 9.31% |

| Early warning (k = 20) | Blocks reaching 20 events | Alerts | True | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| Complete blocks | 88 | 22 | 21 | **0.955** | 0.247 | — (3 normal blocks) |
| Cut-off blocks | 1,560 | 142 | 131 | **0.923** | 0.784 | 0.79% |

- **Lead time on complete blocks:** a true early warning fired with a median 28.6% of the block still to come, a median 285 s (4.8 min) before the block's last line. For cut-off blocks the end is outside the replay; the median of 1,758 s to the last line *seen* is only a lower bound.
- **Complete blocks are mostly anomalous** (51.5% against 2.9% overall), because failing blocks end early. That inflates precision, so the false-alarm rate (1.39% of complete normal blocks; batch V2 is 0.93%) is the fairer number.
- **The whole-block verdict is unreliable on partial blocks.** That covers 6,071 of 7,303 block-anomaly alerts, nearly all of them false. Early warning does not have this problem: it only scores a block's first 20 events, which a partial block has.
- **The 1-hour idle timeout is too short for HDFS.** 34,321 of the 36,279 blocks finalized by timeout had more lines later, because HDFS blocks routinely go quiet for hours between write and read or delete. Those blocks look truncated, and timeout-finalized blocks give block-anomaly precision of 0.145 (929 of 6,396).

---

## 9. OpenStack as a Second Labeled Source

- **Data:** OpenStack nova logs from LogHub: two normal runs and one abnormal run.
- **Parser:** `parser/openstack_parser.py` implements `BaseParser` and outputs the same cleaned columns as the HDFS parser. `block_id` is the VM instance UUID; lines without one are dropped.
- **Labels:** each instance session is labeled by the file it came from: Anomaly if it appears in the abnormal run's file.
- **Models:** the common feature extractor, a baseline Isolation Forest and the GA-optimised model are trained under `data/models/openstack/` with their own vocabulary, versions and metrics. The HDFS models are untouched.
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

**Smart Storage on OpenStack.** On 1,035 held-out sessions (99 anomalies) it saves 22.27% of lines. Weighted, full retention and coverage are all 0.990, meeting the 0.95 target (fixed WARM rule; before it, 31.42% saved at weighted 0.929).

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

All routes are under `/api/`, and every data route takes `?source=hdfs|openstack|apache|cloudtrail|django`. `docs/api_reference.md` has the full list.

| Area | Routes |
|---|---|
| Auth | `POST /api/auth/register`, `/login`, `/change-password`; `GET /api/auth/me`, `/api/health` |
| Detection | `GET /api/dashboard`, `/detections`, `/detections/summary`, `/detections/export`, `/analytics` |
| Models | `GET /api/models`, `POST /api/models/retrain`, `GET /api/models/retrain/<job_id>` |
| Feedback | `POST`, `GET` and `DELETE /api/feedback` |
| Smart Storage | `GET /api/storage/summary`, `/policy`; `POST /api/storage/simulate` |
| Early warning | `GET /api/early-warning/summary` |
| Uploads | `POST /api/sources/upload`, `GET /api/sources/uploads`, `/uploads/<id>/detections`, `/connectors` |
| Stream | `GET /api/stream/status`, `/api/stream/alerts` |

---

## 12. Future Directions

### 12.1 Extra detector: Autoencoder vs Isolation Forest (done)

`scripts/compare_autoencoder.py` runs both detectors unsupervised on the same five stratified 70/30 splits of all 575,061 HDFS blocks (`data/evaluation/autoencoder_comparison.json`). The Autoencoder standardizes its features first. Both are scored at the deployed model's alert rate.

| Detector | Test F1 | ROC AUC | Fit time |
|---|---|---|---|
| **Isolation Forest (V3 settings)** | **0.555 ± 0.009** | **0.936** | 1.9 s |
| Autoencoder, V3's 2 features | 0.346 ± 0.189 | 0.718 | 5.2 s |
| Autoencoder, all 7 features | 0.285 ± 0.157 | 0.801 | 8.6 s |

Isolation Forest wins on every count: higher F1, about twenty times steadier across splits (±0.009 against ±0.189), and better at ranking anomalies (AUC 0.936 against 0.718). The Autoencoder on V3's two features reaches 0.49–0.52 on three splits but collapses to 0.09 and 0.14 on the other two; with all seven features it is worse still. Isolation Forest also needs no labels and no feature scaling, so it stays. (Against V2's settings, before V3 was re-promoted, the gap was smaller on average, 0.518 against 0.528, but the Autoencoder was just as unstable.)

### 12.2 Analyst feedback loop (done)

Analysts mark detections as real or false alarms (`/api/feedback`, Detection Results page). The next retrain of that source overrides the labels with these verdicts. It does this before drawing the GA's fitness sample and before scoring the candidate, and it re-scores the incumbent on the same corrected rows. `tests/test_feedback.py` proves that a marked alert changes the next retrain's fitness data.

### 12.3 Still open

1. **Streaming:** per-block state does not survive a restart. HDFS also has no "block finished" event, and the 1-hour idle timeout closes most blocks too early (§8).
2. **Multi-tenant isolation:** namespaces per cloud tenant.
3. **Cloud connectors:** CloudWatch, Azure Monitor and GCP Logging need provider credentials and pull agents. Syslog and Kafka work today.

---

## 13. References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE ICDM.
2. He, P., et al. (2020). *Loghub: A large collection of system log datasets towards automated log analytics*. IEEE SRDS.
3. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
4. Du, M., et al. (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. ACM CCS.
