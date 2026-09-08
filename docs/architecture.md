# MorphGuard — System Architecture Documentation

---

## 1. Overview

MorphGuard is an evolutionary real-time log analytics and anomaly detection system for cloud environments. It implements a layered SIEM architecture that continuously ingests raw distributed logs, extracts statistical and sequential features, detects behavioral anomalies via Isolation Forests, and evolves detection parameters using Genetic Algorithms.

---

## 2. High-Level Architecture Diagram

```
+-------------------------------------------------------------------------+
|                              Log Sources                                |
|           HDFS Clusters  |  Cloud Syslogs  |  Container Logs            |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        Ingestion & Parsing Layer                        |
|   +--------------------+  +---------------------+  +----------------+   |
|   |  BaseParser (ABC)  |  |   HDFSLogParser     |  |  DjangoParser  |   |
|   +--------------------+  +---------------------+  +----------------+   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       Feature Engineering Pipeline                      |
|   +-----------------------+  +--------------------------------------+   |
|   | DataCleaner (dedup)   |  | FeatureExtractor (event matrix)      |   |
|   +-----------------------+  +--------------------------------------+   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Detection & Inference                           |
|   +---------------------------------+  +----------------------------+   |
|   | IsolationForestDetector (scikit)|  | Decision Threshold Engine  |   |
|   +---------------------------------+  +----------------------------+   |
+-------------------+---------------------------------+-------------------+
                    |                                 |
                    v                                 v
+-------------------------------------+  +--------------------------------+
|      Genetic Algorithm Optimizer    |  |        REST API Service        |
|  +-------------------------------+  |  |  +--------------------------+  |
|  | Chromosome Evolution Pipeline |  |  |  | Flask App + JWT Auth    |  |
|  | (Selection, Crossover, Mutate)|  |  |  +--------------------------+  |
|  +-------------------------------+  |  |  | Endpoints: Ingest/Detect |  |
+-------------------------------------+  +--------------------------------+
                    |                                 |
                    +----------------+----------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                            Persistence Layer                            |
|        MongoDB / MongoDB Atlas  |  Model Storage (pickle/json)          |
+-------------------------------------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Ingestion & Parsing (`parser/`)
- `base_parser.py`: Abstract Base Class enforcing the `parse(filepath: str) -> pd.DataFrame` contract.
- `hdfs_parser.py`: High-throughput regex extraction for block IDs, timestamps, log levels, and event templates.
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
