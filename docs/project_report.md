# MorphGuard — Large-Scale Log Analytics for Cloud Security

## Project Report

---

### Executive Overview

| Attribute | Details |
|---|---|
| **System** | MorphGuard Anomaly Detection Platform |
| **Target Logs** | HDFS & Cloud Distributed Systems |
| **Core Detection** | Isolation Forest with Genetic Algorithm Optimization |
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

## 5. Security & Threat Mitigation

MorphGuard adheres to defense-in-depth principles:
- **Authentication**: Stateless JWT with HMAC-SHA256 signatures and token expiry.
- **Password Protection**: bcrypt key derivation with salted cost factor.
- **Transport Security**: TLS 1.3 encryption across Atlas cluster connections.
- **Input Sanitization**: Path normalization and regex validation on all ingestion routes.

---

## 6. API Reference Summary

Key endpoints provided by the Flask application:
- `POST /api/v1/auth/login`: Issue bearer token.
- `POST /api/v1/ingest`: Stream new log entries into feature store.
- `POST /api/v1/detect`: Run real-time inference on feature vectors.
- `POST /api/v1/optimize/run`: Trigger asynchronous GA generation cycle.
- `GET /api/v1/models/history`: Fetch version lineage and evaluation metrics.

---

## 7. Future Directions

Planned enhancements include:
1. **Streaming Pipeline**: Native Apache Kafka integration for sub-second ingestion.
2. **Deep Learning Detectors**: Autoencoder and Graph Neural Network (GNN) modules.
3. **Multi-Tenant Isolation**: Namespace partitioning for enterprise cloud tenants.
4. **Automated Rollback**: Canary verification with automatic model rollback.

---

## 8. References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE ICDM.
2. He, P., et al. (2020). *Loghub: A large collection of system log datasets towards automated log analytics*. IEEE SRDS.
3. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
4. Du, M., et al. (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. ACM CCS.
