# MorphGuard — API Reference

---

## Overview

The MorphGuard backend exposes a RESTful JSON API over HTTP. All endpoints are prefixed with `/api/` and return JSON responses. The API server runs on port 5000 by default.

**Base URL**: `http://localhost:5000/api`

---

## Authentication

### Overview

MorphGuard uses JWT (JSON Web Token) bearer authentication. Tokens are obtained via the login or register endpoints and must be included in the `Authorization` header of all protected requests.

**Header Format**:
```
Authorization: Bearer <jwt_token>
```

**Token Lifetime**: 24 hours from issuance.

**Token Payload Claims**:
| Claim  | Type   | Description                    |
|--------|--------|--------------------------------|
| sub    | string | User ID (MongoDB ObjectId)     |
| email  | string | User email address             |
| exp    | int    | Expiration timestamp (UTC)     |

---

## Endpoints

### Authentication Endpoints

#### POST /api/auth/register

Create a new user account and receive an authentication token.

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword123",
    "name": "John Doe"
}
```

**Validation Rules**:
- `email`: Must be a valid email format (contains @, no spaces)
- `password`: Minimum 8 characters
- `name`: Optional, trimmed of whitespace

**Success Response** (201 Created):
```json
{
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
        "email": "user@example.com",
        "name": "John Doe"
    }
}
```

**Error Responses**:

| Code | Condition            | Response                                          |
|------|----------------------|---------------------------------------------------|
| 400  | Invalid email        | `{"error": "Enter a valid email address."}`       |
| 400  | Password too short   | `{"error": "Password must be at least 8 characters."}` |
| 409  | Email already exists | `{"error": "An account with that email already exists."}` |
| 503  | Database unavailable | `{"error": "Database temporarily unavailable. Please try again."}` |

---

#### POST /api/auth/login

Authenticate with existing credentials and receive a token.

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword123"
}
```

**Success Response** (200 OK):
```json
{
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
        "email": "user@example.com",
        "name": "John Doe"
    }
}
```

**Error Responses**:

| Code | Condition              | Response                                          |
|------|------------------------|---------------------------------------------------|
| 401  | Invalid credentials    | `{"error": "Invalid email or password."}`         |
| 503  | Database unavailable   | `{"error": "Database temporarily unavailable. Please try again."}` |

---

#### GET /api/health

Health check endpoint. No authentication required.

**Success Response** (200 OK):
```json
{
    "status": "ok"
}
```

---

### Pipeline Endpoints

All pipeline endpoints require JWT authentication via the `Authorization: Bearer` header.

**Common Error Responses for Protected Endpoints**:

| Code | Condition           | Response                                          |
|------|---------------------|---------------------------------------------------|
| 401  | Missing token       | `{"error": "Authentication required."}`           |
| 401  | Expired token       | `{"error": "Session expired. Please sign in again."}` |
| 401  | Invalid token       | `{"error": "Invalid session. Please sign in again."}` |

---

#### GET /api/pipeline/detections

Retrieve recent anomaly detections from the detection pipeline.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "detections": [
        {
            "block_id": "blk_38865049064139660",
            "predicted_label": "Anomaly",
            "anomaly_score": 0.7842,
            "true_label": "Anomaly"
        }
    ],
    "total": 16838,
    "model_version": "v1"
}
```

---

#### GET /api/pipeline/analytics

Retrieve aggregated analytics and statistics from the detection pipeline.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "total_blocks": 575061,
    "total_anomalies": 16838,
    "anomaly_rate": 0.0293,
    "detection_summary": {
        "normal": 558223,
        "anomaly": 16838
    }
}
```

---

#### GET /api/pipeline/model/info

Retrieve information about the currently deployed model.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "version": 1,
    "model_path": "data/models/isolation_forest_v1.pkl",
    "metrics_path": "data/evaluation/isolation_forest_v1_metrics.json",
    "algorithm": "IsolationForest",
    "features_used": 21,
    "training_samples": 575061
}
```

---

#### GET /api/pipeline/model/metrics

Retrieve evaluation metrics for the current model.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "accuracy": 0.9714,
    "precision": 0.4523,
    "recall": 0.8901,
    "f1": 0.5997,
    "confusion_matrix": {
        "labels": ["Normal", "Anomaly"],
        "matrix": [[540000, 18223], [1853, 14985]]
    },
    "support": {
        "total": 575061,
        "normal": 558223,
        "anomaly": 16838
    }
}
```

---

#### GET /api/pipeline/model/evolution

Retrieve the model version history and evolution log.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "current_version": 2,
    "history": [
        {
            "version": 1,
            "action": "initial",
            "metric": "f1",
            "metric_value": 0.5997,
            "timestamp": "2026-09-07T10:00:00Z"
        },
        {
            "version": 2,
            "action": "promoted",
            "metric": "f1",
            "metric_value": 0.6823,
            "previous_value": 0.5997,
            "improvement": 0.0826,
            "timestamp": "2026-09-07T12:30:00Z"
        }
    ]
}
```

---

### Sources Endpoints

All sources endpoints require JWT authentication.

#### POST /api/sources/upload

Upload a log file for processing.

**Authentication**: Required

**Content-Type**: `multipart/form-data`

**Form Fields**:
| Field  | Type   | Required | Description |
|--------|--------|----------|-------------|
| file   | File   | Yes      | Log file to upload |
| source | string | No       | Log type: `hdfs` (default) or `openstack`. It picks the matching parser, vocabulary and deployed model. Other sources configured in `multi_source/` are also accepted. |

**Allowed File Extensions**: `.log`, `.json`, `.csv`, `.txt`, `.gz`

**Per-source scoring**:
- `hdfs` uploads are scored by the deployed HDFS model (`data/models/current_version.json`), and their new blocks extend the HDFS dashboard totals.
- `openstack` uploads are scored by `data/models/openstack/current_version.json`, with OpenStack's own vocabulary, and never touch the HDFS totals.
- Scoring refuses to run, and the upload is marked `failed`, if the source's category vocabulary file is missing or the model has no encoder stamp. Guessing the encoding in either case would give the same file a different anomaly count on a different server.

**Maximum File Size**: 500 MB

**Success Response** (201 Created):
```json
{
    "id": "64f1a2b3c4d5e6f7a8b9c0d1",
    "name": "hdfs_sample.log",
    "source": "Manual Upload",
    "size_bytes": 1048576,
    "records": null,
    "total_lines": null,
    "status": "processing",
    "error": null,
    "uploaded_at": "2026-09-07T10:30:00Z",
    "uploaded_by": "user@example.com"
}
```

**Note**: The response returns immediately with `status: "processing"`. The file is parsed asynchronously on a worker thread. Poll the uploads endpoint to check for status changes to `"processed"` or `"failed"`.

**Error Responses**:

| Code | Condition              | Response                                           |
|------|------------------------|-----------------------------------------------------|
| 400  | No file provided       | `{"error": "No file was provided."}`               |
| 400  | No file selected       | `{"error": "No file was selected."}`               |
| 400  | Invalid extension      | `{"error": "Unsupported file type. Allowed: .csv, .gz, .json, .log, .txt"}` |
| 400  | Unknown `source`       | `{"error": "Unknown source 'x'. Choose one of: hdfs, openstack, ..."}` |
| 400  | Source has no trained model | `{"error": "No trained model for source 'openstack' yet. Run: python3 scripts/run_source_pipeline.py --source openstack"}` |
| 413  | File too large         | (Flask returns 413 automatically)                  |

---

#### GET /api/sources/uploads

List upload history.

**Authentication**: Required

**Query Parameters**:
| Parameter | Type | Default | Max | Description            |
|-----------|------|---------|-----|------------------------|
| limit     | int  | 25      | 100 | Number of results      |

**Success Response** (200 OK):
```json
{
    "uploads": [
        {
            "id": "64f1a2b3c4d5e6f7a8b9c0d1",
            "name": "hdfs_sample.log",
            "source": "Manual Upload",
            "size_bytes": 1048576,
            "records": 500,
            "total_lines": 500,
            "status": "processed",
            "error": null,
            "log_source": "hdfs",
            "results_available": true,
            "blocks_analyzed": "<int>",
            "anomalies_detected": "<int>",
            "anomaly_rate_pct": "<float>",
            "model_version": "<int>",
            "scoring": {
                "model_version": "<int>",
                "model_path": "data/models/isolation_forest_v2.pkl",
                "model_encoder_fingerprint": "<16 hex chars>",
                "vocab_fingerprint": "<16 hex chars>"
            },
            "uploaded_at": "2026-09-07T10:30:00Z",
            "uploaded_by": "user@example.com"
        }
    ],
    "total": 1
}
```

**Anomaly counts come from this server's results file**, the same source the dashboard reads (`upload_store.py`), never from the number stored on the Mongo document.
- If the upload was scored by a different server that shares the MongoDB cluster, its results file isn't here. The row then has `results_available: false`, null counts, and a `results_note`, so it never shows a count that conflicts with the dashboard.
- `scoring` records which model and vocabulary produced the count.

**Upload Status Values**:
| Status      | Description                                        |
|-------------|-----------------------------------------------------|
| processing  | File is being parsed on a worker thread             |
| processed   | Parsing complete, records count available           |
| failed      | Parsing failed, error message available             |

---

#### GET /api/sources/connectors

List available data source connectors and their status.

**Authentication**: Required

**Success Response** (200 OK):
```json
{
    "connectors": [
        {"name": "Manual File Upload", "icon": "Database", "enabled": true, "accent": "primary"},
        {"name": "AWS CloudWatch", "icon": "Cloud", "enabled": false},
        {"name": "Azure Monitor", "icon": "Server", "enabled": false},
        {"name": "GCP Logging", "icon": "Waypoints", "enabled": false},
        {"name": "Kafka Stream", "icon": "ListTree", "enabled": false},
        {"name": "Syslog (RFC 5424)", "icon": "FileText", "enabled": false}
    ],
    "log_sources": [
        {"value": "hdfs", "label": "HDFS", "trained": true, "data_kind": "real", "default": true},
        {"value": "openstack", "label": "OpenStack (nova)", "trained": true, "data_kind": "real", "default": false}
    ]
}
```

`log_sources` lists the values the upload `source` field accepts, and whether each has a trained model on this server. The Log Sources page builds its source selector from this list.

---

### Smart Storage Endpoints

All require JWT authentication. They read `data/storage/policy_best.json` and `data/storage/storage_report.json`, written by `python3 scripts/optimize_storage_policy.py`, and return `404` with that hint until the script has been run. `stale: true` means the deployed model has changed since the policy was tuned.

#### GET /api/storage/summary

Held-out results, tier counts, baseline comparison and the learned policy. The GA history is trimmed to `ga_summary`.

**Success Response** (200 OK), abridged; values are from `data/storage/storage_report.json`:
```json
{
    "config": {"retention_target": 0.95, "warm_credit": 0.5, "penalty_weight": 1000.0, "error_weight": 0.2, "test_size": 0.5, "seed": 42},
    "split": {"tune": {"blocks": 287530, "anomalies": 8419}, "holdout": {"blocks": 287531, "anomalies": 8419, "lines": 5587093}},
    "policy": {
        "weights": {"anomaly_score": 0.7501, "event_frequency": 0.0731, "block_size": 0.1768},
        "error_weight": 0.2,
        "thresholds": {"hot": 0.1723, "warm": 0.0493},
        "summary_fraction": 0.0565
    },
    "holdout": {
        "storage_saved_pct": 68.76, "weighted_retention": 0.9676, "full_retention": 0.9356, "coverage": 0.9996,
        "kept_lines": 1745197, "total_lines": 5587093,
        "tier_counts": {"HOT": 62961, "WARM": 215399, "COLD": 9171},
        "meets_retention_target": true
    },
    "holdout_comparison": [{"name": "ga_policy", "clean": {"storage_saved_pct": 68.76, "weighted_retention": 0.9676}, "deployed": {"storage_saved_pct": 64.19, "weighted_retention": 0.9662}}],
    "ga_beats_baselines": true,
    "stale": false
}
```

#### GET /api/storage/policy

The learned policy alone: `policy`, `retention_target`, `model`, `generated_at`, `stale`.

#### POST /api/storage/simulate

Re-cost the policy with overridden parameters. All body fields are optional; omitted ones keep the learned value.

**Request Body**:
| Field | Type | Description |
|---|---|---|
| weights | object | Any of `anomaly_score`, `event_frequency`, `block_size` (non-negative, rescaled to sum to 1) |
| t_hot, t_warm | float | Thresholds, with `0 ≤ t_warm < t_hot ≤ 1` |
| summary_fraction | float | In (0, 1] |
| warm_credit | float | Most retention credit a WARM anomaly can earn, in [0, 1] |
| warm_ref_fraction | float | Summary share of a block that earns the full `warm_credit`, in (0, 1]; thinner summaries earn proportionally less (default 0.25) |
| retention_target | float | In [0, 1] |
| split | string | `holdout` (default), `tune` or `all` |

**Success Response** (200 OK). With no overrides it reproduces the held-out numbers above:
```json
{
    "split": "holdout",
    "outcome": {"storage_saved_pct": 68.76, "weighted_retention": 0.9676, "full_retention": 0.9356, "coverage": 0.9996,
                "tier_counts": {"HOT": 62961, "WARM": 215399, "COLD": 9171}},
    "meets_retention_target": true,
    "retention_target": 0.95,
    "warm_credit": 0.5,
    "stale": false
}
```

**Errors**: `400` for invalid input, for example `{"error": "thresholds must satisfy 0 <= t_warm < t_hot <= 1"}`.

---

### Early Warning Endpoint

#### GET /api/early-warning/summary

Held-out metrics per prefix length k, and the precision-vs-lead-time trade-off, from `data/evaluation/early_warning_report.json`. It is written by `python3 scripts/build_early_warning.py`; until then the endpoint returns `404`. JWT required.

**Success Response** (200 OK), abridged:
```json
{
    "table": [
        {"k": 5,  "precision": 0.8833, "recall": 0.3671, "f1": 0.5186, "false_alarm_rate": 0.0015, "mean_lead_over_true_alerts": 0.012},
        {"k": 20, "precision": 0.8490, "recall": 0.5722, "f1": 0.6836, "false_alarm_rate": 0.0031, "mean_lead_over_true_alerts": 0.124,
         "in_progress": {"blocks": 42339, "anomalies": 2009, "precision": 0.6961, "recall": 0.5266, "f1": 0.5996}}
    ],
    "trade_off": {"pareto_front": [5, 10, 20], "best_k": 20},
    "trade_off_in_progress": {"pareto_front": [5, 10, 20], "best_k": 20}
}
```
`in_progress` covers only blocks with more than k events, i.e. still running when the alert fires.

---

### Stream Endpoints

Both require JWT authentication and read what `python3 -m streaming.processor` writes to MongoDB. Setup is in `docs/streaming_guide.md`.

#### GET /api/stream/status

The newest processor run's live counters. Optional `run_id` selects a specific run. Returns `404` until a processor has reported, and `503` if MongoDB is unreachable.

**Success Response** (200 OK), with field types:
```json
{
    "run_id": "<string>",
    "live": "<bool: updated within 10 s and not stopped>",
    "seconds_since_update": "<float>",
    "events_per_sec": "<float: over the last 5 s>",
    "blocks_in_flight": "<int>",
    "alerts": {"early_warning": "<int>", "block_anomaly": "<int>", "total": "<int>"},
    "storage": {"blocks_decided": "<int>", "lines_seen": "<int>", "lines_kept": "<int>", "storage_saved_pct": "<float>",
                "tier_counts": {"HOT": "<int>", "WARM": "<int>", "COLD": "<int>"}},
    "finalize_reasons": {"timeout": "<int>", "end_of_stream": "<int>"},
    "log_time_watermark": "<ISO time of the newest log event>",
    "ks": ["<int>"]
}
```

#### GET /api/stream/alerts

Alerts from `stream_alerts`, newest first.

**Query Parameters**:
| Parameter | Default | Description |
|---|---|---|
| limit | 50 | 1–500 |
| page | 1 | |
| type | (all) | `early_warning` or `block_anomaly` |
| run_id | newest run | |

**Alert fields**:
- **Every alert:** `id`, `type`, `block_id`, `score`, `threshold`, `events_seen`, `event_time` (log time) and `detected_at`.
- **`early_warning`:** also `k`. Once its block finishes, it also gets `lead_fraction` (the share of the block's events still to come when the alert fired), `events_total` and `block_finalized`.
- **`block_anomaly`:** also `early_warning_k` and `finalize_reason`. `end_of_stream` means the block was forced out when the topic went quiet, possibly before it finished.

---

## Error Handling

### Global Error Format

All errors (including unhandled exceptions) are returned as JSON:

```json
{
    "error": "Description of what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning                | Usage                                      |
|------|------------------------|--------------------------------------------|
| 200  | OK                     | Successful GET/PUT requests                |
| 201  | Created                | Successful POST (register, upload)         |
| 400  | Bad Request            | Validation errors                          |
| 401  | Unauthorized           | Missing, expired, or invalid JWT token     |
| 404  | Not Found              | Endpoint does not exist                    |
| 409  | Conflict               | Duplicate resource (e.g., email exists)    |
| 413  | Payload Too Large      | Upload exceeds 500MB limit                 |
| 500  | Internal Server Error  | Unhandled server exceptions                |
| 503  | Service Unavailable    | Database connection failure                |

---

## CORS Configuration

The API is configured with the following CORS settings:

| Setting         | Value                        |
|-----------------|------------------------------|
| Allowed Origins | Configurable via CORS_ORIGIN |
| Allowed Headers | Content-Type, Authorization  |
| Default Origin  | http://localhost:3000         |

The Authorization header is explicitly allowed to prevent the browser preflight from blocking JWT bearer tokens.

---

## Rate Limiting

**Note**: The current implementation does not include rate limiting. For production deployments, consider adding rate limiting via:
- Flask-Limiter extension
- Nginx/Apache proxy-level rate limiting
- Cloud provider WAF (Web Application Firewall)

Recommended limits:
| Endpoint Group     | Suggested Limit        |
|--------------------|------------------------|
| /api/auth/*        | 10 requests/minute/IP  |
| /api/pipeline/*    | 60 requests/minute     |
| /api/sources/upload| 5 requests/minute      |

---

## Example Usage

### cURL Examples

**Register a new user**:
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "analyst@morphguard.io", "password": "SecurePass123", "name": "Security Analyst"}'
```

**Login**:
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "analyst@morphguard.io", "password": "SecurePass123"}'
```

**Get detections (authenticated)**:
```bash
TOKEN="eyJhbGci..."
curl -X GET http://localhost:5000/api/pipeline/detections \
  -H "Authorization: Bearer $TOKEN"
```

**Upload a log file**:
```bash
curl -X POST http://localhost:5000/api/sources/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/logfile.log"
```

**Check upload status**:
```bash
curl -X GET http://localhost:5000/api/sources/uploads \
  -H "Authorization: Bearer $TOKEN"
```

### Python Examples

```python
import requests

BASE_URL = "http://localhost:5000/api"

# Login
response = requests.post(f"{BASE_URL}/auth/login", json={
    "email": "analyst@morphguard.io",
    "password": "SecurePass123"
})
token = response.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

# Get model metrics
metrics = requests.get(f"{BASE_URL}/pipeline/model/metrics", headers=headers)
print(metrics.json())

# Upload a log file
with open("sample.log", "rb") as f:
    upload = requests.post(
        f"{BASE_URL}/sources/upload",
        headers=headers,
        files={"file": f}
    )
print(upload.json())
```

---

*This document is part of the MorphGuard project documentation.*
