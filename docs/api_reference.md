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
| Field | Type | Required | Description       |
|-------|------|----------|-------------------|
| file  | File | Yes      | Log file to upload |

**Allowed File Extensions**: `.log`, `.json`, `.csv`, `.txt`

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
| 400  | Invalid extension      | `{"error": "Unsupported file type. Allowed: .csv, .json, .log, .txt"}` |
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
            "uploaded_at": "2026-09-07T10:30:00Z",
            "uploaded_by": "user@example.com"
        }
    ],
    "total": 1
}
```

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
    ]
}
```

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
