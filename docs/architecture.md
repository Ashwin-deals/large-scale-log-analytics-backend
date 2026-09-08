# MorphGuard — System Architecture Documentation

---

## Overview

This document provides a detailed technical architecture overview of the MorphGuard log analytics system, including component diagrams, data flow descriptions, deployment topology, and design rationale for key architectural decisions.

---

## 1. Architecture Principles

The MorphGuard architecture is guided by the following design principles:

### 1.1 Modularity
Each functional component (parsing, feature engineering, detection, optimization) is implemented as an independent Python module with well-defined interfaces. This separation enables:
- Independent testing of each component
- Replacement of individual components without affecting others
- Parallel development across team members
- Clear ownership boundaries

### 1.2 Pipeline-Oriented Design
Data flows through the system in a linear pipeline, with each stage transforming its input and producing output consumed by the next stage. This design simplifies debugging (each stage can be inspected independently) and enables incremental processing.

### 1.3 Separation of Concerns
- **Data processing** (parsing, features, detection) is separated from **API serving** (Flask endpoints)
- **Authentication** is decoupled from business logic via decorators
- **Database access** is centralized in a single module (db.py)
- **Configuration** is externalized to environment variables

### 1.4 Reproducibility
Fixed random seeds, deterministic data joins, and model versioning ensure that results can be reproduced across runs and environments.

---

## 2. High-Level Architecture Diagram

```
                                    +------------------+
                                    |   Dashboard UI   |
                                    |  (Next.js SPA)   |
                                    +--------+---------+
                                             |
                                        HTTPS/REST
                                             |
                                    +--------v---------+
                                    |   Flask API       |
                                    |   (app.py)        |
                                    +---+-------+---+---+
                                        |       |   |
                            +-----------+   +---+   +----------+
                            |               |                  |
                    +-------v------+ +------v------+  +--------v--------+
                    | Auth Module  | | Pipeline API|  | Sources API     |
                    | (auth.py)    | | (pipeline_  |  | (sources_api.py)|
                    |              | |  api.py)    |  |                 |
                    +------+-------+ +------+------+  +--------+--------+
                           |                |                  |
                    +------v-------+        |          +-------v--------+
                    |   MongoDB    |        |          | File System    |
                    | (user_details|        |          | (data/uploads/)|
                    |  uploads)    |        |          +----------------+
                    +--------------+        |
                                            |
                    +-----------------------v-----------------------+
                    |              Data Processing Pipeline          |
                    |                                                |
                    |  +----------+  +-----------+  +------------+  |
                    |  | Parser   |->| Feature   |->| Detection  |  |
                    |  | Module   |  | Eng.      |  | Module     |  |
                    |  +----------+  +-----------+  +-----+------+  |
                    |                                      |         |
                    |                               +------v------+  |
                    |                               | Optimization|  |
                    |                               | Module      |  |
                    |                               +------+------+  |
                    |                                      |         |
                    |                               +------v------+  |
                    |                               | Model       |  |
                    |                               | Evolution   |  |
                    |                               +-------------+  |
                    +------------------------------------------------+
```

---

## 3. Component Details

### 3.1 Flask Application Layer

**Entry Point**: `app.py`

The Flask application serves as the HTTP entry point for all API requests. It is responsible for:

- Initializing the Flask application instance
- Configuring CORS with allowed origins and headers
- Registering API blueprints (pipeline_bp, sources_bp)
- Setting up MongoDB indexes (unique email constraint)
- Implementing authentication endpoints (register, login)
- Global error handling (converting exceptions to JSON responses)
- Health check endpoint

**Request Flow**:
```
HTTP Request
    |
    v
Flask Router (app.py)
    |
    +-- /api/auth/*     --> Auth handlers (app.py)
    |
    +-- /api/pipeline/* --> Pipeline Blueprint (pipeline_api.py)
    |                        |
    |                        +-- @token_required (auth.py)
    |
    +-- /api/sources/*  --> Sources Blueprint (sources_api.py)
    |                        |
    |                        +-- @token_required (auth.py)
    |
    +-- /api/health     --> Health check (app.py)
    |
    +-- * (errors)      --> JSON error handler (app.py)
```

### 3.2 Authentication Layer

**Module**: `auth.py`

The authentication layer provides JWT-based token verification that is shared across all protected blueprints.

**Design Decision — Why auth.py is Separate from app.py**:
The auth module lives outside app.py to prevent circular imports. pipeline_api.py needs to import the `token_required` decorator, but if that decorator lived in app.py, importing it would also import the Flask app — which in turn imports pipeline_api.py. The separate module breaks this cycle.

**Design Decision — Per-Call Secret Reading**:
```python
def _secret():
    return os.getenv("JWT_SECRET", "change-this-secret")
```

The JWT secret is read from the environment on each verification call rather than being cached at import time. This is necessary because Python's module import order means auth.py is imported (via pipeline_api.py) before app.py calls `load_dotenv()`. Reading at import time would permanently pin the fallback value.

### 3.3 Database Layer

**Module**: `db.py`

The database layer provides a single shared MongoDB connection used by all modules.

**Connection Configuration**:
```python
client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),       # TLS certificate bundle
    serverSelectionTimeoutMS=15000    # 15-second connection timeout
)
```

**Collections**:
| Collection     | Used By              | Purpose                    |
|----------------|----------------------|----------------------------|
| user_details   | app.py               | User accounts              |
| uploads        | sources_api.py       | Upload history and status  |

**Design Decision — certifi TLS Bundle**:
The `tlsCAFile=certifi.where()` parameter ensures MongoDB Atlas connections work on macOS systems where Python may not have access to system root certificates. Without this, connections to Atlas would fail with TLS verification errors.

**Design Decision — load_dotenv() in db.py**:
db.py calls `load_dotenv()` itself rather than relying on app.py, because app.py imports the blueprints (which import db.py) before it calls `load_dotenv()`. Without the call in db.py, environment variables would fall back to defaults silently.

### 3.4 Parser Module

**Directory**: `parser/`

**Architecture**: Strategy Pattern

The parser module uses the Strategy design pattern with an abstract base class (BaseParser) and concrete implementations for each log format.

```
BaseParser (ABC)
    |
    +-- HDFSParser
    |
    +-- DjangoParser
    |
    +-- [Future parsers]
```

**Extension Point**: To add support for a new log format:
1. Create a new parser class extending BaseParser
2. Implement the `parse()` method
3. Register the parser in the sources_api.py upload handler

### 3.5 Feature Engineering Module

**Directory**: `feature_engineering/`

The feature engineering module consists of two components:

**FeatureExtractor** (`feature_extractor.py`):
- Aggregates parsed log entries by block_id
- Computes event count, temporal, error, and distributional features
- Outputs a feature DataFrame with one row per block

**DataCleaner** (`data_cleaner.py`):
- Handles missing values and NaN replacement
- Ensures consistent data types
- Replaces infinite values
- Standardizes column ordering

**Feature Pipeline**:
```
Parsed DataFrame (per-line)
    |
    v
groupby(block_id)
    |
    v
Aggregate Features (event counts, temporal, error, distributional)
    |
    v
Data Cleaning (NaN handling, type casting)
    |
    v
Feature DataFrame (per-block)
```

### 3.6 Detection Module

**Directory**: `detection/`

The detection module implements a complete ML pipeline with four components:

**isolation_forest.py** — Model Factory:
- Defines TRAINING_FEATURE_COLUMNS and EXCLUDED_FEATURE_COLUMNS
- Provides `build_isolation_forest()` factory function
- Centralizes model configuration

**train.py** — Training Pipeline:
- Loads and merges features with labels
- Fits Isolation Forest on feature matrix
- Persists trained model with joblib

**predict.py** — Inference Pipeline:
- Generates predictions from trained model
- Computes anomaly scores
- Maps raw predictions to human-readable labels
- Saves prediction output to CSV

**evaluate.py** — Evaluation Pipeline:
- Computes accuracy, precision, recall, F1
- Generates confusion matrix
- Persists metrics to JSON
- Provides formatted console output

### 3.7 Optimization Module

**Directory**: `optimization/`

**ga_isolation_forest.py** — Genetic Algorithm Engine:

The GA engine uses PyGAD to evolve a population of candidate model configurations.

**Chromosome Structure**:
```
[feature_mask_1, ..., feature_mask_N, n_estimators, max_samples, max_features, contamination]
|<--- N binary genes --->|              |<--- 4 hyperparameter genes --->|
```

**Evolution Loop**:
```
Initialize Population (random chromosomes)
    |
    v
+-- Evaluate Fitness (F1 score for each chromosome) <--+
|       |                                                |
|       v                                                |
|   Select Parents (Steady-State Selection)              |
|       |                                                |
|       v                                                |
|   Crossover (Single Point)                             |
|       |                                                |
|       v                                                |
|   Mutation (Random, 20% of genes)                      |
|       |                                                |
|       v                                                |
|   Elitism (preserve best individual)                   |
|       |                                                |
+-------+-- Repeat for num_generations -----------------+
        |
        v
    Best Chromosome --> Candidate Model
```

**model_evolution.py** — Model Versioning:

```
Candidate Model (from GA)
    |
    v
Compare F1 vs. Current Production Model
    |
    +-- Better? --> Promote to v(N+1), update current_version.json
    |
    +-- Worse?  --> Archive to rejected/, log decision
    |
    v
Append to version_history.json
```

---

## 4. Data Storage Layout

```
data/
|-- raw/
|   +-- hdfs/
|       |-- HDFS.log              # Raw log file (~1.5GB)
|       +-- anomaly_label.csv    # Ground truth labels
|
|-- features/
|   +-- features.csv             # Extracted block-level features
|
|-- models/
|   |-- isolation_forest_v1.pkl  # Baseline model
|   |-- isolation_forest_v2.pkl  # GA-optimized model (if promoted)
|   +-- ...
|
|-- evaluation/
|   |-- baseline_isolation_forest_metrics.json
|   |-- isolation_forest_v1_metrics.json
|   +-- ...
|
|-- predictions/
|   |-- isolation_forest_v1_predictions.csv
|   +-- ...
|
|-- evolution/
|   |-- current_version.json     # Active model pointer
|   |-- version_history.json     # All promotion decisions
|   +-- archive/                 # Rejected candidate models
|
+-- uploads/                     # User-uploaded log files
```

---

## 5. API Architecture

### 5.1 Blueprint Registration

```python
# app.py
app.register_blueprint(pipeline_bp)    # /api/pipeline/*
app.register_blueprint(sources_bp)     # /api/sources/*
```

### 5.2 Authentication Flow

```
Client                          Server
  |                               |
  |  POST /api/auth/login         |
  |  { email, password }          |
  |------------------------------>|
  |                               |  Verify credentials
  |                               |  Generate JWT (24h TTL)
  |  { token, user }              |
  |<------------------------------|
  |                               |
  |  GET /api/pipeline/detections |
  |  Authorization: Bearer <jwt>  |
  |------------------------------>|
  |                               |  @token_required
  |                               |  Decode JWT
  |                               |  Set g.user
  |  { detections: [...] }        |
  |<------------------------------|
```

### 5.3 Upload Processing Architecture

```
Client                    Flask Thread              Worker Thread
  |                           |                          |
  |  POST /api/sources/upload |                          |
  |  (multipart file)         |                          |
  |-------------------------->|                          |
  |                           |  Validate file           |
  |                           |  Save to disk            |
  |                           |  Insert MongoDB record   |
  |                           |  (status: "processing")  |
  |                           |                          |
  |                           |  spawn daemon thread --->|
  |                           |                          |  Parse with HDFSParser
  |  { upload: {...} }        |                          |  Count recognized lines
  |<--------------------------|                          |  Update MongoDB record
  |                           |                          |  (status: "processed"
  |                           |                          |   or "failed")
```

---

## 6. Error Handling Architecture

### 6.1 Global Error Handler

All unhandled exceptions are caught by the global error handler in app.py:

```python
@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
```

**Why JSON instead of HTML**: Werkzeug's default HTML error page is emitted before Flask-CORS can attach CORS headers. The browser then sees a CORS failure ("Failed to fetch") instead of the real error. Returning JSON ensures CORS headers are present and the dashboard can display the actual error message.

### 6.2 Authentication Error Codes

| HTTP Code | Condition                    | Message                                |
|-----------|------------------------------|----------------------------------------|
| 400       | Invalid email format         | "Enter a valid email address."         |
| 400       | Password too short           | "Password must be at least 8 chars."   |
| 401       | Missing/invalid token        | "Authentication required."             |
| 401       | Expired token                | "Session expired. Please sign in."     |
| 401       | Wrong credentials            | "Invalid email or password."           |
| 409       | Duplicate email              | "Account already exists."              |
| 503       | Database unavailable         | "Database temporarily unavailable."    |

---

## 7. Deployment Considerations

### 7.1 Environment Variables

| Variable         | Default                  | Description                            |
|-----------------|--------------------------|----------------------------------------|
| MONGO_URI       | mongodb://localhost:27017| MongoDB connection string              |
| MONGO_DB        | users                    | Database name                          |
| MONGO_COLLECTION| user_details             | Collection for user accounts           |
| JWT_SECRET      | change-this-secret       | Secret for JWT signing                 |
| CORS_ORIGIN     | http://localhost:3000    | Allowed CORS origin                    |
| FLASK_DEBUG     | (off)                    | Enable debug mode (1/true/yes)         |
| HF_TOKEN        | (empty)                  | Hugging Face token for gated datasets  |

### 7.2 Port Configuration

The Flask development server runs on port 5000 by default. For production deployments, use a WSGI server such as Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 7.3 MongoDB Atlas Setup

For cloud deployments, configure MONGO_URI with the Atlas connection string:

```
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

The certifi TLS bundle in db.py ensures connections work without system-level certificate configuration.

---

## 8. Scalability Considerations

### 8.1 Current Bottlenecks

1. **Single-Process Flask**: The development server handles one request at a time per worker
2. **Local File Storage**: Uploaded files are stored on the local filesystem
3. **In-Process ML**: Feature engineering and model training run in the same process
4. **Thread-Based Upload Processing**: Upload parsing uses daemon threads with no task queue

### 8.2 Scaling Strategies

1. **Horizontal API Scaling**: Deploy behind a load balancer with multiple Gunicorn workers
2. **Object Storage**: Replace local file storage with S3/GCS for uploaded logs
3. **Task Queue**: Replace daemon threads with Celery + Redis/RabbitMQ for upload processing
4. **Distributed ML**: Use Spark or Dask for feature engineering on large datasets
5. **Model Serving**: Separate model inference into a dedicated service (e.g., TorchServe, MLflow)

---

*This document is part of the MorphGuard project documentation.*
