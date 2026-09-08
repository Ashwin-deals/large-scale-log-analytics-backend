# Changelog

All notable changes to MorphGuard are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) | Versioning: [SemVer](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Planned
- Real-time streaming pipeline via Apache Kafka
- Autoencoder-based anomaly detection
- Docker containerization and docker-compose setup
- CI/CD pipeline with GitHub Actions
- Cloud connector implementations (AWS CloudWatch, Azure Monitor, GCP Logging)
- Rate limiting on authentication endpoints
- Celery task queue for async upload processing

---

## [0.2.0] - 2026-09-07

### Added
- JWT authentication guard on all pipeline API routes via `@token_required` decorator
- Log Sources API: file upload, upload history, connector status (`sources_api.py`)
- Shared MongoDB connection module (`db.py`)
- JSON error responses for all unhandled exceptions
- HF token support in `download_dataset.py`
- `Authorization` header in CORS allowed headers
- `data/uploads/` to `.gitignore`

### Security
- All 7 pipeline routes now require `Authorization: Bearer` JWT
- JWT secret read per-call to avoid import-order fallback pinning
- File upload validation: extension allow-list, 500MB cap, filename sanitization
- Werkzeug interactive debugger disabled by default

---

## [0.1.0] - 2026-09-07

### Added
- HDFS log parser with regex field extraction and event type classification
- Django log parser for web framework access logs
- Abstract `BaseParser` interface for extensible parsing
- Feature engineering: block-level aggregation with event, temporal, error, and distributional features
- Isolation Forest anomaly detection (unsupervised)
- Model training, prediction, and evaluation pipelines
- Genetic Algorithm optimization for joint feature selection and hyperparameter tuning
- Model evolution system with version promotion/rejection and audit trail
- Flask REST API with authentication (JWT + bcrypt)
- MongoDB integration for user accounts
- Hugging Face dataset hosting and download script
- Unit tests for parser, feature engineering, and model evolution
- MIT License

---

*Maintained as part of the MorphGuard project.*
