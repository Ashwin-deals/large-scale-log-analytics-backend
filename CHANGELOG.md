# Changelog

All notable changes to the MorphGuard project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Real-time streaming pipeline via Apache Kafka
- Autoencoder-based anomaly detection
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Cloud connector implementations (AWS, Azure, GCP)

---

## [0.2.0] - 2026-09-07

### Added
- JWT authentication guard on all pipeline API routes (auth.py)
- Log Sources API with file upload, upload history, and connector status (sources_api.py)
- Shared MongoDB connection module (db.py) replacing per-module connections
- JSON error responses for all unhandled exceptions (maintains CORS headers)
- Hugging Face token support in download_dataset.py via load_dotenv()
- Authorization header in CORS allowed headers configuration
- data/uploads/ directory in .gitignore

### Changed
- Flask debug mode now defaults to off (opt in with FLASK_DEBUG=1)
- Error handler returns JSON instead of Werkzeug HTML debug pages

### Security
- All 7 pipeline routes now require Authorization: Bearer JWT
- JWT secret read per-call to avoid import-order pinning of fallback value
- File upload validation: extension allow-list, 500MB cap, filename sanitization
- Interactive Werkzeug debugger no longer exposed by default

---

## [0.1.0] - 2026-09-07

### Added
- Initial project structure and module organization
- HDFS log parser with regex-based field extraction and event type classification
- Django log parser for web framework access logs
- Abstract BaseParser interface for extensible parser architecture
- Feature engineering pipeline with block-level aggregation
- Event count, temporal, error, and distributional feature extraction
- Data cleaning utilities for NaN handling and type normalization
- Isolation Forest model configuration and factory
- Model training pipeline with unsupervised learning
- Prediction pipeline with anomaly scoring and label mapping
- Evaluation pipeline with accuracy, precision, recall, F1, and confusion matrix
- Genetic Algorithm optimization for joint feature selection and hyperparameter tuning
- Chromosome encoding with binary feature mask and continuous hyperparameter genes
- Stratified subsampling for stable GA fitness evaluation
- Model evolution system with version promotion and rejection logic
- Version history logging with full decision audit trail
- Flask REST API with pipeline and authentication endpoints
- User registration and login with bcrypt password hashing
- JWT token generation with 24-hour TTL
- MongoDB integration for user accounts
- Hugging Face dataset hosting and download script
- Dataset exploration and verification scripts
- Unit tests for parser, feature engineering, and model evolution
- MIT License
- README with project overview and setup instructions

---

## Version History Summary

| Version | Date       | Highlights                                        |
|---------|------------|---------------------------------------------------|
| 0.2.0   | 2026-09-07 | Auth guard, Sources API, JSON errors, security    |
| 0.1.0   | 2026-09-07 | Initial release with full ML pipeline             |

---

*This changelog is part of the MorphGuard project documentation.*
