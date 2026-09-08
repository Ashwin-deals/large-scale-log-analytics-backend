# MorphGuard — Setup and Installation Guide

---

## Prerequisites

Before setting up MorphGuard, ensure you have the following installed:

| Requirement       | Minimum Version | Check Command          |
|-------------------|-----------------|------------------------|
| Python            | 3.10+           | `python3 --version`    |
| pip               | 21.0+           | `pip3 --version`       |
| Git               | 2.30+           | `git --version`        |
| MongoDB           | 6.0+ (or Atlas) | `mongod --version`     |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ashwin-deals/large-scale-log-analytics-backend.git
cd large-scale-log-analytics-backend
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# or
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# MongoDB connection (local or Atlas)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=users
MONGO_COLLECTION=user_details

# JWT authentication secret (CHANGE THIS in production!)
JWT_SECRET=your-secure-random-secret-here

# Dashboard CORS origin
CORS_ORIGIN=http://localhost:3000

# Hugging Face token (only needed for gated datasets)
HF_TOKEN=your-huggingface-token
```

### 5. Download the Dataset

```bash
python3 scripts/download_dataset.py
```

This downloads the HDFS log (~1.5GB) and anomaly labels from Hugging Face into `data/raw/hdfs/`.

### 6. Start the API Server

```bash
python3 app.py
```

The server starts on `http://localhost:5000`.

---

## Full Pipeline Setup

### Step 1: Explore the Raw Data

```bash
python3 scripts/exlpore_hdfs.py
```

This script provides an overview of the raw HDFS log data including line counts, format samples, and basic statistics.

### Step 2: Build Block Metadata

```bash
python3 scripts/build_block_metadata.py
```

Extracts block-level metadata from the raw log file.

### Step 3: Build Features

```bash
python3 scripts/build_hdfs_features.py
```

Parses the raw HDFS log, extracts structured fields, and computes block-level features. Output is saved to `data/features/features.csv`.

### Step 4: Verify Features and Labels

```bash
python3 scripts/verify_features_labels.py
```

Validates the integrity of the feature-label join:
- Checks for row loss during the merge
- Verifies no duplicate block IDs exist
- Confirms label distribution matches expectations

### Step 5: Train the Baseline Model

```bash
python3 scripts/train_baseline_isolation_forest.py
```

Trains the V1 Isolation Forest model with default parameters and saves:
- Model: `data/models/isolation_forest_v1.pkl`
- Metrics: `data/evaluation/baseline_isolation_forest_metrics.json`
- Predictions: `data/predictions/isolation_forest_v1_predictions.csv`

### Step 6: Run Genetic Algorithm Optimization

```bash
python3 scripts/optimize_isolation_forest_ga.py
```

Runs the GA to find an optimized feature subset and hyperparameter configuration. This step can take significant time depending on population size and number of generations.

### Step 7: Retrain and Evaluate (Model Evolution)

```bash
python3 scripts/retrain_and_evaluate.py
```

Retrains the model with GA-optimized parameters and evaluates whether the candidate should be promoted to the next version.

---

## MongoDB Setup

### Option A: Local MongoDB

1. Install MongoDB Community Edition:
   - macOS: `brew tap mongodb/brew && brew install mongodb-community`
   - Ubuntu: Follow the [official guide](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/)
   - Windows: Download from [mongodb.com](https://www.mongodb.com/try/download/community)

2. Start the MongoDB service:
   ```bash
   brew services start mongodb-community    # macOS
   sudo systemctl start mongod             # Ubuntu
   ```

3. Set in `.env`:
   ```
   MONGO_URI=mongodb://localhost:27017
   ```

### Option B: MongoDB Atlas (Cloud)

1. Create a free account at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create a cluster (free tier M0 is sufficient for development)
3. Create a database user with read/write permissions
4. Get the connection string from the Atlas dashboard
5. Set in `.env`:
   ```
   MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
   ```

**Note**: The `certifi` TLS bundle in `db.py` ensures Atlas connections work on macOS without additional certificate configuration.

---

## Hugging Face Dataset Access

The HDFS dataset is hosted on Hugging Face. If the dataset is gated (requires approval):

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Request access to the dataset
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set in `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

---

## Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test modules
python3 -m pytest tests/test_hdfs_parser.py -v
python3 -m pytest tests/test_feature_engineering.py -v
python3 -m pytest tests/test_model_evolution.py -v
```

---

## Development Server

### Debug Mode

To enable Flask debug mode with auto-reload:

```bash
FLASK_DEBUG=1 python3 app.py
```

**Warning**: Do not use debug mode in production. The Werkzeug debugger exposes an interactive Python console.

### Custom Port

To run on a different port, modify `app.py` or set the environment variable:

```bash
FLASK_RUN_PORT=8080 python3 app.py
```

---

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker (Future)

A Dockerfile is planned for future releases. The recommended base image is `python:3.11-slim`.

### Environment Checklist for Production

- [ ] Set a strong, random `JWT_SECRET` (at least 32 characters)
- [ ] Configure `CORS_ORIGIN` to your dashboard domain
- [ ] Use MongoDB Atlas or a properly secured MongoDB instance
- [ ] Disable `FLASK_DEBUG` (default is off)
- [ ] Use HTTPS via a reverse proxy (nginx, Cloudflare, etc.)
- [ ] Set up log rotation for application logs
- [ ] Configure monitoring and alerting

---

## Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
# Ensure you are in the project root and venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

**MongoDB connection failures**:
```bash
# Check if MongoDB is running
mongod --version
brew services list | grep mongodb   # macOS
sudo systemctl status mongod        # Ubuntu

# Test connection
python3 -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017').server_info())"
```

**CORS errors in the browser**:
- Verify `CORS_ORIGIN` matches your dashboard URL exactly
- Check that `Authorization` is in the allowed headers
- Ensure the error handler returns JSON (not HTML)

**JWT token issues**:
- Tokens expire after 24 hours — re-authenticate to get a new token
- Verify `JWT_SECRET` is the same across server restarts
- Check that `load_dotenv()` is loading the correct `.env` file

**Dataset download failures**:
- Check your internet connection
- If the dataset is gated, ensure `HF_TOKEN` is set correctly
- Try downloading manually from the Hugging Face page

---

## Directory Structure After Setup

```
large-scale-log-analytics-backend/
|-- app.py
|-- auth.py
|-- db.py
|-- pipeline_api.py
|-- sources_api.py
|-- requirements.txt
|-- .env                    # Your configuration (git-ignored)
|-- .env.example            # Template
|-- data/
|   |-- raw/hdfs/
|   |   |-- HDFS.log       # Downloaded dataset (~1.5GB)
|   |   +-- anomaly_label.csv
|   |-- features/
|   |   +-- features.csv   # After running build_hdfs_features.py
|   |-- models/
|   |   +-- isolation_forest_v1.pkl  # After training
|   |-- evaluation/
|   |   +-- baseline_isolation_forest_metrics.json
|   +-- predictions/
|       +-- isolation_forest_v1_predictions.csv
|-- parser/
|-- feature_engineering/
|-- detection/
|-- optimization/
|-- scripts/
|-- tests/
+-- docs/
```

---

*This document is part of the MorphGuard project documentation.*
