# MorphGuard — Setup and Installation Guide

## Prerequisites

Before setting up MorphGuard, ensure you have the following installed:

| Requirement       | Minimum Version | Check Command          |
|-------------------|-----------------|------------------------|
| Python            | 3.10+           | `python3 --version`    |
| pip               | 21.0+           | `pip3 --version`       |
| Git               | 2.30+           | `git --version`        |
| MongoDB           | 6.0+ (or Atlas) | `mongod --version`     |


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

### Step 5: Train Baseline & Optimize Models

```bash
python3 scripts/train_baseline_isolation_forest.py
python3 scripts/optimize_isolation_forest_ga.py
python3 scripts/retrain_and_evaluate.py
```
