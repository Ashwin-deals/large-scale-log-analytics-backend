# Large-Scale Log Analytics for Cloud Security

An adaptive, real-time anomaly detection framework for cloud environments combining **Isolation Forest** with **Genetic Algorithm (GA)** optimization for automated feature selection and dynamic threshold tuning.

## Project Overview

Modern cloud systems generate millions of streaming log events per second. Traditional rule-based monitors fail against evolving attack vectors, while standard unsupervised models suffer from high false-positive rates due to concept drift and high-dimensional noise.

This framework introduces:
1. **Cloud Security Log Ingestion & Preprocessing**: Generates and parses real-time audit logs (AWS CloudTrail / Linux auth style) covering legitimate user behavior and cyber attacks (brute-force, privilege escalation, data exfiltration).
2. **Evolutionary Optimization (Genetic Algorithm)**: Evolves candidate solutions $\theta = [f_1, f_2, \dots, f_n, T]$ to select optimal feature subsets and calibrate the anomaly score threshold.
3. **Adaptive Real-Time Anomaly Detection**: Evaluates incoming streaming logs using an optimized Isolation Forest and continuously tracks concept drift to trigger adaptive retraining.

## System Architecture

```
[ Cloud Log Stream ]
         │
         ▼
[ Log Ingestion & Preprocessor ] ──► Extracts Sliding Window & Statistical Features
         │
         ▼
[ Genetic Algorithm Optimizer ]  ──► Optimizes Feature Mask & Threshold θ = [f1..fn, T]
         │
         ▼
[ Isolation Forest Detector ]   ──► Computes s(x) = 2^(-E(h(x))/c(n))
         │
         ├──► Normal Traffic
         └──► Threat Alerts (Brute Force, Privilege Escalation, Exfiltration)
         │
         ▼
[ Concept Drift Handler ]        ──► Triggers Evolutionary Retraining on Drift
```

## Setup & Installation

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Execute the end-to-end detection simulation:

```bash
python run_pipeline.py
```

Run test suite:

```bash
python -m pytest tests/ -v
```
