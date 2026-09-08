# MorphGuard — Testing Guide

---

## Overview

This guide describes the testing strategy, test structure, and instructions for running tests in the MorphGuard project.

---

## Test Structure

```
tests/
|-- test_hdfs_parser.py           # Parser module unit tests
|-- test_feature_engineering.py   # Feature extraction unit tests
+-- test_model_evolution.py       # Model evolution unit tests
```

---

## Running Tests

### Run All Tests

```bash
python3 -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Parser tests
python3 -m pytest tests/test_hdfs_parser.py -v

# Feature engineering tests
python3 -m pytest tests/test_feature_engineering.py -v

# Model evolution tests
python3 -m pytest tests/test_model_evolution.py -v
```

### Run with Coverage Report

```bash
pip install pytest-cov
python3 -m pytest tests/ --cov=parser --cov=feature_engineering --cov=optimization --cov-report=term-missing
```

---

## Test Categories

### 1. Parser Tests (`test_hdfs_parser.py`)

Tests the HDFS log parser for correctness of field extraction and event type classification.

**What is tested**:
- Correct extraction of date, time, PID, level, component from log lines
- Block ID extraction from various log formats
- Event type classification (RECEIVING_BLOCK, RECEIVED_BLOCK, etc.)
- Handling of malformed or incomplete log lines
- Correct DataFrame column structure
- Multi-line log file parsing
- Edge cases: empty files, files with no matching lines

**Example test cases**:

| Test Case                        | Description                                      |
|---------------------------------|--------------------------------------------------|
| test_parse_single_line          | Parse one valid HDFS log line                    |
| test_parse_multiple_lines       | Parse a file with multiple lines                 |
| test_event_type_classification  | Verify correct event type assignment             |
| test_block_id_extraction        | Verify block_id is correctly extracted           |
| test_malformed_line             | Handle lines that do not match the HDFS pattern  |
| test_empty_file                 | Gracefully handle an empty input file            |

### 2. Feature Engineering Tests (`test_feature_engineering.py`)

Tests the feature extraction pipeline for correctness of block-level aggregation and feature computation.

**What is tested**:
- Correct aggregation by block_id
- Event count features (n_events, n_receiving_block, etc.)
- Temporal features (session_duration, mean_time_gap, etc.)
- Error features (n_error, error_ratio, is_error)
- Distributional features (event_entropy, n_unique_event_types)
- Feature column completeness (all expected columns present)
- NaN handling in edge cases
- Single-event blocks (degenerate temporal features)

**Example test cases**:

| Test Case                        | Description                                      |
|---------------------------------|--------------------------------------------------|
| test_feature_columns_present    | All expected feature columns exist               |
| test_event_count_aggregation    | Event counts match expected values               |
| test_temporal_features          | Duration and gap statistics are correct          |
| test_error_features             | Error counts and ratios are accurate             |
| test_entropy_computation        | Shannon entropy is correctly computed            |
| test_single_event_block         | Handle blocks with only one event                |

### 3. Model Evolution Tests (`test_model_evolution.py`)

Tests the model versioning and promotion logic.

**What is tested**:
- Promotion decision when candidate F1 > current F1
- Rejection decision when candidate F1 <= current F1
- Version number incrementing
- Current version file creation and updates
- Version history logging
- Archive file creation for rejected candidates
- Edge cases: first model (no existing version), equal metrics

**Example test cases**:

| Test Case                        | Description                                      |
|---------------------------------|--------------------------------------------------|
| test_promote_better_model       | Candidate with higher F1 is promoted             |
| test_reject_worse_model         | Candidate with lower F1 is archived              |
| test_version_increment          | Version number increases by 1 on promotion       |
| test_version_history_append     | Decision is appended to history log              |
| test_first_model_promotion      | First model is promoted as v1                    |
| test_equal_metrics_rejection    | Equal metrics result in rejection (not promotion)|

---

## Writing New Tests

### Conventions

1. **File naming**: Test files should be named `test_<module_name>.py`
2. **Function naming**: Test functions should be named `test_<description>`
3. **Fixtures**: Use pytest fixtures for shared test data
4. **Assertions**: Use plain `assert` statements (pytest provides clear failure messages)
5. **Temporary files**: Use `tmp_path` fixture for temporary file creation

### Example Test Template

```python
"""Tests for <module_name>."""

import pytest
import pandas as pd


class TestClassName:
    """Tests for ClassName functionality."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample test data."""
        data = pd.DataFrame({
            "block_id": ["blk_001", "blk_001", "blk_002"],
            "event_type": ["RECEIVED_BLOCK", "SERVING_BLOCK", "DELETION"],
            "level": ["INFO", "INFO", "WARN"],
        })
        filepath = tmp_path / "test_data.csv"
        data.to_csv(filepath, index=False)
        return filepath

    def test_basic_functionality(self, sample_data):
        """Verify basic functionality works correctly."""
        # Arrange
        expected_result = ...

        # Act
        actual_result = function_under_test(sample_data)

        # Assert
        assert actual_result == expected_result

    def test_edge_case(self):
        """Verify edge case is handled correctly."""
        pass
```

---

## Test Data

### Sample Test Data Location

Test data is generated within test fixtures using `tmp_path` rather than relying on external files. This ensures tests are:
- **Self-contained**: No dependency on external data files
- **Reproducible**: Same data every run
- **Fast**: No I/O overhead from large files

### Generating Test Data

For larger integration tests, use the sample data script:

```bash
python3 scripts/sample_test_data.py
```

This creates a small subset of the full dataset suitable for testing without requiring the full 1.5GB download.

---

## Continuous Integration

### Recommended CI Configuration

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short
```

---

## Coverage Goals

| Module                | Current Coverage | Target Coverage |
|-----------------------|-----------------|-----------------|
| parser/               | Moderate        | 90%+            |
| feature_engineering/  | Moderate        | 85%+            |
| detection/            | Low             | 80%+            |
| optimization/         | Low             | 75%+            |
| API endpoints         | None            | 70%+            |

---

## Known Test Gaps

The following areas currently lack test coverage and are candidates for future test development:

1. **API endpoint tests**: No integration tests for Flask routes
2. **Authentication flow**: No tests for register/login/token validation
3. **Upload processing**: No tests for the async upload parsing pipeline
4. **GA convergence**: No tests verifying GA improves fitness over generations
5. **Data cleaning**: Limited tests for edge cases in DataCleaner
6. **Django parser**: No tests for the Django log parser

---

*This document is part of the MorphGuard project documentation.*
