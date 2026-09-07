# Contributing to MorphGuard

Thank you for your interest in contributing to MorphGuard! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Commit Messages](#commit-messages)
7. [Pull Request Process](#pull-request-process)
8. [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/large-scale-log-analytics-backend.git
   cd large-scale-log-analytics-backend
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/Ashwin-deals/large-scale-log-analytics-backend.git
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## How to Contribute

### Types of Contributions

We welcome the following types of contributions:

- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new functionality to the system
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **New Parsers**: Add support for new log formats
- **New Detection Algorithms**: Implement alternative anomaly detection methods

### Areas Looking for Help

- Additional log format parsers (Apache, Nginx, Kubernetes, etc.)
- Autoencoder-based anomaly detection implementation
- Real-time streaming pipeline (Kafka integration)
- API endpoint test coverage
- CI/CD pipeline configuration
- Docker containerization
- Cloud connector implementations (AWS CloudWatch, Azure Monitor, etc.)

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- MongoDB 6.0+ (local or Atlas)
- Git

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your local configuration

# Run tests to verify setup
python3 -m pytest tests/ -v
```

### Running the Development Server

```bash
FLASK_DEBUG=1 python3 app.py
```

---

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 120 characters
- Use docstrings for all public classes and functions

### Documentation Style

- Use Google-style docstrings
- Include parameter descriptions and return value documentation
- Add inline comments for non-obvious logic

### Example

```python
def process_log_entry(
    line: str,
    parser: BaseParser,
    strict: bool = False,
) -> dict | None:
    """Parse a single log entry and return structured data.

    Args:
        line: Raw log line text.
        parser: Parser instance to use for extraction.
        strict: If True, raise on unparseable lines instead of returning None.

    Returns:
        Dictionary of extracted fields, or None if the line could not be parsed
        and strict is False.

    Raises:
        ParseError: If strict is True and the line cannot be parsed.
    """
    ...
```

---

## Commit Messages

### Format

Follow the Conventional Commits specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type     | Description                                |
|----------|---------------------------------------------|
| feat     | A new feature                               |
| fix      | A bug fix                                   |
| docs     | Documentation only changes                  |
| style    | Formatting, missing semicolons, etc.        |
| refactor | Code change that neither fixes nor adds     |
| perf     | Performance improvement                     |
| test     | Adding or updating tests                    |
| chore    | Build process or auxiliary tool changes     |

### Examples

```
feat(parser): add Apache access log parser

fix(auth): read JWT secret per-call to avoid import-order issue

docs(api): add API reference documentation

test(detection): add unit tests for evaluation metrics

refactor(db): consolidate MongoDB connections into single module
```

---

## Pull Request Process

### Before Submitting

1. Ensure your code follows the coding standards
2. Run the test suite and verify all tests pass:
   ```bash
   python3 -m pytest tests/ -v
   ```
3. Update documentation if your changes affect:
   - API endpoints
   - Configuration options
   - Module interfaces
   - Data flow
4. Add or update tests for your changes

### PR Description Template

Your pull request description should include:

1. **What**: A clear description of the changes
2. **Why**: The motivation or problem being solved
3. **How**: Brief overview of the implementation approach
4. **Testing**: How you verified the changes work correctly
5. **Breaking Changes**: Any backward-incompatible changes

### Review Process

1. Submit your PR against the `main` branch
2. A maintainer will review your code within 48 hours
3. Address any requested changes
4. Once approved, a maintainer will merge your PR

---

## Issue Guidelines

### Bug Reports

When reporting a bug, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to reproduce the problem
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, MongoDB version
6. **Error Output**: Full error messages or stack traces

### Feature Requests

When requesting a feature, please include:

1. **Problem**: The problem or use case the feature would address
2. **Proposed Solution**: Your idea for how to solve it
3. **Alternatives**: Any alternative approaches you considered
4. **Additional Context**: Any relevant background information

---

## Project Structure Guide

When adding new code, place it in the appropriate module:

| Directory               | What goes here                              |
|-------------------------|---------------------------------------------|
| `parser/`               | New log format parsers                      |
| `feature_engineering/`  | Feature extraction and data cleaning logic  |
| `detection/`            | Anomaly detection models and evaluation     |
| `optimization/`         | GA and model evolution logic                |
| `scripts/`              | Utility scripts and pipeline runners        |
| `tests/`                | Test files (mirror the module structure)    |
| `docs/`                 | Documentation files                         |

---

## License

By contributing to MorphGuard, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for contributing to MorphGuard!*
