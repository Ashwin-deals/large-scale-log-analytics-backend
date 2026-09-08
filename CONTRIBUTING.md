# Contributing to MorphGuard

Thank you for your interest in contributing to MorphGuard! This document provides guidelines for contributing.

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

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

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

- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **New Parsers**: Add support for new log formats
- **New Detection Algorithms**: Implement alternative anomaly detection methods

### Priority Areas

The following areas are actively seeking contributions:

| Area                        | Priority | Difficulty |
|-----------------------------|----------|------------|
| Additional log parsers      | High     | Medium     |
| Autoencoder detection       | High     | Hard       |
| Kafka streaming pipeline    | High     | Hard       |
| API endpoint tests          | Medium   | Easy       |
| CI/CD pipeline              | Medium   | Medium     |
| Docker containerization     | Medium   | Easy       |
| Cloud connectors            | Low      | Medium     |

---

## Development Setup

### Prerequisites

- Python 3.10+
- MongoDB 6.0+ (or Atlas)
- Git

### Quick Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m pytest tests/ -v
```

### Running the Development Server

```bash
FLASK_DEBUG=1 python3 app.py
```

---

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 120 characters
- Use Google-style docstrings

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
        parser: Parser instance for field extraction.
        strict: If True, raise on unparseable lines.

    Returns:
        Dictionary of extracted fields, or None if unparseable.

    Raises:
        ParseError: If strict is True and parsing fails.
    """
    ...
```

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>
```

### Types

| Type     | Description                                |
|----------|---------------------------------------------|
| feat     | A new feature                               |
| fix      | A bug fix                                   |
| docs     | Documentation changes                       |
| style    | Formatting changes                          |
| refactor | Code restructuring                          |
| perf     | Performance improvement                     |
| test     | Adding or updating tests                    |
| chore    | Build/tool changes                          |

### Examples

```
feat(parser): add Apache access log parser
fix(auth): read JWT secret per-call to avoid import-order bug
docs(api): add API reference documentation
test(detection): add evaluation metrics unit tests
```

---

## Pull Request Process

1. Ensure code follows the coding standards
2. Run tests: `python3 -m pytest tests/ -v`
3. Update documentation if applicable
4. Add tests for new functionality
5. Submit PR against `main`
6. A maintainer will review within 48 hours

### PR Description Should Include

- **What**: Clear description of changes
- **Why**: Motivation or problem solved
- **Testing**: How you verified correctness
- **Breaking Changes**: Any backward-incompatible changes

---

## Issue Guidelines

### Bug Reports

Include: description, steps to reproduce, expected vs actual behavior, environment info, and error output.

### Feature Requests

Include: problem statement, proposed solution, alternatives considered, and implementation notes.

---

## Project Structure

| Directory               | Purpose                                     |
|-------------------------|---------------------------------------------|
| `parser/`               | Log format parsers                          |
| `feature_engineering/`  | Feature extraction and cleaning             |
| `detection/`            | Anomaly detection models                    |
| `optimization/`         | GA and model evolution                      |
| `scripts/`              | Pipeline and utility scripts                |
| `tests/`                | Unit and integration tests                  |
| `docs/`                 | Documentation                               |

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

*Thank you for contributing to MorphGuard!*
