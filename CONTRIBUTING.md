# Contributing to DeepCompress

Thank you for your interest in contributing to DeepCompress! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.2+ (for OCR processing)
- Docker (optional, for containerized development)
- Redis (for caching)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-org/deepcompress.git
cd deepcompress
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -e ".[all]"
```

4. **Copy environment variables:**

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Install pre-commit hooks:**

```bash
pre-commit install
```

## Development Workflow

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Run formatters before committing:

```bash
black deepcompress/
ruff check deepcompress/ --fix
mypy deepcompress/
```

### Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=deepcompress --cov-report=html
```

### Adding New Features

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Write tests first (TDD approach):**

```python
# tests/test_your_feature.py
import pytest
from deepcompress import YourFeature

def test_your_feature():
    result = YourFeature().process()
    assert result is not None
```

3. **Implement the feature:**

```python
# deepcompress/your_feature.py
class YourFeature:
    def process(self):
        # Implementation
        pass
```

4. **Update documentation:**
   - Add docstrings (Google style)
   - Update README.md if needed
   - Add entry to CHANGELOG.md

5. **Commit with conventional commits:**

```bash
git commit -m "feat: add support for custom TOON formats"
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation**
3. **Add entry to CHANGELOG.md**
4. **Request review from maintainers**
5. **Address review comments**
6. **Squash commits before merging**

## Architecture Guidelines

### Module Organization

- `deepcompress/core/` - Core compression and extraction logic
- `deepcompress/models/` - Pydantic data models
- `deepcompress/integrations/` - External service integrations
- `deepcompress/processing/` - Document processing utilities
- `deepcompress/utils/` - Helper utilities

### Code Principles

1. **Async-first**: Use `async/await` for I/O operations
2. **Type hints**: All functions must have type annotations
3. **Error handling**: Use custom exceptions from `deepcompress.exceptions`
4. **Validation**: Use Pydantic for data validation
5. **Logging**: Use structured logging with trace IDs
6. **Testing**: Aim for >90% code coverage

### Performance Considerations

- Use batch processing for multiple documents
- Leverage Redis caching aggressively
- Profile GPU memory usage
- Monitor token usage and costs

## Reporting Issues

### Bug Reports

Include:
- Python version
- CUDA version (if GPU-related)
- Minimal reproducible example
- Error logs with trace IDs
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Expected API design
- Performance considerations
- Breaking changes (if any)

