"""Pytest collection settings for the offline test suite."""

from pathlib import Path


OFFLINE_EXCLUDED = {
    "test_compression_kpis_explained.py",
    "test_fixes.py",
    "test_with_llm.py",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Skip live/API/GPU scripts during default offline test runs."""
    return collection_path.name in OFFLINE_EXCLUDED
