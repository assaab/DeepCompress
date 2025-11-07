"""
Utility modules for EDC.
"""

from deepcompress.utils.cost import calculate_savings
from deepcompress.utils.logging import get_logger, setup_logging
from deepcompress.utils.metrics import MetricsCollector

__all__ = [
    "calculate_savings",
    "get_logger",
    "setup_logging",
    "MetricsCollector",
]

