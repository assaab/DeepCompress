"""
Structured logging utilities.
"""

import logging
import sys
import uuid
from typing import Any

import orjson


class StructuredLogger:
    """
    Structured JSON logger with trace IDs.

    Features:
    - JSON output for log aggregation
    - Trace ID for request correlation
    - Contextual information
    """

    def __init__(self, name: str, level: str = "INFO") -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.trace_id: str  or None = None

    def set_trace_id(self, trace_id: str  or None = None) -> str:
        """
        Set trace ID for request correlation.

        Args:
            trace_id: Optional trace ID (generates UUID if None)

        Returns:
            Trace ID
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        return self.trace_id

    def _format_message(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Format log message as JSON."""
        log_data = {
            "timestamp": self._get_timestamp(),
            "level": level,
            "logger": self.name,
            "message": message,
            "trace_id": self.trace_id,
            **kwargs,
        }

        return orjson.dumps(log_data).decode("utf-8")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message("DEBUG", message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._format_message("INFO", message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message("WARNING", message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._format_message("ERROR", message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message("CRITICAL", message, **kwargs))

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
    """
    if structured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    logging.root.setLevel(getattr(logging, level.upper()))
    logging.root.addHandler(handler)


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """
    Get structured logger instance.

    Args:
        name: Logger name
        level: Log level

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)

