"""
Structured logging utilities with performance tracking and diagnostics.
"""

import logging
import sys
import time
import uuid
from typing import Any, Dict
from contextlib import contextmanager

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
    
    @contextmanager
    def performance_context(self, operation: str, **kwargs: Any):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation: Name of the operation being tracked
            **kwargs: Additional context to log
            
        Example:
            >>> with logger.performance_context("ocr_extraction", page=1):
            >>>     # perform OCR
            >>>     pass
        """
        start_time = time.time()
        self.debug(f"Starting {operation}", operation=operation, **kwargs)
        
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Completed {operation}",
                operation=operation,
                duration_ms=duration_ms,
                status="success",
                **kwargs
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(
                f"Failed {operation}",
                operation=operation,
                duration_ms=duration_ms,
                status="failed",
                error=str(e),
                error_type=type(e).__name__,
                **kwargs
            )
            raise
    
    def log_system_info(self):
        """Log system information for diagnostics."""
        import platform
        import sys
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        
        try:
            import torch
            system_info["torch_version"] = torch.__version__
            system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["cuda_version"] = torch.version.cuda
                system_info["cuda_device_count"] = torch.cuda.device_count()
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            system_info["torch_available"] = False
        
        self.info("System information", **system_info)


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


def setup_model_warning_suppression(suppress: bool = True) -> None:
    """
    Setup comprehensive warning suppression for model-related warnings.
    
    Args:
        suppress: Whether to suppress warnings (True) or show them (False)
    """
    import warnings
    
    if suppress:
        # Suppress common transformer/model warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*model of type.*not supported.*")
        warnings.filterwarnings("ignore", message=".*deepseek_vl_v2.*DeepseekOCR.*")
        warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
        warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*pad token.*")
        warnings.filterwarnings("ignore", message=".*Flash Attention.*")
        warnings.filterwarnings("ignore", message=".*past_key_value.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*seen_tokens.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*get_max_cache.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*cache_position.*")
        
        # Suppress specific library warnings
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except:
            pass
        
        try:
            import torch
            torch.set_warn_always(False)
        except:
            pass
    else:
        # Reset warnings to default
        warnings.resetwarnings()


def log_performance_metrics(logger: StructuredLogger, metrics: Dict[str, Any]) -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: StructuredLogger instance
        metrics: Dictionary of metrics to log
    """
    logger.info("Performance metrics", **metrics)

