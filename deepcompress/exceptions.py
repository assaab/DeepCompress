"""
Custom exceptions for DeepCompress.

All exceptions inherit from DeepCompressError for easy catching of all library errors.
"""


class DeepCompressError(Exception):
    """Base exception for all DeepCompress errors."""

    def __init__(self, message: str, details: dict or None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class OCRError(DeepCompressError):
    """Raised when OCR processing fails."""

    pass


class GPUError(DeepCompressError):
    """Raised when GPU-related operations fail (OOM, CUDA errors, etc.)."""

    pass


class CacheError(DeepCompressError):
    """Raised when cache operations fail."""

    pass


class VectorDBError(DeepCompressError):
    """Raised when vector database operations fail."""

    pass


class LLMError(DeepCompressError):
    """Raised when LLM API calls fail."""

    pass


class ValidationError(DeepCompressError):
    """Raised when data validation fails."""

    pass


class ProcessingError(DeepCompressError):
    """Raised when document processing fails."""

    pass


class ConfigurationError(DeepCompressError):
    """Raised when configuration is invalid."""

    pass


class StorageError(DeepCompressError):
    """Raised when storage operations (S3, local) fail."""

    pass

