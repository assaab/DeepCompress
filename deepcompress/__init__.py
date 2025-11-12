"""
DeepCompress

A production-ready Python library that reduces LLM document processing costs by 96%
using DeepSeek-OCR vision-based compression and D-TOON optimization.

Example:
    >>> from deepcompress import compress_and_analyze
    >>> result = await compress_and_analyze(
    ...     file="document.pdf",
    ...     query="What is the total income?",
    ...     llm="openai"
    ... )
    >>> print(result.answer)
"""

from typing import Optional

from deepcompress.core.compressor import DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import (
    CacheError,
    DeepCompressError,
    GPUError,
    LLMError,
    OCRError,
    ValidationError,
    VectorDBError,
)
from deepcompress.models.document import Entity, ExtractedDocument, Page, Table
from deepcompress.models.response import CompressionResult
from deepcompress.processing.batch import BatchProcessor
from deepcompress.utils.cost import calculate_savings

__version__ = "1.4.2"
__author__ = "Your Organization"
__license__ = "MIT"

__all__ = [
    # Core classes
    "DocumentCompressor",
    "BatchProcessor",
    "DeepCompressConfig",
    # Models
    "ExtractedDocument",
    "Page",
    "Entity",
    "Table",
    "CompressionResult",
    # Exceptions
    "DeepCompressError",
    "OCRError",
    "GPUError",
    "CacheError",
    "VectorDBError",
    "LLMError",
    "ValidationError",
    # Utilities
    "calculate_savings",
    # High-level API
    "compress_and_analyze",
]


async def compress_and_analyze(
    file: str,
    query: str,
    llm: str = "openai",
    cache: bool = True,
    scrub_pii: bool = True,
    config: Optional[DeepCompressConfig] = None,
) -> CompressionResult:
    """
    One-liner API for document compression and LLM analysis.

    Args:
        file: Path to document (local path, S3 URI, or HTTP URL)
        query: Question to ask the LLM about the document
        llm: LLM provider ('openai', 'claude', 'llama')
        cache: Whether to use Redis caching
        scrub_pii: Whether to scrub PII before sending to LLM
        config: Optional custom DeepCompressConfig (uses defaults if None)

    Returns:
        CompressionResult with compressed document, answer, and metadata

    Example:
        >>> result = await compress_and_analyze(
        ...     file="loan_application.pdf",
        ...     query="What is the applicant's total monthly income?",
        ...     llm="openai"
        ... )
        >>> print(result.answer)
        >>> print(f"Tokens saved: {result.tokens_saved}")
    """
    from deepcompress.integrations.cache import CacheManager
    from deepcompress.integrations.llm import LLMClient
    from deepcompress.processing.pii import PIIScrubber

    if config is None:
        config = DeepCompressConfig()

    compressor = DocumentCompressor(config)
    cache_manager = CacheManager(config) if cache else None
    llm_client = LLMClient(provider=llm, config=config)
    pii_scrubber = PIIScrubber() if scrub_pii else None

    compressed = await compressor.compress(file, cache_manager=cache_manager)

    compressed_text = compressed.optimized_text
    if pii_scrubber:
        compressed_text = pii_scrubber.scrub(compressed_text)

    answer = await llm_client.query(compressed_text, query)

    return CompressionResult(
        document_id=compressed.document_id,
        original_tokens=compressed.original_tokens,
        compressed_tokens=compressed.compressed_tokens,
        compression_ratio=compressed.compression_ratio,
        optimized_text=compressed_text,
        answer=answer.text,
        processing_time_ms=compressed.processing_time_ms + answer.processing_time_ms,
        tokens_saved=compressed.tokens_saved,
        cost_saved_usd=compressed.cost_saved_usd,
        cache_hit=compressed.cache_hit,
    )

