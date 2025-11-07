"""
Batch processing for high-throughput document compression.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from deepcompress.core.compressor import CompressedDocument, DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import ProcessingError
from deepcompress.integrations.cache import CacheManager
from deepcompress.models.response import BatchResult


class BatchProcessor:
    """
    High-throughput batch processor for document compression.

    Features:
    - Parallel processing with configurable workers
    - Progress tracking
    - Error handling and recovery
    - Cache integration
    - Metrics collection
    """

    def __init__(
        self,
        compressor: DocumentCompressor,
        config: DeepCompressConfig,
        cache_manager: CacheManager  or None = None,
    ) -> None:
        self.compressor = compressor
        self.config = config
        self.cache_manager = cache_manager
        self._processed = 0
        self._failed = 0
        self._total_tokens_saved = 0
        self._total_cost_saved = 0.0
        self._errors: list[str] = []

    async def process_directory(
        self,
        directory: str,
        batch_size: int = 10,
        pattern: str = "*.pdf",
    ) -> AsyncGenerator[CompressedDocument, None]:
        """
        Process all documents in a directory.

        Args:
            directory: Directory path (local or S3 URI)
            batch_size: Number of documents to process in parallel
            pattern: File pattern (e.g., "*.pdf")

        Yields:
            CompressedDocument for each processed file

        Example:
            >>> processor = BatchProcessor(compressor, config)
            >>> async for result in processor.process_directory("s3://bucket/docs/"):
            ...     print(f"Processed: {result.document_id}")
        """
        files = await self._list_files(directory, pattern)

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            results = await self._process_batch(batch)

            for result in results:
                if isinstance(result, CompressedDocument):
                    yield result

    async def process_files(
        self,
        files: list[str],
        batch_size: int = 10,
    ) -> BatchResult:
        """
        Process a list of files.

        Args:
            files: List of file paths
            batch_size: Batch size for parallel processing

        Returns:
            BatchResult with summary statistics

        Example:
            >>> processor = BatchProcessor(compressor, config)
            >>> result = await processor.process_files(["doc1.pdf", "doc2.pdf"])
            >>> print(f"Processed: {result.successful}/{result.total_documents}")
        """
        start_time = time.time()
        total_pages = 0

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            results = await self._process_batch(batch)

            for result in results:
                if isinstance(result, CompressedDocument):
                    total_pages += result.extracted.page_count

        processing_time_ms = (time.time() - start_time) * 1000
        cache_hit_rate = await self._calculate_cache_hit_rate()

        return BatchResult(
            total_documents=len(files),
            successful=self._processed,
            failed=self._failed,
            total_tokens_saved=self._total_tokens_saved,
            total_cost_saved_usd=self._total_cost_saved,
            total_processing_time_ms=processing_time_ms,
            throughput_pages_per_sec=total_pages / (processing_time_ms / 1000),
            cache_hit_rate=cache_hit_rate,
            errors=self._errors,
        )

    async def _process_batch(
        self,
        files: list[str],
    ) -> list:
        """
        Process a batch of files in parallel.

        Args:
            files: List of file paths

        Returns:
            List of results or exceptions
        """
        tasks = [
            self._process_single(file)
            for file in files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._failed += 1
                self._errors.append(str(result))
            elif isinstance(result, CompressedDocument):
                self._processed += 1
                self._total_tokens_saved += result.tokens_saved
                self._total_cost_saved += result.cost_saved_usd

        return results

    async def _process_single(self, file: str) -> CompressedDocument:
        """
        Process single document with error handling.

        Args:
            file: File path

        Returns:
            CompressedDocument

        Raises:
            ProcessingError: If processing fails after retries
        """
        for attempt in range(self.config.retry_attempts):
            try:
                result = await self.compressor.compress(
                    file=file,
                    cache_manager=self.cache_manager,
                )
                return result

            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise ProcessingError(
                        f"Failed to process {file} after {self.config.retry_attempts} attempts",
                        details={"error": str(e)},
                    )

                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        raise ProcessingError(f"Failed to process {file}")

    async def _list_files(self, directory: str, pattern: str) -> list[str]:
        """
        List files in directory (local or S3).

        Args:
            directory: Directory path
            pattern: File pattern

        Returns:
            List of file paths
        """
        if directory.startswith("s3://"):
            return await self._list_s3_files(directory, pattern)
        else:
            return self._list_local_files(directory, pattern)

    def _list_local_files(self, directory: str, pattern: str) -> list[str]:
        """List local files."""
        path = Path(directory)
        return [str(f) for f in path.glob(pattern)]

    async def _list_s3_files(self, s3_uri: str, pattern: str) -> list[str]:
        """List S3 files."""
        from urllib.parse import urlparse

        import aioboto3

        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        session = aioboto3.Session()
        files = []

        async with session.client(
            "s3",
            region_name=self.config.aws_region,
            aws_access_key_id=self.config.aws_access_key_id or None,
            aws_secret_access_key=self.config.aws_secret_access_key or None,
        ) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if self._matches_pattern(key, pattern):
                        files.append(f"s3://{bucket}/{key}")

        return files

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if S3 key matches pattern."""
        import fnmatch

        return fnmatch.fnmatch(key, pattern)

    async def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.cache_manager:
            return 0.0

        try:
            stats = await self.cache_manager.get_stats()
            return stats.get("hit_rate", 0.0)
        except Exception:
            return 0.0

    def get_progress(self) -> dict[str, Any]:
        """
        Get current progress statistics.

        Returns:
            Dictionary with progress metrics
        """
        return {
            "processed": self._processed,
            "failed": self._failed,
            "total_tokens_saved": self._total_tokens_saved,
            "total_cost_saved_usd": self._total_cost_saved,
            "errors": len(self._errors),
        }

