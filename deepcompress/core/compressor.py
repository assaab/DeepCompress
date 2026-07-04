"""
Main document compression engine.
"""

import time
from typing import Any

from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.extractor import OCRExtractor
from deepcompress.core.optimizer import DTOONOptimizer
from deepcompress.exceptions import ConfigurationError, ProcessingError
from deepcompress.models.document import ExtractedDocument
from deepcompress.utils.token_counter import count_tokens


class CompressedDocument:
    """
    Result of document compression with metadata.

    Attributes:
        document_id: Unique document identifier
        extracted: ExtractedDocument from OCR
        optimized_text: D-TOON optimized text
        original_tokens: Original token count (estimated)
        compressed_tokens: Compressed token count
        compression_ratio: Compression ratio
        tokens_saved: Number of tokens saved
        cost_saved_usd: Cost saved in USD (based on GPT-4o pricing)
        processing_time_ms: Total processing time in milliseconds
        cache_hit: Whether result was served from cache
    """

    def __init__(
        self,
        document_id: str,
        extracted: ExtractedDocument,
        optimized_text: str,
        original_tokens: int,
        compressed_tokens: int,
        processing_time_ms: float,
        cache_hit: bool = False,
        original_tokens_measured: int or None = None,
        compressed_tokens_measured: int or None = None,
        compression_ratio_measured: float or None = None,
        token_counter_provider: str = "openai",
        token_counter_model: str = "gpt-4o",
        is_estimated: bool = False,
    ) -> None:
        self.document_id = document_id
        self.extracted = extracted
        self.optimized_text = optimized_text
        self.original_tokens_measured = (
            original_tokens_measured
            if original_tokens_measured is not None
            else original_tokens
        )
        self.compressed_tokens_measured = (
            compressed_tokens_measured
            if compressed_tokens_measured is not None
            else compressed_tokens
        )
        self.compression_ratio_measured = (
            compression_ratio_measured
            if compression_ratio_measured is not None
            else self._calculate_compression_ratio(
                self.original_tokens_measured,
                self.compressed_tokens_measured,
            )
        )
        self.token_counter_provider = token_counter_provider
        self.token_counter_model = token_counter_model
        self.is_estimated = is_estimated

        # Backward-compatible aliases now use the measured values.
        self.original_tokens = self.original_tokens_measured
        self.compressed_tokens = self.compressed_tokens_measured
        self.compression_ratio = self.compression_ratio_measured
        self.tokens_saved = self.original_tokens - self.compressed_tokens
        self.cost_saved_usd = self._calculate_cost_saved(
            self.original_tokens, self.compressed_tokens
        )
        self.processing_time_ms = processing_time_ms
        self.cache_hit = cache_hit

    def _calculate_compression_ratio(self, original: int, compressed: int) -> float:
        """Calculate compression ratio with a safe zero-token fallback."""
        return original / compressed if compressed > 0 else 1.0

    def _calculate_cost_saved(self, original: int, compressed: int) -> float:
        """
        Calculate cost savings based on GPT-4o pricing.

        GPT-4o pricing (as of Nov 2024):
        - Input: $0.005 per 1K tokens
        - Output: $0.015 per 1K tokens

        We assume 2x input usage (embedding + query).
        """
        tokens_saved = original - compressed
        cost_per_token = 0.005 / 1000
        return tokens_saved * cost_per_token * 2


class DocumentCompressor:
    """
    Main compression engine combining OCR extraction and D-TOON optimization.

    Reduces document tokens by 96% through:
    1. DeepSeek-OCR vision compression (5000 → 200 tokens/page)
    2. D-TOON format optimization (200 → 80 tokens/page)
    """

    def __init__(self, config: DeepCompressConfig  or None = None) -> None:
        """
        Initialize document compressor.

        Args:
            config: DeepCompressConfig instance (uses defaults if None)
        """
        self.config = config or DeepCompressConfig()
        self.config.validate_config()

        self.extractor = OCRExtractor(self.config)
        self.optimizer = DTOONOptimizer(
            include_bbox=False,
            include_confidence=False,
            min_confidence=0.8,
        )

    async def compress(
        self,
        file: str,
        document_id: str  or None = None,
        cache_manager: Any = None,
    ) -> CompressedDocument:
        """
        Compress document using OCR + D-TOON optimization.

        Args:
            file: Path to document (local path, S3 URI, or HTTP URL)
            document_id: Optional document ID (generated if None)
            cache_manager: Optional CacheManager for caching

        Returns:
            CompressedDocument with compression results

        Example:
            >>> compressor = DocumentCompressor()
            >>> result = await compressor.compress("document.pdf")
            >>> print(f"Compression ratio: {result.compression_ratio:.1f}x")
            >>> print(f"Tokens saved: {result.tokens_saved}")
        """
        start_time = time.time()

        if cache_manager and document_id:
            cached = await cache_manager.get(document_id)
            if cached:
                return cached

        try:
            file_path = await self._resolve_file_path(file)

            extracted = await self.extractor.extract(
                file_path=file_path,
                document_id=document_id,
            )

            optimized_text = await self.optimizer.optimize_async(
                extracted,
                mode=self.config.dtoon_mode,
                llm_client=await self._get_dtoon_llm_client(),
            )

            original_count, compressed_count = self._count_compression_tokens(
                extracted,
                optimized_text,
            )
            compression_ratio = (
                original_count.count / compressed_count.count
                if compressed_count.count > 0
                else 1.0
            )

            processing_time_ms = (time.time() - start_time) * 1000

            result = CompressedDocument(
                document_id=extracted.document_id,
                extracted=extracted,
                optimized_text=optimized_text,
                original_tokens=original_count.count,
                compressed_tokens=compressed_count.count,
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                original_tokens_measured=original_count.count,
                compressed_tokens_measured=compressed_count.count,
                compression_ratio_measured=compression_ratio,
                token_counter_provider=compressed_count.provider,
                token_counter_model=compressed_count.model,
                is_estimated=original_count.is_estimated or compressed_count.is_estimated,
            )

            if cache_manager:
                await cache_manager.set(extracted.document_id, result)

            return result

        except ConfigurationError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Failed to compress document: {file}",
                details={"error": str(e)},
            )

    async def _resolve_file_path(self, file: str) -> str:
        """
        Resolve file path (local, S3, HTTP).

        Args:
            file: File identifier

        Returns:
            Local file path
        """
        if file.startswith("s3://"):
            return await self._download_from_s3(file)
        elif file.startswith("http://") or file.startswith("https://"):
            return await self._download_from_http(file)
        else:
            return file

    async def _download_from_s3(self, s3_uri: str) -> str:
        """Download file from S3."""
        import tempfile
        from urllib.parse import urlparse

        import aioboto3

        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        session = aioboto3.Session()
        async with session.client(
            "s3",
            region_name=self.config.aws_region,
            aws_access_key_id=self.config.aws_access_key_id or None,
            aws_secret_access_key=self.config.aws_secret_access_key or None,
        ) as s3:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            await s3.download_file(bucket, key, temp_file.name)
            return temp_file.name

    async def _download_from_http(self, url: str) -> str:
        """Download file from HTTP."""
        import tempfile

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name

    def _estimate_original_tokens(self, document: ExtractedDocument) -> int:
        """
        Estimate original token count before compression.

        Assumes ~5000 tokens per page for typical financial documents.
        """
        return document.page_count * 5000

    async def _get_dtoon_llm_client(self) -> Any:
        """Create an LLM client only for LLM-assisted D-TOON modes."""
        if self.config.dtoon_mode == "raw":
            return None

        if self.config.llm_provider == "llama":
            raise ConfigurationError(
                "D-TOON structured and rag modes require an API-backed LLM provider"
            )

        if not self.config.llm_api_key:
            raise ConfigurationError(
                f"D-TOON {self.config.dtoon_mode} mode requires llm_api_key"
            )

        from deepcompress.integrations.llm import LLMClient

        return LLMClient(provider=self.config.llm_provider, config=self.config)

    def _count_compression_tokens(
        self,
        document: ExtractedDocument,
        optimized_text: str,
    ):
        """Count original OCR text and compressed output tokens."""
        original_text = self._document_text(document)
        provider = self.config.token_counter_provider
        model = self.config.token_counter_model or self.config.llm_model

        original_count = count_tokens(
            original_text,
            provider=provider,
            model=model,
            api_key=self.config.llm_api_key or None,
        )
        compressed_count = count_tokens(
            optimized_text,
            provider=provider,
            model=model,
            api_key=self.config.llm_api_key or None,
        )

        return original_count, compressed_count

    def _document_text(self, document: ExtractedDocument) -> str:
        """Concatenate OCR page text for measured original token counts."""
        return "\n\n".join(
            page.raw_text.strip()
            for page in document.pages
            if page.raw_text and page.raw_text.strip()
        )

    async def compress_batch(
        self,
        files: list[str],
        cache_manager: Any = None,
    ) -> list[CompressedDocument]:
        """
        Compress multiple documents in batch.

        Args:
            files: List of file paths
            cache_manager: Optional CacheManager

        Returns:
            List of CompressedDocuments
        """
        import asyncio

        tasks = [
            self.compress(file, cache_manager=cache_manager) for file in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise ProcessingError(
                    f"Batch compression failed for {files[i]}",
                    details={"error": str(result)},
                )
            documents.append(result)

        return documents

