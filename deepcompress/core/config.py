"""
Configuration management for DeepCompress.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepCompressConfig(BaseSettings):
    """
    Centralized configuration for DeepCompress.

    Loads configuration from environment variables with DEEPCOMPRESS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OCR Configuration
    ocr_model: str = Field(
        default="deepseek-ai/DeepSeek-OCR",
        description="DeepSeek-OCR model identifier",
    )
    ocr_model_revision: str = Field(
        default="main",
        description="Model revision/commit hash to pin for security and stability",
    )
    ocr_mode: Literal["small", "base", "large"] = Field(
        default="small",
        description="OCR mode (small=100 tokens, base=200, large=400)",
    )
    ocr_device: str = Field(
        default="cuda:0",
        description="Device for OCR processing (cuda:0, cuda:1, cpu)",
    )
    ocr_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of pages to process concurrently in each batch (higher = faster but more memory)",
    )
    ocr_max_workers: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum OCR worker processes",
    )
    ocr_engine: str = Field(
        default="deepseek",
        description="OCR engine (ignored, for compatibility only - always uses DeepSeek-OCR)",
    )
    ocr_max_new_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum new tokens to generate per page during OCR",
    )
    ocr_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for OCR text generation",
    )
    ocr_repetition_penalty: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Repetition penalty for OCR generation (prevents repetitive text)",
    )
    ocr_inference_timeout: int = Field(
        default=0,
        ge=0,
        le=3600,
        description="Timeout in seconds for OCR inference per page (0=disabled, prevents infinite generation)",
    )
    ocr_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of retry attempts for OCR extraction per page before skipping",
    )
    ocr_skip_failed_pages: bool = Field(
        default=True,
        description="Skip pages that fail OCR extraction instead of raising an error",
    )
    min_ocr_text_length: int = Field(
        default=10,
        ge=0,
        description="Minimum text length for a page to be considered valid post-OCR. Pages with fewer characters will be warned about but not failed.",
    )

    # Cache Configuration
    cache_url: str = Field(
        default="redis://localhost:6379",
        description="Redis cache URL",
    )
    cache_ttl: int = Field(
        default=86400,
        ge=0,
        description="Cache TTL in seconds (default: 24 hours)",
    )
    cache_enabled: bool = Field(
        default=False,
        description="Whether to enable caching",
    )

    # Vector Database Configuration
    vector_db_provider: Literal["pinecone", "weaviate", "none"] = Field(
        default="pinecone",
        description="Vector database provider",
    )
    vector_db_api_key: str = Field(
        default="",
        description="Vector database API key",
    )
    vector_db_environment: str = Field(
        default="us-east-1-aws",
        description="Vector database environment/region",
    )
    vector_db_index_name: str = Field(
        default="edc-documents",
        description="Vector database index name",
    )

    # LLM Configuration
    llm_provider: Literal["openai", "claude", "llama"] = Field(
        default="openai",
        description="LLM provider",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API key (optional for compression-only operations)",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model identifier",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens for LLM generation",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )

    # AWS Configuration
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key ID",
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS secret access key",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region",
    )
    aws_s3_bucket: str = Field(
        default="",
        description="S3 bucket for document storage",
    )

    # Security Configuration
    pii_scrubbing: bool = Field(
        default=True,
        description="Enable PII scrubbing",
    )
    encryption_key: str = Field(
        default="",
        description="Fernet encryption key for cache",
    )

    # Processing Configuration
    max_pages_per_document: int = Field(
        default=1000,
        ge=1,
        description="Maximum pages per document",
    )
    processing_timeout: int = Field(
        default=300,
        ge=1,
        description="Processing timeout in seconds",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts for API calls",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0,
        description="Delay between retries in seconds",
    )

    # Monitoring Configuration
    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    structured_logging: bool = Field(
        default=True,
        description="Enable structured JSON logging",
    )
    enable_performance_logging: bool = Field(
        default=True,
        description="Enable detailed performance logging for profiling",
    )
    suppress_model_warnings: bool = Field(
        default=True,
        description="Suppress non-critical model warnings for cleaner output",
    )

    # Performance Configuration
    gpu_memory_fraction: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use",
    )
    use_bfloat16: bool = Field(
        default=True,
        description="Use bfloat16 precision",
    )
    enable_flash_attention: bool = Field(
        default=True,
        description="Enable Flash Attention optimization",
    )

    def validate_config(self) -> None:
        """Validate configuration and raise errors for invalid settings."""
        from deepcompress.exceptions import ConfigurationError

        if self.cache_enabled and not self.cache_url:
            raise ConfigurationError("cache_url required when cache_enabled=True")

        if self.vector_db_provider != "none" and not self.vector_db_api_key:
            raise ConfigurationError(
                f"vector_db_api_key required for provider={self.vector_db_provider}"
            )

        # llm_api_key is now optional - only required when actually using LLM functionality
        # For compression-only operations (compressor.compress()), it's not needed

        # Only validate CUDA if using cuda device
        if self.ocr_device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    # Warn but don't fail - will fall back to CPU
                    import warnings
                    warnings.warn("CUDA not available, will use CPU instead")
                    self.ocr_device = "cpu"
            except ImportError:
                raise ConfigurationError("torch not installed but ocr_device=cuda")

    @property
    def tokens_per_page_by_mode(self) -> dict[str, int]:
        """Get token counts per page for each OCR mode."""
        return {
            "small": 100,
            "base": 200,
            "large": 400,
        }

    @property
    def vision_tokens_per_page(self) -> int:
        """Get vision tokens per page for current OCR mode."""
        return self.tokens_per_page_by_mode[self.ocr_mode]

