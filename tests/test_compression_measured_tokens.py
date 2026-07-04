import asyncio

import pytest

from deepcompress.core.compressor import DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import ConfigurationError
from deepcompress.models.document import ExtractedDocument, Page
from deepcompress.models.response import CompressionResult
from deepcompress.utils.token_counter import TokenCount


def make_config(**overrides):
    values = {
        "ocr_device": "cpu",
        "vector_db_provider": "none",
        "cache_enabled": False,
    }
    values.update(overrides)
    return DeepCompressConfig(**values)


def make_document():
    return ExtractedDocument(
        document_id="doc-1",
        page_count=1,
        mode="small",
        pages=[Page(page_number=1, raw_text="Original document text")],
    )


def test_compressor_populates_measured_token_fields(monkeypatch):
    compressor = DocumentCompressor(make_config())

    async def fake_extract(file_path, document_id=None):
        return make_document()

    def fake_count_tokens(text, provider="openai", model="gpt-4o", api_key=None):
        if "Document ID:" in text:
            return TokenCount(
                count=5,
                provider=provider,
                model=model,
                is_estimated=False,
            )
        return TokenCount(
            count=20,
            provider=provider,
            model=model,
            is_estimated=False,
        )

    monkeypatch.setattr(compressor.extractor, "extract", fake_extract)
    monkeypatch.setattr("deepcompress.core.compressor.count_tokens", fake_count_tokens)

    result = asyncio.run(compressor.compress("unused.pdf"))

    assert result.original_tokens_measured == 20
    assert result.compressed_tokens_measured == 5
    assert result.compression_ratio_measured == 4.0
    assert result.original_tokens == result.original_tokens_measured
    assert result.compressed_tokens == result.compressed_tokens_measured
    assert result.compression_ratio == result.compression_ratio_measured
    assert result.tokens_saved == 15
    assert result.token_counter_provider == "openai"
    assert result.token_counter_model == "gpt-4o"
    assert result.is_estimated is False


def test_structured_mode_requires_api_key(monkeypatch):
    compressor = DocumentCompressor(make_config(dtoon_mode="structured", llm_api_key=""))

    async def fake_extract(file_path, document_id=None):
        return make_document()

    monkeypatch.setattr(compressor.extractor, "extract", fake_extract)

    with pytest.raises(ConfigurationError):
        asyncio.run(compressor.compress("unused.pdf"))


def test_compression_result_accepts_measured_fields():
    result = CompressionResult(
        document_id="doc-1",
        original_tokens=20,
        compressed_tokens=5,
        compression_ratio=4.0,
        original_tokens_measured=20,
        compressed_tokens_measured=5,
        compression_ratio_measured=4.0,
        token_counter_provider="openai",
        token_counter_model="gpt-4o",
        is_estimated=False,
        optimized_text="compressed",
        processing_time_ms=10.0,
        tokens_saved=15,
        cost_saved_usd=0.01,
    )

    assert result.original_tokens_measured == result.original_tokens
    assert result.compression_ratio_measured == result.compression_ratio
