import asyncio

from deepcompress.core.compressor import CompressedDocument, DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.models.document import ExtractedDocument, Page
from deepcompress.models.response import CompressionResult
from deepcompress.processing.protected_facts import (
    append_missing_protected_facts,
    extract_protected_facts,
)
from deepcompress.utils.token_counter import TokenCount


def make_document():
    return ExtractedDocument(
        document_id="contract-1",
        page_count=2,
        mode="small",
        pages=[
            Page(
                page_number=1,
                raw_text=(
                    "Applicant: Ada Lovelace\n"
                    "Email: ada@example.com\n"
                    "Phone: (555) 123-4567\n"
                    "Invoice No: INV-48291\n"
                    "Account number: ACCT-778899\n"
                    "Effective Date: 2026-03-15\n"
                    "Rate: 7.5%\n"
                ),
            ),
            Page(
                page_number=2,
                raw_text=(
                    "Payroll income $17,000/month and freelance income $3,200.\n"
                    "Total monthly income: $20,200.\n"
                    "Payment terms: net 30. The receiving party shall keep all "
                    "confidential information private."
                ),
            ),
        ],
    )


def test_extract_protected_facts_covers_enterprise_values():
    facts = extract_protected_facts(make_document())

    assert "INV-48291" in facts["invoice_numbers"]
    assert "ada@example.com" in facts["emails"]
    assert "(555) 123-4567" in facts["phones"]
    assert "2026-03-15" in facts["dates"]
    assert "$20,200" in facts["amounts"]
    assert "7.5%" in facts["percentages"]
    assert "Ada Lovelace" in facts["names"]
    assert "ACCT-778899" in facts["account_numbers"]
    assert any("Total monthly income" in value for value in facts["table_totals"])
    assert any("shall keep" in value for value in facts["legal_clauses"])
    assert any("Payment terms" in value for value in facts["contract_terms"])


def test_append_missing_protected_facts_deduplicates_existing_values():
    compressed = "[p2] Income: payroll $17,000/month."
    facts = {
        "amounts": [
            {"value": "$17,000/month", "page": 2},
            {"value": "$20,200", "page": 2},
        ]
    }

    output = append_missing_protected_facts(compressed, facts)

    assert output.count("$17,000/month") == 1
    assert "- amounts: [p2] $20,200" in output


def test_compressor_exposes_protected_facts(monkeypatch):
    config = DeepCompressConfig(
        ocr_device="cpu",
        vector_db_provider="none",
        cache_enabled=False,
        protect_facts=True,
    )
    compressor = DocumentCompressor(config)

    async def fake_extract(file_path, document_id=None):
        return make_document()

    def fake_count_tokens(text, provider="openai", model="gpt-4o", api_key=None):
        return TokenCount(
            count=max(1, len(text.split())),
            provider=provider,
            model=model,
            is_estimated=False,
        )

    monkeypatch.setattr(compressor.extractor, "extract", fake_extract)
    monkeypatch.setattr("deepcompress.core.compressor.count_tokens", fake_count_tokens)

    result = asyncio.run(compressor.compress("unused.pdf"))

    assert "$20,200" in result.optimized_text
    assert result.protected_facts["emails"] == ["ada@example.com"]
    assert "INV-48291" in result.protected_facts["invoice_numbers"]


def test_result_models_accept_protected_facts():
    document = make_document()
    compressed = CompressedDocument(
        document_id="contract-1",
        extracted=document,
        optimized_text="compressed",
        original_tokens=10,
        compressed_tokens=5,
        processing_time_ms=1.0,
        protected_facts={"amounts": ["$20,200"]},
    )
    result = CompressionResult(
        document_id="contract-1",
        original_tokens=10,
        compressed_tokens=5,
        compression_ratio=2.0,
        protected_facts=compressed.protected_facts,
        optimized_text="compressed",
        processing_time_ms=1.0,
        tokens_saved=5,
        cost_saved_usd=0.01,
    )

    assert compressed.protected_facts == {"amounts": ["$20,200"]}
    assert result.protected_facts == compressed.protected_facts
