"""
Advanced Usage Examples

Demonstrates advanced features and customization options.
"""

import asyncio

from deepcompress import DocumentCompressor, DeepCompressConfig
from deepcompress.core.optimizer import DTOONOptimizer
from deepcompress.integrations.cache import CacheManager
from deepcompress.integrations.llm import LLMClient
from deepcompress.integrations.vector_db import VectorDBClient
from deepcompress.processing.pii import PIIScrubber
from deepcompress.utils.cost import calculate_cost_per_document, calculate_savings


async def custom_compression() -> None:
    """Custom compression with advanced options."""
    print("=== Custom Compression ===\n")

    config = DeepCompressConfig(
        ocr_mode="base",
        ocr_batch_size=16,
        use_bfloat16=True,
        gpu_memory_fraction=0.85,
    )

    compressor = DocumentCompressor(config)

    optimizer = DTOONOptimizer(
        include_bbox=True,
        include_confidence=True,
        min_confidence=0.9,
    )
    compressor.optimizer = optimizer

    result = await compressor.compress("complex_financial_report.pdf")

    json_tokens, toon_tokens, ratio = optimizer.calculate_compression_ratio(
        result.extracted
    )

    print(f"JSON tokens: {json_tokens:,}")
    print(f"TOON tokens: {toon_tokens:,}")
    print(f"Compression ratio: {ratio:.1f}x\n")


async def custom_pii_scrubbing() -> None:
    """Custom PII scrubbing with additional patterns."""
    print("=== Custom PII Scrubbing ===\n")

    scrubber = PIIScrubber()

    scrubber.add_pattern(
        name="employee_id",
        pattern=r"\bEMP\d{6}\b",
        replacement="[REDACTED_EMP_ID]",
    )

    scrubber.add_pattern(
        name="passport",
        pattern=r"\b[A-Z]{1,2}\d{6,9}\b",
        replacement="[REDACTED_PASSPORT]",
    )

    text = """
    Employee: John Doe
    Employee ID: EMP123456
    SSN: 123-45-6789
    Passport: AB1234567
    """

    scrubbed = scrubber.scrub(text)
    print("Scrubbed text:")
    print(scrubbed)

    detected = scrubber.detect(text)
    print("\nDetected PII:")
    for pii_type, values in detected.items():
        print(f"  {pii_type}: {values}")
    print()


async def semantic_search() -> None:
    """Semantic search with vector database."""
    print("=== Semantic Search ===\n")

    config = DeepCompressConfig()
    compressor = DocumentCompressor(config)
    vector_db = VectorDBClient(config)
    llm_client = LLMClient("openai", config)

    await vector_db.initialize()
    await llm_client.initialize()

    documents = [
        "loan_app_001.pdf",
        "loan_app_002.pdf",
        "loan_app_003.pdf",
    ]

    print("Indexing documents...")
    for doc_path in documents:
        compressed = await compressor.compress(doc_path)
        embedding = await llm_client.embed(compressed.optimized_text)

        await vector_db.upsert(
            document_id=compressed.document_id,
            embedding=embedding,
            metadata={
                "file": doc_path,
                "compressed_text": compressed.optimized_text,
            },
        )

    print("Documents indexed.")

    query = "high income borrowers with low debt"
    print(f"\nSearching for: {query}")

    query_embedding = await llm_client.embed(query)
    results = await vector_db.query(
        embedding=query_embedding,
        top_k=3,
        filters=None,
    )

    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['file']} (score: {result['score']:.3f})")
    print()


async def cost_analysis() -> None:
    """Detailed cost analysis."""
    print("=== Cost Analysis ===\n")

    savings = calculate_savings(
        pages_per_month=250000,
        avg_tokens_per_page=5000,
        target_llm="gpt-4o",
        gpu_cost_per_month=4000,
    )

    print("Monthly Analysis:")
    print(f"  Pages processed: {savings['pages_per_month']:,}")
    print(f"  Cost without DeepCompress: ${savings['cost_without_deepcompress']:,.2f}")
    print(f"  Cost with DeepCompress: ${savings['total_cost_with_deepcompress']:,.2f}")
    print(f"  Monthly savings: ${savings['monthly_savings']:,.2f}")
    print(f"  Payback period: {savings['payback_months']:.1f} months")
    print(f"  3-year ROI: {savings['three_year_roi_percent']:.0f}%\n")

    per_doc_cost = calculate_cost_per_document(
        pages=50,
        tokens_per_page=5000,
        llm="gpt-4o",
    )

    print("Per-Document Cost (50 pages):")
    print(f"  Without DeepCompress: ${per_doc_cost['cost_without_deepcompress']:.4f}")
    print(f"  With DeepCompress (TOON): ${per_doc_cost['cost_with_deepcompress_toon']:.4f}")
    print(f"  Savings: ${per_doc_cost['savings_toon']:.4f}\n")


async def cache_management() -> None:
    """Cache management and statistics."""
    print("=== Cache Management ===\n")

    config = DeepCompressConfig(cache_enabled=True)
    cache = CacheManager(config)

    await cache.connect()

    await cache.set("test_key", {"data": "test_value"}, ttl=3600)

    value = await cache.get("test_key")
    print(f"Retrieved value: {value}")

    exists = await cache.exists("test_key")
    print(f"Key exists: {exists}")

    stats = await cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  Total keys: {stats['keys']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")

    await cache.disconnect()
    print()


async def main() -> None:
    """Run all advanced examples."""
    await custom_compression()
    await custom_pii_scrubbing()
    await semantic_search()
    await cost_analysis()
    await cache_management()

    print("=== All Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

