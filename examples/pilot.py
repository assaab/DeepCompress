"""
Pilot Batch Processing Example

Process a batch of loan applications to validate accuracy and cost savings.
"""

import asyncio

from deepcompress import BatchProcessor, DocumentCompressor, DeepCompressConfig
from deepcompress.integrations.cache import CacheManager


async def pilot() -> None:
    """Run pilot batch processing."""
    print("=" * 60)
    print("DeepCompress Pilot Batch Processing")
    print("=" * 60)

    config = DeepCompressConfig(
        ocr_mode="small",
        cache_enabled=True,
        pii_scrubbing=True,
    )

    compressor = DocumentCompressor(config)
    cache_manager = CacheManager(config)
    processor = BatchProcessor(compressor, config, cache_manager)

    print("\nProcessing 1,000 loan applications (50,000 pages)...")
    print("This may take several minutes...\n")

    results = []
    processed_count = 0

    async for result in processor.process_directory(
        directory="data/pilot/loan-applications/",
        batch_size=50,
        pattern="*.pdf",
    ):
        processed_count += 1
        results.append(result)

        if processed_count % 100 == 0:
            progress = processor.get_progress()
            print(f"Progress: {processed_count} documents")
            print(f"  Tokens saved: {progress['total_tokens_saved']:,}")
            print(f"  Cost saved: ${progress['total_cost_saved_usd']:.2f}")
            print(f"  Failed: {progress['failed']}")

    summary = processor.get_progress()

    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)

    print(f"\nDocuments processed: {summary['processed']:,}")
    print(f"Documents failed: {summary['failed']:,}")
    print(f"Success rate: {summary['processed'] / (summary['processed'] + summary['failed']) * 100:.1f}%")

    print(f"\nTotal tokens saved: {summary['total_tokens_saved']:,}")
    print(f"Total cost saved: ${summary['total_cost_saved_usd']:,.2f}")

    avg_processing_time = (
        sum(r.processing_time_ms for r in results) / len(results)
    )
    print(f"Average processing time: {avg_processing_time:.0f}ms/document")

    cache_stats = await cache_manager.get_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print("\nManually validate 100 random documents for:")
    print("  - Table accuracy (target: >97%)")
    print("  - Entity extraction completeness")
    print("  - PII scrubbing effectiveness")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(pilot())

