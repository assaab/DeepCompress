"""
DeepCompress Quickstart Example

Demonstrates basic document compression and LLM analysis.
"""

import asyncio

from deepcompress import compress_and_analyze


async def main() -> None:
    """Run quickstart example."""
    print("=" * 60)
    print("DeepCompress Quickstart Example")
    print("=" * 60)

    print("\nCompressing and analyzing document...")

    result = await compress_and_analyze(
        file="sample_document.pdf",
        query="What are the key highlights from this document?",
        llm="openai",
        cache=True,
        scrub_pii=True,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nDocument ID: {result.document_id}")
    print(f"\nOriginal tokens: {result.original_tokens:,}")
    print(f"Compressed tokens: {result.compressed_tokens:,}")
    print(f"Compression ratio: {result.compression_ratio:.1f}x")
    print(f"Tokens saved: {result.tokens_saved:,}")
    print(f"Cost saved: ${result.cost_saved_usd:.4f}")
    print(f"Processing time: {result.processing_time_ms:.0f}ms")
    print(f"Cache hit: {result.cache_hit}")

    print(f"\n--- LLM Answer ---")
    print(result.answer)

    print(f"\n--- Compressed Document (D-TOON) ---")
    print(result.optimized_text[:500] + "...")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

