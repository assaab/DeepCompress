"""
Example: Configurable Batch Processing

This example demonstrates how to configure batch processing for optimal
performance when processing multi-page documents.
"""

import asyncio
import time
from deepcompress import DeepCompressConfig, DocumentCompressor


async def example_sequential():
    """
    Example 1: Sequential processing (batch_size=1)
    
    Processes one page at a time. Slowest but uses minimal memory.
    """
    print("=" * 70)
    print("Example 1: Sequential Processing (batch_size=1)")
    print("=" * 70)
    
    config = DeepCompressConfig(
        ocr_mode="small",
        ocr_device="cuda:0",
        ocr_batch_size=1,  # Process 1 page at a time (sequential)
        
        # Safety parameters
        ocr_max_new_tokens=1024,
        ocr_temperature=0.0,
        ocr_repetition_penalty=1.5,
        
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  Batch Size:           {config.ocr_batch_size} page(s) at a time")
    print(f"  Processing Mode:      Sequential")
    print(f"  Memory Usage:         Low")
    print(f"  Speed:                Slowest")
    print("\n✓ Best for: Low memory systems, CPU processing")


async def example_small_batches():
    """
    Example 2: Small batches (batch_size=2-4)
    
    Good balance between speed and memory. Recommended for most users.
    """
    print("\n" + "=" * 70)
    print("Example 2: Small Batches (batch_size=4)")
    print("=" * 70)
    
    config = DeepCompressConfig(
        ocr_mode="small",
        ocr_device="cuda:0",
        ocr_batch_size=4,  # Process 4 pages concurrently
        
        # Safety parameters
        ocr_max_new_tokens=1024,
        ocr_temperature=0.0,
        ocr_repetition_penalty=1.5,
        
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  Batch Size:           {config.ocr_batch_size} pages at a time")
    print(f"  Processing Mode:      Concurrent batches")
    print(f"  Memory Usage:         Moderate")
    print(f"  Speed:                Good (2-3x faster than sequential)")
    print("\n✓ Best for: Most use cases, balanced performance")


async def example_medium_batches():
    """
    Example 3: Medium batches (batch_size=8)
    
    Default setting. Good for GPUs with 8GB+ VRAM.
    """
    print("\n" + "=" * 70)
    print("Example 3: Medium Batches (batch_size=8) [DEFAULT]")
    print("=" * 70)
    
    config = DeepCompressConfig(
        ocr_mode="small",
        ocr_device="cuda:0",
        ocr_batch_size=8,  # Default: Process 8 pages concurrently
        
        # Safety parameters
        ocr_max_new_tokens=1024,
        ocr_temperature=0.0,
        ocr_repetition_penalty=1.5,
        
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  Batch Size:           {config.ocr_batch_size} pages at a time")
    print(f"  Processing Mode:      Concurrent batches")
    print(f"  Memory Usage:         Moderate-High")
    print(f"  Speed:                Very Good (4-5x faster than sequential)")
    print("\n✓ Best for: Standard GPU setups (8-12GB VRAM)")


async def example_large_batches():
    """
    Example 4: Large batches (batch_size=16)
    
    Maximum parallelism. Requires high-end GPU.
    """
    print("\n" + "=" * 70)
    print("Example 4: Large Batches (batch_size=16)")
    print("=" * 70)
    
    config = DeepCompressConfig(
        ocr_mode="small",
        ocr_device="cuda:0",
        ocr_batch_size=16,  # Process 16 pages concurrently
        
        # Safety parameters
        ocr_max_new_tokens=1024,
        ocr_temperature=0.0,
        ocr_repetition_penalty=1.5,
        
        # May need to limit GPU memory
        gpu_memory_fraction=0.9,
        
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  Batch Size:           {config.ocr_batch_size} pages at a time")
    print(f"  Processing Mode:      High concurrency")
    print(f"  Memory Usage:         High")
    print(f"  Speed:                Fastest (6-8x faster than sequential)")
    print("\n✓ Best for: High-end GPUs (16GB+ VRAM), large documents")


async def example_adaptive():
    """
    Example 5: Adaptive batch size based on document size
    
    Automatically adjust batch size based on number of pages.
    """
    print("\n" + "=" * 70)
    print("Example 5: Adaptive Batch Size")
    print("=" * 70)
    
    def get_optimal_batch_size(num_pages: int) -> int:
        """Calculate optimal batch size based on document size"""
        if num_pages <= 5:
            return num_pages  # Small doc: process all at once
        elif num_pages <= 20:
            return 8  # Medium doc: use default
        elif num_pages <= 100:
            return 16  # Large doc: use large batches
        else:
            return 32  # Very large doc: maximum parallelism
    
    # Example for different document sizes
    examples = [
        (3, "Small document (3 pages)"),
        (15, "Medium document (15 pages)"),
        (50, "Large document (50 pages)"),
        (200, "Very large document (200 pages)"),
    ]
    
    print("\nAdaptive Batch Size Recommendations:")
    print("-" * 70)
    for num_pages, description in examples:
        batch_size = get_optimal_batch_size(num_pages)
        print(f"  {description:40} → batch_size={batch_size}")
    
    print("\n✓ Best for: Variable document sizes, production systems")


async def example_performance_comparison():
    """
    Example 6: Performance comparison
    
    Shows expected performance gains with different batch sizes.
    """
    print("\n" + "=" * 70)
    print("Example 6: Performance Comparison")
    print("=" * 70)
    
    # Simulated performance data (adjust based on your hardware)
    performance_data = [
        {
            "batch_size": 1,
            "pages_20": "60s",
            "pages_100": "300s",
            "speedup": "1.0x (baseline)",
            "memory": "~2GB",
        },
        {
            "batch_size": 4,
            "pages_20": "22s",
            "pages_100": "110s",
            "speedup": "2.7x",
            "memory": "~4GB",
        },
        {
            "batch_size": 8,
            "pages_20": "14s",
            "pages_100": "70s",
            "speedup": "4.3x",
            "memory": "~6GB",
        },
        {
            "batch_size": 16,
            "pages_20": "10s",
            "pages_100": "50s",
            "speedup": "6.0x",
            "memory": "~10GB",
        },
    ]
    
    print("\nExpected Performance (GPU: RTX 3080, OCR mode: small):")
    print("-" * 70)
    print(f"{'Batch':>6} | {'20 pages':>10} | {'100 pages':>11} | {'Speedup':>12} | {'Memory':>8}")
    print("-" * 70)
    
    for data in performance_data:
        print(f"{data['batch_size']:>6} | {data['pages_20']:>10} | {data['pages_100']:>11} | "
              f"{data['speedup']:>12} | {data['memory']:>8}")
    
    print("-" * 70)
    print("\nKey Insights:")
    print("  • Batch size 4-8 offers best balance for most use cases")
    print("  • Diminishing returns beyond batch_size=16")
    print("  • Memory usage scales linearly with batch size")
    print("  • Actual performance depends on GPU, CPU, and document complexity")


async def example_real_world_usage():
    """
    Example 7: Real-world usage patterns
    """
    print("\n" + "=" * 70)
    print("Example 7: Real-World Usage Patterns")
    print("=" * 70)
    
    print("\n📄 Scenario 1: Invoice Processing (2-5 pages)")
    print("-" * 70)
    config1 = DeepCompressConfig(
        ocr_batch_size=5,      # Process all pages at once
        ocr_mode="small",      # Simple documents
        ocr_max_new_tokens=512,  # Invoices are short
    )
    print(f"  Batch Size: {config1.ocr_batch_size}")
    print(f"  Rationale: Small documents, process all pages concurrently")
    print(f"  Expected Time: 5-10 seconds per document")
    
    print("\n📚 Scenario 2: Research Paper (10-30 pages)")
    print("-" * 70)
    config2 = DeepCompressConfig(
        ocr_batch_size=8,      # Default batch size
        ocr_mode="base",       # More detail needed
        ocr_max_new_tokens=2048,  # Dense text
    )
    print(f"  Batch Size: {config2.ocr_batch_size}")
    print(f"  Rationale: Standard size, balanced approach")
    print(f"  Expected Time: 30-60 seconds per document")
    
    print("\n📖 Scenario 3: Book Processing (200+ pages)")
    print("-" * 70)
    config3 = DeepCompressConfig(
        ocr_batch_size=16,     # Large batches for efficiency
        ocr_mode="small",      # Speed over quality
        ocr_max_new_tokens=1024,
        gpu_memory_fraction=0.9,  # Use most of GPU
    )
    print(f"  Batch Size: {config3.ocr_batch_size}")
    print(f"  Rationale: Large document, maximize throughput")
    print(f"  Expected Time: 5-10 minutes per book")
    
    print("\n🏢 Scenario 4: Batch Processing Service (1000s of documents)")
    print("-" * 70)
    config4 = DeepCompressConfig(
        ocr_batch_size=8,      # Moderate for stability
        ocr_mode="small",      # Consistent performance
        ocr_max_new_tokens=1024,
        gpu_memory_fraction=0.8,  # Leave headroom
    )
    print(f"  Batch Size: {config4.ocr_batch_size}")
    print(f"  Rationale: Stability over speed, predictable memory usage")
    print(f"  Expected Throughput: 50-100 documents/hour")


async def main():
    """Run all examples"""
    print("\n📊 Batch Processing Configuration Guide\n")
    
    await example_sequential()
    await example_small_batches()
    await example_medium_batches()
    await example_large_batches()
    await example_adaptive()
    await example_performance_comparison()
    await example_real_world_usage()
    
    print("\n" + "=" * 70)
    print("🎯 Quick Reference")
    print("=" * 70)
    print("""
Batch Size Recommendations:

┌─────────────┬──────────────┬────────────┬──────────────┐
│ Batch Size  │ Use Case     │ Memory     │ Speedup      │
├─────────────┼──────────────┼────────────┼──────────────┤
│ 1           │ Low memory   │ Very Low   │ 1.0x         │
│ 2-4         │ Balanced     │ Low-Mod    │ 2-3x         │
│ 8 (default) │ Standard     │ Moderate   │ 4-5x         │
│ 16          │ High perf    │ High       │ 6-8x         │
│ 32+         │ Massive docs │ Very High  │ 8-10x        │
└─────────────┴──────────────┴────────────┴──────────────┘

Configuration Examples:

# Low memory / CPU
ocr_batch_size=1

# Balanced (recommended)
ocr_batch_size=4

# Standard GPU
ocr_batch_size=8  # Default

# High-end GPU
ocr_batch_size=16

# Environment variable
export OCR_BATCH_SIZE=8

Tips:
  1. Start with batch_size=4 and increase if you have memory
  2. Monitor GPU memory usage (nvidia-smi)
  3. Reduce batch size if you get OOM errors
  4. Larger batches aren't always faster (overhead increases)
  5. Consider document size when choosing batch size

For more information:
  - docs/OCR_HALLUCINATION_FIX.md
  - examples/prevent_ocr_hallucination.py
    """)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

