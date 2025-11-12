"""
Fixed test script for DeepCompress with DeepSeek-OCR integration.

This demonstrates the corrected usage after fixing the tokenizer/processor issues.
Includes comprehensive error handling and automatic fallback mechanisms.
"""
import asyncio
import os
import sys
from deepcompress import compress_and_analyze, DeepCompressConfig


async def run_simple_query(file_path, query="Give a 1-sentence summary of the document.", config=None):
    """
    Run a simple compression and analysis query on a document.
    
    Args:
        file_path: Path to the PDF or image file
        query: Question to ask about the document
        config: Optional DeepCompressConfig instance
    """
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        return None
    
    # Use provided config or create default one
    if config is None:
        config = create_optimal_config(openai_api_key)
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"Query: {query}")
    print(f"Configuration:")
    print(f"  OCR Mode: {config.ocr_mode}")
    print(f"  Device: {config.ocr_device}")
    print(f"  Flash Attention: {config.enable_flash_attention}")
    print(f"  BFloat16: {config.use_bfloat16}")
    print(f"{'='*60}\n")
    
    try:
        result = await compress_and_analyze(
            file=file_path,
            query=query,
            llm="openai",
            config=config,
            cache=False,  # Disable caching
            scrub_pii=True,  # Enable PII scrubbing for security
        )
        
        print(f"\n✅ Success! Results for {file_path}")
        print(f"\nAnswer: {result.answer}")
        print(f"\nCompression Stats:")
        print(f"  Original tokens: {result.original_tokens}")
        print(f"  Compressed tokens: {result.compressed_tokens}")
        print(f"  Tokens saved: {result.tokens_saved}")
        print(f"  Compression ratio: {result.compression_ratio:.2f}x")
        print(f"  Cost saved (USD): ${result.cost_saved_usd:.4f}")
        print(f"  Processing time: {result.processing_time_ms:.0f}ms")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error processing {file_path}: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. If you see 'position_embeddings' error:")
        print("     - Update transformers: pip install 'transformers>=4.38.0,<4.48.0'")
        print("  2. If you see Flash Attention errors:")
        print("     - Install flash-attn: pip install flash-attn --no-build-isolation")
        print("     - Or disable it: config.enable_flash_attention=False")
        print("  3. If you see GPU/CUDA errors:")
        print("     - Use CPU: config.ocr_device='cpu'")
        print("  4. For memory issues:")
        print("     - Reduce GPU memory fraction: config.gpu_memory_fraction=0.7")
        print("     - Use smaller mode: config.ocr_mode='small'")
        
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        
        return None


def create_optimal_config(api_key):
    """
    Create optimal configuration with automatic fallbacks.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        DeepCompressConfig with optimal settings
    """
    import torch
    
    # Determine optimal device
    if torch.cuda.is_available():
        device = "cuda:0"
        print("✓ CUDA detected, using GPU acceleration")
    else:
        device = "cpu"
        print("⚠ CUDA not available, using CPU (slower)")
    
    # Check for Flash Attention support
    enable_flash_attention = False
    if device.startswith("cuda"):
        try:
            import flash_attn
            enable_flash_attention = True
            print("✓ Flash Attention available")
        except ImportError:
            print("⚠ Flash Attention not available, using eager attention")
    
    # Configure DeepCompress with optimal settings
    config = DeepCompressConfig(
        llm_api_key=api_key,
        vector_db_provider="none",  # Disable vector DB
        cache_enabled=False,  # Disable Redis cache
        ocr_mode="small",  # Options: "small" (640x640), "base" (1024x1024), "large" (1280x1280)
        ocr_device=device,
        ocr_model_revision="main",  # Pin to specific revision for stability
        use_bfloat16=True if device.startswith("cuda") else False,
        enable_flash_attention=enable_flash_attention,
        gpu_memory_fraction=0.9,  # Use 90% of GPU memory
    )
    
    return config


async def main():
    """Main function to test multiple documents."""
    
    print("\n" + "="*60)
    print("DeepCompress Test Suite - DeepSeek-OCR Integration")
    print("="*60)
    
    # Create optimal configuration once
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("\n❌ OPENAI_API_KEY not set. Cannot run tests.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    config = create_optimal_config(openai_api_key)
    
    # Test with your PDF files
    test_files = [
        ("arxiv_transformer.pdf", "What is the main contribution of this paper?"),
        ("nasa_wbs.pdf", "Summarize the key components of the work breakdown structure."),
    ]
    
    successful = 0
    failed = 0
    
    for file_path, query in test_files:
        if os.path.exists(file_path):
            result = await run_simple_query(file_path, query, config=config)
            if result is not None:
                successful += 1
            else:
                failed += 1
        else:
            print(f"\n⚠️  File not found: {file_path}")
            print(f"   Skipping this test...")
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Total: {successful + failed}")
    print(f"{'='*60}\n")
    
    if failed == 0 and successful > 0:
        print("🎉 All tests passed!")
    elif successful > 0:
        print("⚠️  Some tests passed, but there were failures.")
    else:
        print("❌ All tests failed. Please check the configuration and error messages.")


def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("  python test_deepcompress_fixed.py [--debug]")
    print("\nOptions:")
    print("  --debug    Enable detailed error tracebacks")
    print("\nEnvironment Variables:")
    print("  OPENAI_API_KEY    Required: Your OpenAI API key")
    print("\nExamples:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  python test_deepcompress_fixed.py")
    print("  python test_deepcompress_fixed.py --debug")
    print()


if __name__ == "__main__":
    # Print usage if --help is passed
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        sys.exit(0)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print("\nDeepCompress Test Script")
    print(f"Python version: {sys.version.split()[0]}")
    
    # Notebook-safe asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
        print("✓ nest_asyncio applied (notebook-safe)")
    except ImportError:
        pass
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)

