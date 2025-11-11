"""
Example: Preventing OCR Hallucinations

This example demonstrates how to configure DeepCompress to prevent
repetitive/hallucinatory OCR output that can cause infinite generation.
"""

import asyncio
from deepcompress import DeepCompressConfig, DocumentCompressor


async def example_strict_generation():
    """
    Example 1: Strict generation limits (best for preventing hallucinations)
    
    Use this configuration when:
    - You encounter repetitive output
    - Processing takes too long
    - Output contains garbage/hallucinated text
    """
    print("=" * 70)
    print("Example 1: Strict Generation Limits")
    print("=" * 70)
    
    config = DeepCompressConfig(
        # OCR Settings
        ocr_mode="small",           # 100 vision tokens per page
        ocr_device="cuda:0",        # Use GPU
        
        # CRITICAL: Generation limits to prevent hallucinations
        ocr_max_new_tokens=1024,    # Limit to 1024 tokens per page
        ocr_temperature=0.0,        # Greedy decoding (most deterministic)
        ocr_repetition_penalty=1.5, # Strong penalty against repetition
        
        # Optional: Disable features that require API keys
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  OCR Mode:             {config.ocr_mode}")
    print(f"  Max New Tokens:       {config.ocr_max_new_tokens}")
    print(f"  Temperature:          {config.ocr_temperature}")
    print(f"  Repetition Penalty:   {config.ocr_repetition_penalty}")
    
    # Example usage (replace with your actual file)
    # compressor = DocumentCompressor(config)
    # result = await compressor.compress("document.pdf")
    # print(f"\nCompressed to {result.compressed_tokens} tokens")
    
    print("\n✓ Configuration created successfully")
    print("  This config will prevent most hallucination issues.")


async def example_balanced():
    """
    Example 2: Balanced configuration (good for most documents)
    
    Use this configuration when:
    - Documents are well-formatted
    - You want balance between speed and completeness
    - Previous runs showed no hallucination issues
    """
    print("\n" + "=" * 70)
    print("Example 2: Balanced Configuration")
    print("=" * 70)
    
    config = DeepCompressConfig(
        # OCR Settings
        ocr_mode="base",            # 200 vision tokens per page (more detail)
        ocr_device="cuda:0",
        
        # Balanced generation settings
        ocr_max_new_tokens=2048,    # Default: allows longer output
        ocr_temperature=0.1,        # Slight randomness for better quality
        ocr_repetition_penalty=1.2, # Moderate anti-repetition
        
        # Optional
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  OCR Mode:             {config.ocr_mode}")
    print(f"  Max New Tokens:       {config.ocr_max_new_tokens}")
    print(f"  Temperature:          {config.ocr_temperature}")
    print(f"  Repetition Penalty:   {config.ocr_repetition_penalty}")
    
    print("\n✓ Configuration created successfully")
    print("  This config balances speed, quality, and safety.")


async def example_permissive():
    """
    Example 3: Permissive configuration (for dense technical documents)
    
    Use this configuration when:
    - Documents have very dense text
    - Pages contain a lot of content
    - Previous runs showed truncated output
    """
    print("\n" + "=" * 70)
    print("Example 3: Permissive Configuration (Dense Documents)")
    print("=" * 70)
    
    config = DeepCompressConfig(
        # OCR Settings
        ocr_mode="large",           # 400 vision tokens per page (maximum detail)
        ocr_device="cuda:0",
        
        # Permissive generation settings
        ocr_max_new_tokens=4096,    # Allow long output
        ocr_temperature=0.1,        # Low but non-zero for quality
        ocr_repetition_penalty=1.1, # Light anti-repetition
        
        # Optional
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  OCR Mode:             {config.ocr_mode}")
    print(f"  Max New Tokens:       {config.ocr_max_new_tokens}")
    print(f"  Temperature:          {config.ocr_temperature}")
    print(f"  Repetition Penalty:   {config.ocr_repetition_penalty}")
    
    print("\n✓ Configuration created successfully")
    print("  This config allows maximum content extraction.")


async def example_debugging():
    """
    Example 4: Debug configuration (when things go wrong)
    
    Use this configuration when:
    - You need to diagnose issues
    - You want to see what's happening
    - Output quality is unexpected
    """
    print("\n" + "=" * 70)
    print("Example 4: Debug Configuration")
    print("=" * 70)
    
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    config = DeepCompressConfig(
        # OCR Settings
        ocr_mode="small",           # Start small for faster debugging
        ocr_device="cuda:0",
        
        # Very strict for debugging
        ocr_max_new_tokens=512,     # Very limited output
        ocr_temperature=0.0,        # Completely deterministic
        ocr_repetition_penalty=1.8, # Very strong anti-repetition
        
        # Logging
        log_level="DEBUG",          # Verbose logging
        
        # Optional
        use_vector_db=False,
        vector_db_provider='none',
    )
    
    print("\nConfiguration:")
    print(f"  OCR Mode:             {config.ocr_mode}")
    print(f"  Max New Tokens:       {config.ocr_max_new_tokens}")
    print(f"  Temperature:          {config.ocr_temperature}")
    print(f"  Repetition Penalty:   {config.ocr_repetition_penalty}")
    print(f"  Log Level:            {config.log_level}")
    
    print("\n✓ Configuration created successfully")
    print("  Debug logging enabled - you'll see detailed output.")


async def example_environment_vars():
    """
    Example 5: Using environment variables
    
    Use this approach when:
    - You want to configure without code changes
    - Different environments need different settings
    - You're deploying to production
    """
    print("\n" + "=" * 70)
    print("Example 5: Environment Variables")
    print("=" * 70)
    
    print("\nCreate a .env file with:")
    print("""
# OCR Configuration
OCR_MODE=small
OCR_DEVICE=cuda:0
OCR_MAX_NEW_TOKENS=1024
OCR_TEMPERATURE=0.0
OCR_REPETITION_PENALTY=1.5

# Optional: Other settings
USE_VECTOR_DB=False
VECTOR_DB_PROVIDER=none
    """)
    
    # Then in your code:
    config = DeepCompressConfig()  # Automatically loads from .env
    
    print("\nConfiguration loaded from environment:")
    print(f"  OCR Mode:             {config.ocr_mode}")
    print(f"  Max New Tokens:       {config.ocr_max_new_tokens}")
    print(f"  Temperature:          {config.ocr_temperature}")
    print(f"  Repetition Penalty:   {config.ocr_repetition_penalty}")
    
    print("\n✓ Configuration loaded from environment variables")
    print("  This approach is best for production deployments.")


async def main():
    """Run all examples"""
    print("\n🔬 OCR Hallucination Prevention Examples\n")
    
    await example_strict_generation()
    await example_balanced()
    await example_permissive()
    await example_debugging()
    await example_environment_vars()
    
    print("\n" + "=" * 70)
    print("📚 Summary")
    print("=" * 70)
    print("""
Key Takeaways:

1. Use STRICT settings if you encounter repetitive output:
   - ocr_max_new_tokens=1024
   - ocr_temperature=0.0
   - ocr_repetition_penalty=1.5

2. Use BALANCED settings for most documents:
   - ocr_max_new_tokens=2048
   - ocr_temperature=0.1
   - ocr_repetition_penalty=1.2

3. Use PERMISSIVE settings for dense documents:
   - ocr_max_new_tokens=4096
   - ocr_temperature=0.1
   - ocr_repetition_penalty=1.1

4. Always set these parameters - don't rely on defaults if you've
   experienced hallucination issues.

5. Monitor logs for truncation warnings - they indicate when the
   safety mechanisms have activated.

For more information, see: docs/OCR_HALLUCINATION_FIX.md
    """)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

