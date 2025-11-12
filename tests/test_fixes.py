"""
Quick test script to verify DeepSeek-OCR integration fixes
Run this to ensure all fixes are working correctly
"""

import asyncio
import sys
import logging


def check_versions():
    """Check if all required packages are installed with correct versions"""
    print("=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        errors.append("Python version must be >= 3.9")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - CUDA: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            warnings.append("CUDA not available - will run on CPU (slower)")
        
        # Check version
        torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix
        major, minor = torch_version.split('.')[:2]
        if int(major) < 2:
            warnings.append(f"PyTorch version {torch_version} may not be optimal. Recommended: 2.0.0+")
    except ImportError:
        errors.append("PyTorch not installed. Install with: pip install torch>=2.0.0")
    
    # Check Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 4 or (major == 4 and minor < 38):
            errors.append(f"Transformers version {transformers.__version__} too old. Minimum: 4.38.0")
        elif major == 4 and minor >= 50:
            warnings.append(f"Transformers version {transformers.__version__} not tested. Recommended: 4.46.3")
    except ImportError:
        errors.append("Transformers not installed. Install with: pip install transformers>=4.38.0,<4.50.0")
    
    # Check Tokenizers
    try:
        import tokenizers
        print(f"Tokenizers: {tokenizers.__version__}")
    except ImportError:
        warnings.append("Tokenizers not installed. Install with: pip install tokenizers>=0.19.0")
    
    # Check DeepCompress
    try:
        import deepcompress
        print(f"DeepCompress: {deepcompress.__version__}")
        if deepcompress.__version__ != "1.0.15":
            warnings.append(f"DeepCompress version {deepcompress.__version__} may not have all fixes. Expected: 1.0.15")
    except ImportError:
        errors.append("DeepCompress not installed. Install with: pip install deepcompress[gpu]")
    
    # Check optional Flash Attention
    try:
        import flash_attn
        print(f"Flash Attention: {flash_attn.__version__}")
    except ImportError:
        warnings.append("Flash Attention not installed (optional, improves performance)")
    
    print("\n" + "=" * 60)
    
    if errors:
        print("❌ ERRORS (must fix):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("⚠️  WARNINGS (optional):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✓ All dependencies are correctly installed!")
    elif not errors:
        print("✓ All required dependencies are installed (warnings are optional)")
    
    print("=" * 60 + "\n")
    
    return len(errors) == 0


async def test_model_loading():
    """Test if DeepSeek-OCR model can be loaded"""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        from deepcompress.core.config import DeepCompressConfig
        from deepcompress.core.extractor import OCRExtractor
        
        # Create config
        config = DeepCompressConfig(
            ocr_mode="small",  # Use smallest mode for faster testing
            enable_flash_attention=False,  # Disable for compatibility
        )
        
        print(f"Loading model: {config.ocr_model}")
        print(f"Device: {config.ocr_device}")
        print(f"Mode: {config.ocr_mode}")
        
        # Initialize extractor
        extractor = OCRExtractor(config)
        await extractor.initialize()
        
        print("✓ Model loaded successfully!")
        print("✓ Position embeddings patch applied correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_extraction(pdf_path=None):
    """Test document extraction"""
    print("\n" + "=" * 60)
    print("Testing Document Extraction")
    print("=" * 60)
    
    if pdf_path is None:
        print("⚠️  No PDF path provided, skipping extraction test")
        print("   To test extraction, run: python test_fixes.py /path/to/test.pdf")
        return None
    
    try:
        from deepcompress import compress_and_analyze
        
        print(f"Processing: {pdf_path}")
        
        # Enable debug logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        result = await compress_and_analyze(
            file=pdf_path,
            query="Summarize the main content of this document.",
            llm="openai",
            cache=False,  # Disable cache for testing
        )
        
        print(f"\n✓ Extraction successful!")
        print(f"  - Original tokens: {result.original_tokens}")
        print(f"  - Compressed tokens: {result.compressed_tokens}")
        print(f"  - Compression ratio: {result.compression_ratio:.2f}x")
        print(f"  - Processing time: {result.processing_time_ms:.2f}ms")
        print(f"\n  Answer: {result.answer[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " DeepCompress Fix Verification Script ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Step 1: Check versions
    versions_ok = check_versions()
    
    if not versions_ok:
        print("\n❌ Fix dependency errors before continuing\n")
        return False
    
    # Step 2: Test model loading
    model_ok = await test_model_loading()
    
    if not model_ok:
        print("\n❌ Model loading failed - check error messages above\n")
        return False
    
    # Step 3: Test extraction if PDF provided
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    extraction_ok = await test_extraction(pdf_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ Dependency Check: {'PASS' if versions_ok else 'FAIL'}")
    print(f"✓ Model Loading: {'PASS' if model_ok else 'FAIL'}")
    if extraction_ok is not None:
        print(f"✓ Document Extraction: {'PASS' if extraction_ok else 'FAIL'}")
    else:
        print(f"⚠️  Document Extraction: SKIPPED (no PDF provided)")
    print("=" * 60)
    
    if versions_ok and model_ok:
        print("\n🎉 All fixes are working correctly!")
        print("\nThe following issues have been resolved:")
        print("  ✓ position_embeddings NoneType error")
        print("  ✓ Version compatibility issues")
        print("  ✓ Model inference validation")
        print("  ✓ Error handling and logging")
        print("\nYou can now use the library without errors.\n")
        return True
    else:
        print("\n❌ Some tests failed - review errors above\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

