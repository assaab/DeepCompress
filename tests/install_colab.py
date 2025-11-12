"""
Optimized Installation Script for Google Colab
Run this in a Colab notebook cell

This script uses the 'colab' extra which:
- Uses Colab's pre-installed PyTorch and Transformers
- Skips flash-attn (requires compilation)
- Installs only necessary additional packages
"""

# Install deepcompress with Colab-optimized dependencies
import subprocess
import sys

print("Installing DeepCompress (Colab-optimized)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "--upgrade", "--quiet", "pip"
])

# Use colab extra for fast installation
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "deepcompress[colab,llm]", "--upgrade", "--quiet"
])

print("\nVerifying installation...")

# Verify installation
import deepcompress
print(f"✓ DeepCompress version: {deepcompress.__version__}")

# Check dependencies
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available - will run on CPU")
    # Check PyTorch version
    version_parts = torch.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 2:
        print(f"⚠ Warning: PyTorch version {torch.__version__} may not be compatible with DeepSeek-OCR")
        print(f"  Recommended: torch>=2.0.0,<2.7.0")
except ImportError as e:
    print(f"✗ PyTorch import error: {e}")

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    # Check if version is compatible with DeepSeek-OCR
    version_parts = transformers.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 4 or (major == 4 and minor < 38):
        print(f"⚠ Warning: Transformers version {transformers.__version__} may not be compatible with DeepSeek-OCR")
        print(f"  Recommended: transformers>=4.38.0,<4.50.0")
except ImportError as e:
    print(f"✗ Transformers import error: {e}")

try:
    import pillow
    print(f"✓ Pillow installed")
except ImportError:
    try:
        import PIL
        print(f"✓ PIL installed")
    except ImportError as e:
        print(f"✗ Pillow import error: {e}")

print("\n" + "="*50)
print("✓ Installation complete! Ready to use DeepCompress")
print("="*50)
print("\nQuick start:")
print("  from deepcompress import compress_and_analyze")
print("  result = await compress_and_analyze('document.pdf', 'Your query')")

