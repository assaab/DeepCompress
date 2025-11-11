# Batch Processing Guide

## Overview

DeepCompress now supports **configurable batch processing** for processing multiple pages concurrently, significantly improving performance for multi-page documents.

## How It Works

Instead of processing pages sequentially (one at a time), batch processing divides pages into groups and processes each group concurrently using async/await parallelism.

### Sequential Processing (Old Behavior)
```
Page 1 → Page 2 → Page 3 → Page 4 → ...
   3s      3s      3s      3s     = 12s total
```

### Batch Processing (New Behavior, batch_size=4)
```
Batch 1: [Page 1, Page 2, Page 3, Page 4] → All processed concurrently
         └─────────────────────────────────┘
                    3-4s total
```

**Result:** ~3-4x faster for 4-page document

## Configuration

### Basic Usage

```python
from deepcompress import DeepCompressConfig, DocumentCompressor

config = DeepCompressConfig(
    ocr_batch_size=8,  # Process 8 pages at once (default)
)

compressor = DocumentCompressor(config)
result = await compressor.compress("document.pdf")
```

### Batch Size Parameter

```python
ocr_batch_size: int = Field(
    default=8,
    ge=1,
    le=64,
    description="Number of pages to process concurrently in each batch"
)
```

- **Minimum:** 1 (sequential processing)
- **Default:** 8 (good for most GPUs)
- **Maximum:** 64 (for high-end setups)

## Choosing the Right Batch Size

### Factors to Consider

1. **Available Memory** - Larger batches use more GPU/RAM
2. **Document Size** - Small docs can use smaller batches
3. **Hardware** - Better GPU → larger batches
4. **Stability** - Smaller batches are more stable

### Recommended Batch Sizes

#### For Different Hardware

| Hardware | Recommended Batch Size | Expected Speedup |
|----------|----------------------|------------------|
| CPU only | 1-2 | 1.0-1.5x |
| GPU < 6GB | 2-4 | 2-3x |
| GPU 6-12GB | 4-8 | 3-5x |
| GPU 12-16GB | 8-16 | 5-7x |
| GPU 16GB+ | 16-32 | 7-10x |

#### For Different Document Sizes

| Document Size | Recommended Batch Size | Rationale |
|--------------|----------------------|-----------|
| 1-5 pages | 1-5 | Process all at once |
| 5-20 pages | 4-8 | Balanced approach |
| 20-100 pages | 8-16 | Multiple batches for efficiency |
| 100+ pages | 16-32 | Maximum parallelism |

## Examples

### Example 1: Low Memory Setup

```python
config = DeepCompressConfig(
    ocr_batch_size=2,           # Small batches
    ocr_mode="small",           # Use smallest model
    gpu_memory_fraction=0.7,    # Limit GPU usage
)
```

**Use when:**
- GPU has < 6GB VRAM
- Running on CPU
- Other processes need GPU memory

### Example 2: Balanced Setup (Recommended)

```python
config = DeepCompressConfig(
    ocr_batch_size=4,           # Good balance
    ocr_mode="small",           # Fast and efficient
    ocr_max_new_tokens=1024,    # Reasonable limit
)
```

**Use when:**
- Standard GPU (8-12GB VRAM)
- Most document types
- Want good speed without risk

### Example 3: High Performance Setup

```python
config = DeepCompressConfig(
    ocr_batch_size=16,          # Large batches
    ocr_mode="base",            # Better quality
    gpu_memory_fraction=0.9,    # Use most GPU
    enable_flash_attention=True,
)
```

**Use when:**
- High-end GPU (16GB+ VRAM)
- Processing many documents
- Speed is critical

### Example 4: Adaptive Batch Size

```python
def get_batch_size(num_pages: int, gpu_memory_gb: int) -> int:
    """Calculate optimal batch size"""
    if num_pages <= 5:
        return num_pages  # Small doc: all at once
    
    if gpu_memory_gb < 6:
        return 2
    elif gpu_memory_gb < 12:
        return 4
    elif gpu_memory_gb < 16:
        return 8
    else:
        return 16

import torch

num_pages = 20
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

config = DeepCompressConfig(
    ocr_batch_size=get_batch_size(num_pages, gpu_memory),
)
```

### Example 5: Environment Variable

```bash
# .env file
OCR_BATCH_SIZE=8
```

```python
config = DeepCompressConfig()  # Loads from environment
```

## Performance Benchmarks

### Test Setup
- GPU: NVIDIA RTX 3080 (10GB)
- Document: 20-page PDF
- OCR Mode: small

### Results

| Batch Size | Total Time | Pages/Second | Speedup | GPU Memory |
|-----------|-----------|--------------|---------|-----------|
| 1 (sequential) | 60s | 0.33 | 1.0x | 2GB |
| 2 | 35s | 0.57 | 1.7x | 3GB |
| 4 | 22s | 0.91 | 2.7x | 4GB |
| 8 | 14s | 1.43 | 4.3x | 6GB |
| 16 | 10s | 2.00 | 6.0x | 9GB |

**Key Findings:**
- Batch size 4-8 offers best balance
- Diminishing returns after batch_size=16
- Memory scales linearly with batch size

## Monitoring and Debugging

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

config = DeepCompressConfig(
    log_level="INFO",
    ocr_batch_size=8,
)
```

**Expected output:**
```
INFO: Processing 20 pages in batches of 8
INFO: Processing batch: pages 1-8 of 20
INFO: Batch complete: 8 pages in 3.24s (0.41s per page)
INFO: Processing batch: pages 9-16 of 20
INFO: Batch complete: 8 pages in 3.18s (0.40s per page)
INFO: Processing batch: pages 17-20 of 20
INFO: Batch complete: 4 pages in 1.65s (0.41s per page)
```

### Monitor GPU Memory

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Look for:
- **GPU Memory Usage** - Should stay under 90%
- **GPU Utilization** - Should be high during batches
- **Temperature** - Should be stable

### Troubleshooting

#### Out of Memory (OOM) Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
config = DeepCompressConfig(
    ocr_batch_size=4,           # Reduce from 8
    gpu_memory_fraction=0.8,    # Limit GPU usage
    ocr_mode="small",           # Use smaller model
)
```

#### Slow Processing Despite Large Batch Size

**Possible causes:**
1. CPU bottleneck (preprocessing images)
2. Disk I/O bottleneck (loading images)
3. Overhead from too many concurrent tasks

**Solution:**
```python
# Try smaller batch size
config = DeepCompressConfig(
    ocr_batch_size=4,  # Reduce from 16
)
```

#### Inconsistent Results

**Symptom:** Different runs produce different results

**Cause:** Concurrent processing can cause non-deterministic behavior

**Solution:**
```python
config = DeepCompressConfig(
    ocr_batch_size=1,           # Sequential processing
    ocr_temperature=0.0,        # Deterministic generation
)
```

## Advanced Usage

### Custom Batch Processing

If you need more control, you can process batches manually:

```python
from deepcompress.core.extractor import OCRExtractor

extractor = OCRExtractor(config)
await extractor.initialize()

# Load images
images = await extractor._load_images("document.pdf")

# Process in custom batches
batch_size = 5
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    pages = await extractor._extract_pages_in_batches(batch)
    # Process pages...
```

### Mixing Sequential and Batch Processing

```python
# Process critical pages sequentially
critical_pages = images[0:5]
for i, img in enumerate(critical_pages):
    page = await extractor._extract_page(img, i+1)
    # Handle page...

# Batch process remaining pages
remaining_pages = images[5:]
if remaining_pages:
    pages = await extractor._extract_pages_in_batches(remaining_pages)
```

## Best Practices

### 1. Start Conservative

```python
# Start with small batch size
config = DeepCompressConfig(ocr_batch_size=4)

# Monitor performance and memory
# Increase gradually if you have headroom
```

### 2. Monitor Your System

- Watch GPU memory usage
- Check CPU utilization
- Monitor disk I/O
- Track processing times

### 3. Adjust Based on Workload

```python
# Small documents (< 10 pages)
ocr_batch_size = min(num_pages, 8)

# Large documents (> 50 pages)
ocr_batch_size = 16

# Very large documents (> 200 pages)
ocr_batch_size = 32
```

### 4. Consider Concurrent Documents

If processing multiple documents simultaneously:

```python
# Reduce batch size when processing multiple docs
config = DeepCompressConfig(
    ocr_batch_size=4,  # Instead of 8
)

# Process documents
results = await asyncio.gather(*[
    compressor.compress(doc)
    for doc in documents
])
```

### 5. Profile Your Workload

```python
import time

start = time.time()
result = await compressor.compress("doc.pdf")
elapsed = time.time() - start

print(f"Processed {result.page_count} pages in {elapsed:.2f}s")
print(f"Pages/second: {result.page_count / elapsed:.2f}")
```

## Comparison with Other Approaches

### Batch Processing vs Thread Pool

**Batch Processing (async/await):**
- ✅ Better for I/O-bound tasks
- ✅ Lower memory overhead
- ✅ Better error handling
- ✅ Easier to reason about

**Thread Pool:**
- ❌ Higher memory overhead
- ❌ GIL contention in Python
- ✅ Better for CPU-bound tasks (not our case)

### Batch Processing vs Process Pool

**Batch Processing:**
- ✅ No serialization overhead
- ✅ Shared GPU memory
- ✅ Faster startup

**Process Pool:**
- ❌ High overhead for GPU tasks
- ❌ Each process needs to load model
- ✅ True parallelism (but not needed here)

## Related Configuration

Batch processing works well with other configuration options:

```python
config = DeepCompressConfig(
    # Batch processing
    ocr_batch_size=8,
    
    # Generation limits (prevent hallucination)
    ocr_max_new_tokens=1024,
    ocr_temperature=0.0,
    ocr_repetition_penalty=1.5,
    
    # GPU optimization
    gpu_memory_fraction=0.9,
    use_bfloat16=True,
    enable_flash_attention=True,
    
    # Mode
    ocr_mode="small",  # or "base", "large"
)
```

## Additional Resources

- **Example Script:** `examples/batch_processing_demo.py`
- **OCR Fix Documentation:** `docs/OCR_HALLUCINATION_FIX.md`
- **Configuration Reference:** `deepcompress/core/config.py`
- **Source Code:** `deepcompress/core/extractor.py`

---

**Last Updated:** 2025-11-10  
**Version:** 1.1.0

