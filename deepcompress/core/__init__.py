"""
Core EDC processing modules.
"""

from deepcompress.core.compressor import DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.extractor import OCRExtractor
from deepcompress.core.optimizer import DTOONOptimizer

__all__ = [
    "DocumentCompressor",
    "DeepCompressConfig",
    "OCRExtractor",
    "DTOONOptimizer",
]

