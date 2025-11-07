"""
OCR extraction using DeepSeek-OCR integration.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import GPUError, OCRError
from deepcompress.models.document import Entity, ExtractedDocument, Page, Table


class OCRExtractor:
    """
    DeepSeek-OCR integration for vision-based document extraction.

    Uses a 3B parameter vision-language model with:
    - SAM-base vision encoder
    - CLIP-large global attention
    - MoE decoder (64 experts, 6 active)
    - 16× compression of vision tokens
    """

    def __init__(self, config: DeepCompressConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._device: str = config.ocr_device

    async def initialize(self) -> None:
        """
        Initialize the OCR model and processor.

        Loads DeepSeek-OCR model onto GPU with bfloat16 precision.
        """
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.config.ocr_model,
                trust_remote_code=True,
            )

            self._model = AutoModel.from_pretrained(
                self.config.ocr_model,
                torch_dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float32,
                trust_remote_code=True,
            )

            if self._device.startswith("cuda"):
                self._model = self._model.to(self._device)
                if self.config.gpu_memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.gpu_memory_fraction,
                        device=int(self._device.split(":")[-1]),
                    )

            self._model.eval()

        except ImportError as e:
            raise OCRError(
                "Failed to import required libraries. Install with: pip install edc[gpu]",
                details={"error": str(e)},
            )
        except Exception as e:
            raise GPUError(
                "Failed to initialize OCR model on GPU",
                details={"device": self._device, "error": str(e)},
            )

    async def extract(
        self,
        file_path: str,
        document_id: str  or None = None,
    ) -> ExtractedDocument:
        """
        Extract document content using DeepSeek-OCR.

        Args:
            file_path: Path to document (PDF or image)
            document_id: Optional document ID (generated if None)

        Returns:
            ExtractedDocument with extracted entities, tables, and text

        Raises:
            OCRError: If extraction fails
            GPUError: If GPU operations fail
        """
        if self._model is None:
            await self.initialize()

        start_time = time.time()

        try:
            images = await self._load_images(file_path)

            if document_id is None:
                document_id = self._generate_document_id(file_path)

            pages = []
            for page_num, image in enumerate(images, start=1):
                page = await self._extract_page(image, page_num)
                pages.append(page)

            processing_time_ms = (time.time() - start_time) * 1000

            return ExtractedDocument(
                document_id=document_id,
                page_count=len(pages),
                mode=self.config.ocr_mode,
                pages=pages,
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "model": self.config.ocr_model,
                    "device": self._device,
                },
            )

        except Exception as e:
            raise OCRError(
                f"Failed to extract document: {file_path}",
                details={"error": str(e)},
            )

    async def _load_images(self, file_path: str) -> list[Any]:
        """
        Load images from PDF or image file.

        Args:
            file_path: Path to file

        Returns:
            List of PIL Images
        """
        from PIL import Image

        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path

                loop = asyncio.get_event_loop()
                images = await loop.run_in_executor(
                    None,
                    lambda: convert_from_path(
                        str(path),
                        dpi=300,
                        fmt="png",
                    ),
                )
                return images
            except ImportError:
                raise OCRError(
                    "pdf2image not installed. Install with: pip install edc[gpu]"
                )
        else:
            image = Image.open(path).convert("RGB")
            return [image]

    async def _extract_page(self, image: Any, page_number: int) -> Page:
        """
        Extract single page using DeepSeek-OCR.

        Args:
            image: PIL Image
            page_number: Page number (1-indexed)

        Returns:
            Page with extracted entities and tables
        """
        import torch

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )

        if self._device.startswith("cuda"):
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.vision_tokens_per_page,
                do_sample=False,
            )

        result = self._processor.decode(outputs[0], skip_special_tokens=True)

        entities, tables = self._parse_ocr_output(result)

        return Page(
            page_number=page_number,
            layout="multi_column",
            entities=entities,
            tables=tables,
            raw_text=result,
            metadata={"vision_tokens": len(outputs[0])},
        )

    def _parse_ocr_output(self, output: str) -> tuple[list[Entity], list[Table]]:
        """
        Parse DeepSeek-OCR JSON output into entities and tables.

        Args:
            output: JSON string from OCR model

        Returns:
            Tuple of (entities, tables)
        """
        import orjson

        try:
            data = orjson.loads(output)
        except Exception:
            data = {"entities": [], "tables": []}

        entities = []
        for ent_data in data.get("entities", []):
            entities.append(
                Entity(
                    type=ent_data.get("type", "unknown"),
                    text=ent_data.get("text", ""),
                    bbox=ent_data.get("bbox"),
                    confidence=ent_data.get("confidence", 1.0),
                )
            )

        tables = []
        for table_data in data.get("tables", []):
            tables.append(
                Table(
                    headers=table_data.get("headers", []),
                    rows=table_data.get("rows", []),
                    bbox=table_data.get("bbox"),
                    confidence=table_data.get("confidence", 1.0),
                )
            )

        return entities, tables

    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate unique document ID from file path.

        Args:
            file_path: Path to file

        Returns:
            Document ID (hash of file path)
        """
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    async def extract_batch(
        self,
        file_paths: list[str],
    ) -> list[ExtractedDocument]:
        """
        Extract multiple documents in batch.

        Args:
            file_paths: List of file paths

        Returns:
            List of ExtractedDocuments
        """
        tasks = [self.extract(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise OCRError(
                    f"Batch extraction failed for {file_paths[i]}",
                    details={"error": str(result)},
                )
            documents.append(result)

        return documents

