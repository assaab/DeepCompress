"""
D-TOON (Document-TOON) optimizer for token compression.
"""

import orjson

from deepcompress.models.document import ExtractedDocument


class DTOONOptimizer:
    """
    Document text optimizer for clean text formatting.

    Converts extracted OCR text to clean, readable format for LLM consumption.
    Focuses on presenting raw text content without complex structured data.
    """

    def __init__(
        self,
        include_bbox: bool = False,
        include_confidence: bool = False,
        min_confidence: float = 0.0,
    ) -> None:
        """
        Initialize D-TOON optimizer.

        Args:
            include_bbox: Whether to include bounding boxes
            include_confidence: Whether to include confidence scores
            min_confidence: Minimum confidence threshold for entities/tables
        """
        self.include_bbox = include_bbox
        self.include_confidence = include_confidence
        self.min_confidence = min_confidence

    def optimize(self, document: ExtractedDocument) -> str:
        """
        Convert ExtractedDocument to optimized text format.

        Args:
            document: ExtractedDocument to optimize

        Returns:
            Optimized text string with document content

        Example:
            >>> doc = ExtractedDocument(...)
            >>> optimizer = DTOONOptimizer()
            >>> optimized = optimizer.optimize(doc)
        """
        lines = []

        lines.append(f"Document ID: {document.document_id}")
        lines.append(f"Total Pages: {document.page_count}")
        lines.append("")

        for page in document.pages:
            if page.raw_text and page.raw_text.strip():
                raw_text = page.raw_text.strip()
                lines.append(f"=== Page {page.page_number} ===")
                lines.append(raw_text)
                lines.append("")

        return "\n".join(lines)


    def to_json(self, document: ExtractedDocument) -> str:
        """
        Convert ExtractedDocument to compact JSON (baseline comparison).

        Args:
            document: ExtractedDocument to convert

        Returns:
            Compact JSON string
        """
        data = document.model_dump(mode="json")
        return orjson.dumps(data).decode("utf-8")

    def calculate_compression_ratio(
        self,
        document: ExtractedDocument,
    ) -> tuple[int, int, float]:
        """
        Calculate token compression ratio.

        Args:
            document: ExtractedDocument

        Returns:
            Tuple of (json_tokens, toon_tokens, compression_ratio)
        """
        json_str = self.to_json(document)
        toon_str = self.optimize(document)

        json_tokens = self._estimate_tokens(json_str)
        toon_tokens = self._estimate_tokens(toon_str)

        compression_ratio = json_tokens / toon_tokens if toon_tokens > 0 else 1.0

        return json_tokens, toon_tokens, compression_ratio

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 4 chars per token).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

