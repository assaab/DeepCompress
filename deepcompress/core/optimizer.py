"""
D-TOON (Document-TOON) optimizer for token compression.
"""

import orjson

from deepcompress.exceptions import ConfigurationError
from deepcompress.models.document import ExtractedDocument


class DTOONOptimizer:
    """
    Document text optimizer for D-TOON compression formats.

    Raw mode keeps clean OCR text. Structured and RAG modes use an LLM client
    to compress the document into page-referenced semantic output.
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

    def optimize(self, document: ExtractedDocument, mode: str = "raw") -> str:
        """
        Convert ExtractedDocument to optimized text format.

        Args:
            document: ExtractedDocument to optimize
            mode: D-TOON output mode. Only raw mode is synchronous.

        Returns:
            Optimized text string with document content

        Example:
            >>> doc = ExtractedDocument(...)
            >>> optimizer = DTOONOptimizer()
            >>> optimized = optimizer.optimize(doc)
        """
        if mode != "raw":
            raise ConfigurationError(
                "D-TOON structured and rag modes require async LLM optimization"
            )
        return self._optimize_raw(document)

    async def optimize_async(
        self,
        document: ExtractedDocument,
        mode: str = "raw",
        llm_client=None,
    ) -> str:
        """
        Convert ExtractedDocument to the selected D-TOON mode.

        Structured and RAG modes require an initialized-compatible LLM client
        with a query(context, question, system_prompt) coroutine.
        """
        if mode == "raw":
            return self._optimize_raw(document)
        if mode == "structured":
            return await self._optimize_structured(document, llm_client)
        if mode == "rag":
            return await self._optimize_rag(document, llm_client)

        raise ConfigurationError(f"Unsupported D-TOON mode: {mode}")

    def _optimize_raw(self, document: ExtractedDocument) -> str:
        """Build clean page-referenced OCR text."""
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

    async def _optimize_structured(self, document: ExtractedDocument, llm_client) -> str:
        if llm_client is None:
            raise ConfigurationError("D-TOON structured mode requires an LLM client")

        response = await llm_client.query(
            context=self._page_referenced_context(document),
            question=(
                "Extract a compact structured D-TOON representation with: document title, "
                "sections, tables, key-value fields, dates, amounts, names, entities, and "
                "page references. Preserve exact values and cite source pages as [pN]."
            ),
            system_prompt=(
                "You convert OCR text into concise enterprise document structure. "
                "Return only the structured D-TOON output, with no explanatory prose."
            ),
        )
        return response.text.strip()

    async def _optimize_rag(self, document: ExtractedDocument, llm_client) -> str:
        if llm_client is None:
            raise ConfigurationError("D-TOON rag mode requires an LLM client")

        response = await llm_client.query(
            context=self._page_referenced_context(document),
            question=(
                "Create compact RAG chunks from this OCR text. Each chunk must start with "
                "a page citation like [p3], preserve exact values, and compress related "
                "facts into short retrieval-ready statements."
            ),
            system_prompt=(
                "You create concise page-cited RAG chunks from OCR text. "
                "Return only chunks, one per line, with no explanatory prose."
            ),
        )
        return response.text.strip()

    def _page_referenced_context(self, document: ExtractedDocument) -> str:
        lines = [
            f"Document ID: {document.document_id}",
            f"Total Pages: {document.page_count}",
            "",
        ]
        for page in document.pages:
            if page.raw_text and page.raw_text.strip():
                lines.append(f"[p{page.page_number}] {page.raw_text.strip()}")
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

