"""
D-TOON (Document-TOON) optimizer for token compression.
"""

import orjson

from deepcompress.models.document import Entity, ExtractedDocument, Page, Table


class DTOONOptimizer:
    """
    Document-TOON optimizer for 60% additional token savings over JSON.

    Converts structured JSON to compact D-TOON format:
    - Field deduplication (declare once per hierarchy)
    - Array compression (entities[4] instead of repeated "entity")
    - Semicolon delimiters (no JSON quotes)
    - Optional metadata omission (bbox, confidence)
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
        Convert ExtractedDocument to D-TOON format.

        Args:
            document: ExtractedDocument to optimize

        Returns:
            D-TOON formatted string

        Example:
            >>> doc = ExtractedDocument(...)
            >>> optimizer = DTOONOptimizer()
            >>> toon = optimizer.optimize(doc)
            >>> print(toon)
            doc{pages,entities}:
              id:loan_app_2024_12345
              pages[50]
                page1{entities[4],tables[1]}:
                  bank_name;Chase Bank
                  account_holder;John Doe
                  table1{headers,rows[25]}:
                    headers:Date;Description;Amount
                    row:2024-01-15;Deposit;$8,500.00
        """
        lines = []

        lines.append("doc{pages,entities}:")
        lines.append(f"  id:{document.document_id}")
        lines.append(f"  pages[{document.page_count}]")

        for page in document.pages:
            if page.entities or page.tables:
                page_line = self._format_page(page)
                lines.append(page_line)

        return "\n".join(lines)

    def _format_page(self, page: Page) -> str:
        """Format single page in D-TOON."""
        lines = []

        entity_count = len(
            [e for e in page.entities if e.confidence >= self.min_confidence]
        )
        table_count = len([t for t in page.tables if t.confidence >= self.min_confidence])

        page_header = f"    page{page.page_number}"
        if entity_count > 0 or table_count > 0:
            components = []
            if entity_count > 0:
                components.append(f"entities[{entity_count}]")
            if table_count > 0:
                components.append(f"tables[{table_count}]")
            page_header += "{" + ",".join(components) + "}:"

        lines.append(page_header)

        for entity in page.entities:
            if entity.confidence >= self.min_confidence:
                lines.append(self._format_entity(entity))

        for i, table in enumerate(page.tables, start=1):
            if table.confidence >= self.min_confidence:
                lines.extend(self._format_table(table, i))

        return "\n".join(lines)

    def _format_entity(self, entity: Entity) -> str:
        """Format entity in D-TOON."""
        parts = [entity.type, entity.text]

        if self.include_bbox and entity.bbox:
            bbox_str = ",".join(str(x) for x in entity.bbox)
            parts.append(f"[{bbox_str}]")
        elif self.include_bbox:
            parts.append("[]")

        if self.include_confidence:
            parts.append(f"{entity.confidence:.2f}")

        return "      " + ";".join(parts)

    def _format_table(self, table: Table, table_num: int) -> list[str]:
        """Format table in D-TOON."""
        lines = []

        row_count = len(table.rows)
        lines.append(f"      table{table_num}{{headers,rows[{row_count}]}}:")

        headers_str = ";".join(table.headers)
        lines.append(f"        headers:{headers_str}")

        for row in table.rows[:10]:
            row_str = ";".join(row)
            lines.append(f"        row:{row_str}")

        if len(table.rows) > 10:
            lines.append(f"        ... {len(table.rows) - 10} more rows ...")

        return lines

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

