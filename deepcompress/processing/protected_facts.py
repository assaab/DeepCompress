"""
Protected exact-value extraction for enterprise document compression.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from deepcompress.models.document import ExtractedDocument


FactMap = dict[str, list[str]]


FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "invoice_numbers": (
        r"\b(?:invoice|inv)\s*(?:number|no\.?|#|id)?\s*[:#-]?\s*([A-Z]{2,}-?\d{3,})\b",
    ),
    "ids": (
        r"\b[A-Z]{2,}(?:[-_]\d{3,})+\b",
        r"\b(?:id|identifier)\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
        r"\bapplication\s*(?:id|number|no\.?|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
    ),
    "emails": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ),
    "phones": (
        r"(?<!\d)(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
    ),
    "dates": (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+(?:19|20)\d{2}\b",
    ),
    "amounts": (
        r"(?<!\w)(?:USD\s*)?\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*/\s*(?:month|year|yr|mo))?",
        r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars)\b",
    ),
    "percentages": (
        r"\b\d+(?:\.\d+)?\s?%",
    ),
    "account_numbers": (
        r"\b(?:account|acct)\s*(?:number|no\.?|#)?\s*[:#-]?\s*([A-Z0-9-]{6,})\b",
    ),
    "names": (
        r"\b(?:applicant|borrower|customer|client|employee|vendor|party)[ \t]*[:#-][ \t]*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})\b",
        r"\b(?:name)[ \t]*[:#-][ \t]*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})\b",
    ),
}

LINE_FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "legal_clauses": (
        r"\b(?:shall|must|confidential|liability|indemnif|termination|governed by|warranty|breach)\b",
    ),
    "table_totals": (
        r"\btotal\b",
    ),
    "contract_terms": (
        r"\b(?:term|effective date|renewal|payment terms?|net \d+|due within|expires?|expiration)\b",
    ),
}


def extract_protected_facts(document_or_text: ExtractedDocument | str) -> FactMap:
    """Extract exact values that should survive compression."""
    facts_with_pages = extract_protected_facts_with_pages(document_or_text)
    return {
        category: [item["value"] for item in items]
        for category, items in facts_with_pages.items()
        if items
    }


def extract_protected_facts_with_pages(
    document_or_text: ExtractedDocument | str,
) -> dict[str, list[dict[str, Any]]]:
    """Extract protected facts with source page metadata when available."""
    pages = list(_iter_pages(document_or_text))
    found: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, set[str]] = {}

    for page_number, text in pages:
        for category, patterns in FACT_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = _match_value(match)
                    _add_fact(found, seen, category, value, page_number)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for category, patterns in LINE_FACT_PATTERNS.items():
                if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
                    _add_fact(found, seen, category, line, page_number)

    return {category: items for category, items in found.items() if items}


def append_missing_protected_facts(
    compressed_text: str,
    facts: dict[str, list[dict[str, Any]]] | FactMap,
) -> str:
    """Append protected facts that are absent from compressed text."""
    lines = []
    for category, items in facts.items():
        for item in items:
            value = item["value"] if isinstance(item, dict) else item
            if value.lower() in compressed_text.lower():
                continue
            page = item.get("page") if isinstance(item, dict) else None
            citation = f"[p{page}] " if page else ""
            lines.append(f"- {category}: {citation}{value}")

    if not lines:
        return compressed_text

    separator = "\n\n" if compressed_text.strip() else ""
    return f"{compressed_text.rstrip()}{separator}Protected facts:\n" + "\n".join(lines)


def _iter_pages(document_or_text: ExtractedDocument | str) -> Iterable[tuple[int | None, str]]:
    if isinstance(document_or_text, ExtractedDocument):
        for page in document_or_text.pages:
            yield page.page_number, page.raw_text or ""
        return

    yield None, document_or_text


def _match_value(match: re.Match[str]) -> str:
    for group in match.groups():
        if group:
            return _normalize_value(group)
    return _normalize_value(match.group(0))


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n:;,."))


def _add_fact(
    found: dict[str, list[dict[str, Any]]],
    seen: dict[str, set[str]],
    category: str,
    value: str,
    page: int | None,
) -> None:
    if not value:
        return

    key = value.lower()
    seen.setdefault(category, set())
    if key in seen[category]:
        return

    seen[category].add(key)
    found.setdefault(category, []).append({"value": value, "page": page})
