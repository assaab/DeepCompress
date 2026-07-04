"""Offline enterprise RAG demo using a local synthetic contract fixture."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.rag_benchmark import Chunk, answer_from_retrieval, retrieve
from deepcompress.models.document import ExtractedDocument, Page
from deepcompress.processing.protected_facts import (
    append_missing_protected_facts,
    extract_protected_facts,
    extract_protected_facts_with_pages,
)
from deepcompress.utils.cost import _calculate_llm_cost, _get_llm_pricing
from deepcompress.utils.token_counter import count_tokens


SAMPLE_CONTRACT = Path(__file__).with_name("sample_contract.txt")
QUESTION = {
    "id": "income",
    "document_id": "sample-contract",
    "question": "What is the applicant's total monthly income and where is it mentioned?",
    "expected_answer_terms": ["$20,200", "total monthly income"],
    "expected_values": ["$17,000/month", "$3,200/month", "$20,200"],
    "expected_pages": [2],
}


def load_sample_contract() -> ExtractedDocument:
    """Load a page-marked text fixture as an extracted document."""
    pages = []
    page_number = None
    lines: list[str] = []

    for raw_line in SAMPLE_CONTRACT.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[p") and line.endswith("]"):
            if page_number is not None:
                pages.append(Page(page_number=page_number, raw_text="\n".join(lines)))
            page_number = int(line.strip("[]p"))
            lines = []
        elif page_number is not None:
            lines.append(raw_line)

    if page_number is not None:
        pages.append(Page(page_number=page_number, raw_text="\n".join(lines)))

    return ExtractedDocument(
        document_id="sample-contract",
        page_count=len(pages),
        mode="demo",
        pages=pages,
    )


def full_text_chunks(document: ExtractedDocument) -> list[Chunk]:
    """Build one full-text retrieval chunk per source page."""
    return [
        Chunk(
            document_id=document.document_id,
            page=page.page_number,
            text=f"[p{page.page_number}] {page.raw_text}",
        )
        for page in document.pages
    ]


def compressed_rag_chunks(document_id: str) -> list[Chunk]:
    """Build compact DeepCompress-style RAG chunks for the fixture."""
    return [
        Chunk(
            document_id=document_id,
            page=1,
            text="[p1] Applicant: Ada Lovelace; application APP-2026-001; date 2026-03-15.",
        ),
        Chunk(
            document_id=document_id,
            page=2,
            text="[p2] Income: payroll $17,000/month; freelance $3,200/month; total monthly income $20,200.",
        ),
        Chunk(
            document_id=document_id,
            page=3,
            text="[p3] Debt: car loan $420/month; credit card $180/month; total monthly debt $600/month.",
        ),
        Chunk(
            document_id=document_id,
            page=4,
            text="[p4] Terms: payment net 30; confidentiality required; annual renewal unless notice is given.",
        ),
    ]


def build_local_index(chunks: list[Chunk]) -> list[Chunk]:
    """Represent a vector index with deterministic local chunks."""
    return chunks


def main() -> int:
    """Run the enterprise RAG demo with the sample contract fixture."""
    extracted = load_sample_contract()
    question = QUESTION

    full_context = "\n".join(chunk.text for chunk in full_text_chunks(extracted))
    protected_facts_with_pages = extract_protected_facts_with_pages(extracted)
    protected_facts = extract_protected_facts(extracted)
    scalar_facts = {
        category: values
        for category, values in protected_facts_with_pages.items()
        if category
        in {
            "invoice_numbers",
            "ids",
            "emails",
            "phones",
            "dates",
            "amounts",
            "percentages",
            "account_numbers",
            "names",
        }
    }
    protected_context = append_missing_protected_facts(
        "\n".join(chunk.text for chunk in compressed_rag_chunks(extracted.document_id)),
        scalar_facts,
    )
    compressed_chunks = [
        Chunk(document_id=extracted.document_id, page=index + 1, text=line)
        for index, line in enumerate(protected_context.splitlines())
        if line.strip()
    ]
    local_index = build_local_index(compressed_chunks)
    compressed_context = "\n".join(chunk.text for chunk in compressed_chunks)

    retrieved = retrieve(question, local_index)
    answer = answer_from_retrieval(question, retrieved)
    evidence_values = [
        value.lower()
        for value in question["expected_values"] + question["expected_answer_terms"]
    ]
    evidence = [
        chunk
        for chunk in retrieved
        if any(value in chunk.text.lower() for value in evidence_values)
    ]

    model = "gpt-4o"
    original_tokens = count_tokens(full_context, provider="openai", model=model).count
    compressed_tokens = count_tokens(compressed_context, provider="openai", model=model).count
    reduction = 1.0 - (compressed_tokens / original_tokens) if original_tokens else 0.0

    pricing = _get_llm_pricing(model)
    original_cost = _calculate_llm_cost(original_tokens, pricing)
    compressed_cost = _calculate_llm_cost(compressed_tokens, pricing)
    cost_saved = original_cost - compressed_cost

    print("Question:")
    print(question["question"])
    print()
    print("Answer:")
    print(answer)
    print()
    print("Evidence:")
    for chunk in evidence:
        print(chunk.text)
    print()
    print("Protected facts:")
    for category, values in protected_facts.items():
        print(f"{category}: {', '.join(values)}")
    print()
    print("Tokens:")
    print(f"Original: {original_tokens:,}")
    print(f"Compressed: {compressed_tokens:,}")
    print(f"Reduction: {reduction:.1%}")
    print(f"Cost saved: ${cost_saved:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
