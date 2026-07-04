"""Offline enterprise RAG demo using synthetic benchmark fixtures."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.rag_benchmark import (
    DEFAULT_DATASET,
    DEFAULT_QUESTIONS,
    answer_from_retrieval,
    deepcompress_rag_chunks,
    full_text_chunks,
    load_json,
    retrieve,
)
from deepcompress.utils.cost import _calculate_llm_cost, _get_llm_pricing
from deepcompress.utils.token_counter import count_tokens


def main() -> int:
    """Run the enterprise RAG demo with the loan application fixture."""
    documents = load_json(DEFAULT_DATASET)
    questions = load_json(DEFAULT_QUESTIONS)

    document = next(doc for doc in documents if doc["document_id"] == "loan-app-001")
    question = next(item for item in questions if item["id"] == "loan-income")

    full_context = "\n".join(chunk.text for chunk in full_text_chunks(document))
    compressed_chunks = deepcompress_rag_chunks(document)
    compressed_context = "\n".join(chunk.text for chunk in compressed_chunks)

    retrieved = retrieve(question, compressed_chunks)
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
    print("Tokens:")
    print(f"Original: {original_tokens:,}")
    print(f"Compressed: {compressed_tokens:,}")
    print(f"Reduction: {reduction:.1%}")
    print(f"Cost saved: ${cost_saved:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
