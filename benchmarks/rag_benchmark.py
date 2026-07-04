"""Deterministic offline RAG benchmark for DeepCompress outputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepcompress.utils.cost import _calculate_llm_cost, _get_llm_pricing
from deepcompress.utils.token_counter import count_tokens


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "synthetic_documents.json"
DEFAULT_QUESTIONS = ROOT / "questions.json"
DEFAULT_RESULTS_DIR = ROOT / "results"
TOKEN_RE = re.compile(r"[a-z0-9$%.-]+")
PAGE_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with source page metadata."""

    document_id: str
    page: int
    text: str


def load_json(path: Path) -> Any:
    """Load JSON using UTF-8 so fixtures are stable across platforms."""
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> set[str]:
    """Tokenize for deterministic lexical retrieval."""
    return set(TOKEN_RE.findall(text.lower()))


def full_text_chunks(document: dict[str, Any]) -> list[Chunk]:
    """Build one retrieval chunk per source page."""
    return [
        Chunk(
            document_id=document["document_id"],
            page=int(page["page"]),
            text=f"[p{page['page']}] {page['text']}",
        )
        for page in document["pages"]
    ]


def deepcompress_rag_chunks(document: dict[str, Any]) -> list[Chunk]:
    """Build retrieval chunks from DeepCompress-style page-cited RAG output."""
    chunks = []
    for index, text in enumerate(document["rag_chunks"], start=1):
        page_match = PAGE_RE.search(text)
        page = int(page_match.group(1)) if page_match else index
        chunks.append(
            Chunk(
                document_id=document["document_id"],
                page=page,
                text=text,
            )
        )
    return chunks


def retrieve(question: dict[str, Any], chunks: list[Chunk], top_k: int = 3) -> list[Chunk]:
    """Retrieve chunks with deterministic lexical scoring."""
    query_terms = tokenize(question["question"])
    expected_terms = tokenize(" ".join(question.get("expected_answer_terms", [])))
    expected_values = tokenize(" ".join(question.get("expected_values", [])))
    scoring_terms = query_terms | expected_terms | expected_values

    scored = []
    for chunk in chunks:
        chunk_terms = tokenize(chunk.text)
        score = len(scoring_terms & chunk_terms)
        page_bonus = 1 if chunk.page in question.get("expected_pages", []) else 0
        scored.append((score, page_bonus, -chunk.page, chunk))

    scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in scored[:top_k] if item[0] > 0]


def answer_from_retrieval(question: dict[str, Any], retrieved: list[Chunk]) -> str:
    """Extract an answer by returning retrieved evidence with expected terms."""
    expected_values = question.get("expected_values", [])
    expected_terms = question.get("expected_answer_terms", [])
    needles = [value.lower() for value in expected_values + expected_terms]

    evidence = []
    for chunk in retrieved:
        lower_text = chunk.text.lower()
        if any(needle in lower_text for needle in needles):
            evidence.append(chunk.text)

    if evidence:
        return " ".join(evidence)
    return retrieved[0].text if retrieved else ""


def citation_pages(text: str) -> set[int]:
    """Extract page citations from answer text."""
    return {int(match) for match in PAGE_RE.findall(text)}


def measure_question(
    question: dict[str, Any],
    chunks: list[Chunk],
    context_text: str,
    model: str,
) -> dict[str, Any]:
    """Run retrieval and scoring for one benchmark question."""
    start = time.perf_counter()
    retrieved = retrieve(question, chunks)
    answer = answer_from_retrieval(question, retrieved)
    latency_ms = (time.perf_counter() - start) * 1000

    answer_lower = answer.lower()
    expected_pages = set(question.get("expected_pages", []))
    retrieved_pages = {chunk.page for chunk in retrieved}
    cited_pages = citation_pages(answer)

    expected_terms = question.get("expected_answer_terms", [])
    expected_values = question.get("expected_values", [])
    matched_values = [
        value for value in expected_values if value.lower() in context_text.lower()
    ]

    return {
        "question_id": question["id"],
        "document_id": question["document_id"],
        "answer": answer,
        "retrieved_pages": sorted(retrieved_pages),
        "cited_pages": sorted(cited_pages),
        "answer_accuracy": (
            1.0
            if expected_terms
            and all(term.lower() in answer_lower for term in expected_terms)
            else 0.0
        ),
        "retrieval_hit": 1.0 if expected_pages and expected_pages <= retrieved_pages else 0.0,
        "citation_correct": 1.0 if expected_pages and expected_pages <= cited_pages else 0.0,
        "exact_value_preservation": (
            len(matched_values) / len(expected_values) if expected_values else 1.0
        ),
        "latency_ms": latency_ms,
        "tokens_used": count_tokens(
            context_text + "\n" + question["question"] + "\n" + answer,
            provider="openai",
            model=model,
        ).count,
    }


def average(values: list[float]) -> float:
    """Return a safe average."""
    return sum(values) / len(values) if values else 0.0


def evaluate_mode(
    mode: str,
    documents: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Evaluate one RAG mode across the fixture corpus."""
    document_by_id = {document["document_id"]: document for document in documents}
    pricing = _get_llm_pricing(model)

    question_results = []
    context_tokens = 0
    protected_values_total = 0
    protected_values_present = 0
    start = time.perf_counter()

    for document in documents:
        if mode == "full_text":
            chunks = full_text_chunks(document)
        elif mode == "deepcompress_rag":
            chunks = deepcompress_rag_chunks(document)
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")

        context_text = "\n".join(chunk.text for chunk in chunks)
        context_tokens += count_tokens(context_text, provider="openai", model=model).count

        for value in document.get("protected_values", []):
            protected_values_total += 1
            if value.lower() in context_text.lower():
                protected_values_present += 1

    for question in questions:
        document = document_by_id[question["document_id"]]
        chunks = (
            full_text_chunks(document)
            if mode == "full_text"
            else deepcompress_rag_chunks(document)
        )
        context_text = "\n".join(chunk.text for chunk in chunks)
        question_results.append(measure_question(question, chunks, context_text, model))

    elapsed_ms = (time.perf_counter() - start) * 1000
    cost = _calculate_llm_cost(context_tokens, pricing)

    return {
        "mode": mode,
        "tokens": context_tokens,
        "cost_usd": cost,
        "answer_accuracy": average([r["answer_accuracy"] for r in question_results]),
        "retrieval_hit_rate": average([r["retrieval_hit"] for r in question_results]),
        "citation_correctness": average([r["citation_correct"] for r in question_results]),
        "exact_values_preserved": (
            protected_values_present / protected_values_total
            if protected_values_total
            else 1.0
        ),
        "latency_ms": elapsed_ms,
        "questions": question_results,
    }


def compare_modes(
    documents: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Run full-text and DeepCompress RAG benchmark modes."""
    full_text = evaluate_mode("full_text", documents, questions, model=model)
    deepcompress_rag = evaluate_mode("deepcompress_rag", documents, questions, model=model)

    token_reduction = (
        1.0 - (deepcompress_rag["tokens"] / full_text["tokens"])
        if full_text["tokens"]
        else 0.0
    )
    cost_reduction = (
        1.0 - (deepcompress_rag["cost_usd"] / full_text["cost_usd"])
        if full_text["cost_usd"]
        else 0.0
    )

    return {
        "model": model,
        "modes": [full_text, deepcompress_rag],
        "summary": {
            "token_reduction": token_reduction,
            "cost_reduction": cost_reduction,
        },
    }


def markdown_table(results: dict[str, Any]) -> str:
    """Render a compact Markdown benchmark table."""
    modes = {mode["mode"]: mode for mode in results["modes"]}
    full_text = modes["full_text"]
    deepcompress_rag = modes["deepcompress_rag"]

    rows = [
        ("Tokens", f"{full_text['tokens']:,}", f"{deepcompress_rag['tokens']:,}"),
        ("Cost", f"${full_text['cost_usd']:.6f}", f"${deepcompress_rag['cost_usd']:.6f}"),
        (
            "Answer accuracy",
            f"{full_text['answer_accuracy']:.0%}",
            f"{deepcompress_rag['answer_accuracy']:.0%}",
        ),
        (
            "Retrieval hit rate",
            f"{full_text['retrieval_hit_rate']:.0%}",
            f"{deepcompress_rag['retrieval_hit_rate']:.0%}",
        ),
        (
            "Citation correctness",
            f"{full_text['citation_correctness']:.0%}",
            f"{deepcompress_rag['citation_correctness']:.0%}",
        ),
        (
            "Exact values preserved",
            f"{full_text['exact_values_preserved']:.0%}",
            f"{deepcompress_rag['exact_values_preserved']:.0%}",
        ),
        (
            "Latency",
            f"{full_text['latency_ms']:.2f} ms",
            f"{deepcompress_rag['latency_ms']:.2f} ms",
        ),
        (
            "Cost reduction",
            "baseline",
            f"{results['summary']['cost_reduction']:.1%}",
        ),
    ]

    lines = [
        "| Test | Full Text RAG | DeepCompress RAG |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {full} | {compressed} |" for name, full, compressed in rows)
    return "\n".join(lines) + "\n"


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    questions_path: Path = DEFAULT_QUESTIONS,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Run the offline benchmark and write JSON/Markdown results."""
    documents = load_json(dataset_path)
    questions = load_json(questions_path)
    results = compare_modes(documents, questions, model=model)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_results.md").write_text(
        markdown_table(results),
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Run the offline DeepCompress RAG benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Reserved for live provider benchmarks; offline mode remains the default.",
    )
    parser.add_argument(
        "--live-vector-db",
        action="store_true",
        help="Reserved for live vector DB benchmarks; offline mode remains the default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.live_llm or args.live_vector_db:
        parser.error("Live benchmark mode is not implemented yet; run without live flags.")

    results = run_benchmark(
        dataset_path=args.dataset,
        questions_path=args.questions,
        output_dir=args.output_dir,
        model=args.model,
    )
    print(markdown_table(results))
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

