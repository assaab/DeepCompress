"""Benchmark DeepCompress on realistic local document fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.rag_benchmark import (  # noqa: E402
    Chunk,
    answer_from_retrieval,
    average,
    load_json,
    markdown_table,
    measure_question,
    retrieve,
)
from deepcompress.core.config import DeepCompressConfig  # noqa: E402
from deepcompress.core.optimizer import DTOONOptimizer  # noqa: E402
from deepcompress.models.document import ExtractedDocument, Page  # noqa: E402
from deepcompress.processing.protected_facts import (  # noqa: E402
    append_missing_protected_facts,
    extract_protected_facts,
    extract_protected_facts_with_pages,
)
from deepcompress.utils.cost import _calculate_llm_cost, _get_llm_pricing  # noqa: E402
from deepcompress.utils.token_counter import count_tokens  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURES_DIR = ROOT / "fixtures" / "real"
DEFAULT_QUESTIONS = DEFAULT_FIXTURES_DIR / "questions.json"
DEFAULT_RESULTS_DIR = ROOT / "results"
PAGE_MARKER_RE = re.compile(r"^\[p(\d+)\]\s*", re.IGNORECASE)
SECTION_PAGE_RE = re.compile(r"^=== Page (\d+) ===$")


@dataclass(frozen=True)
class DocumentFixture:
    """A local page-marked document used for real-data benchmarking."""

    document_id: str
    path: Path
    extracted: ExtractedDocument


@dataclass(frozen=True)
class CompressionRun:
    """Compression output from the local DeepCompress pipeline."""

    document_id: str
    optimized_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    protected_facts: dict[str, list[str]]
    chunks: list[Chunk]


def load_page_marked_document(path: Path, document_id: str | None = None) -> ExtractedDocument:
    """Load a page-marked text fixture as an ExtractedDocument."""
    resolved_id = document_id or path.stem
    pages: list[Page] = []
    page_number: int | None = None
    lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        marker = raw_line.strip()
        if marker.startswith("[p") and marker.endswith("]"):
            if page_number is not None:
                pages.append(Page(page_number=page_number, raw_text="\n".join(lines)))
            page_number = int(marker.strip("[]p"))
            lines = []
        elif page_number is not None:
            lines.append(raw_line)

    if page_number is not None:
        pages.append(Page(page_number=page_number, raw_text="\n".join(lines)))

    return ExtractedDocument(
        document_id=resolved_id,
        page_count=len(pages),
        mode="fixture",
        pages=pages,
    )


def discover_fixtures(fixtures_dir: Path) -> list[DocumentFixture]:
    """Discover page-marked text fixtures in a directory."""
    fixtures = []
    for path in sorted(fixtures_dir.glob("*.txt")):
        extracted = load_page_marked_document(path)
        fixtures.append(
            DocumentFixture(
                document_id=extracted.document_id,
                path=path,
                extracted=extracted,
            )
        )
    return fixtures


def full_text_chunks(document: ExtractedDocument) -> list[Chunk]:
    """Build one retrieval chunk per source page."""
    return [
        Chunk(
            document_id=document.document_id,
            page=page.page_number,
            text=f"[p{page.page_number}] {page.raw_text.strip()}",
        )
        for page in document.pages
        if page.raw_text and page.raw_text.strip()
    ]


def chunks_from_optimized_text(document_id: str, optimized_text: str) -> list[Chunk]:
    """Convert DeepCompress output into page-cited retrieval chunks."""
    chunks: list[Chunk] = []
    current_page: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_page
        if current_page is None or not current_lines:
            current_lines = []
            return
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    page=current_page,
                    text=text if text.lower().startswith("[p") else f"[p{current_page}] {text}",
                )
            )
        current_lines = []

    for raw_line in optimized_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        section_match = SECTION_PAGE_RE.match(stripped)
        if section_match:
            flush()
            current_page = int(section_match.group(1))
            continue

        page_match = PAGE_MARKER_RE.match(stripped)
        if page_match:
            flush()
            current_page = int(page_match.group(1))
            current_lines.append(stripped)
            continue

        if stripped.startswith("Document ID:") or stripped.startswith("Total Pages:"):
            continue

        if current_page is None:
            current_page = 1
        current_lines.append(stripped)

    flush()

    if not chunks and optimized_text.strip():
        chunks.append(
            Chunk(
                document_id=document_id,
                page=1,
                text=optimized_text.strip(),
            )
        )
    return chunks


async def _get_dtoon_llm_client(config: DeepCompressConfig) -> Any:
    """Create an LLM client for D-TOON structured/rag modes."""
    if config.dtoon_mode == "raw":
        return None

    if not config.llm_api_key:
        raise ValueError(
            f"D-TOON {config.dtoon_mode} mode requires llm_api_key "
            "(set OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment)."
        )

    from deepcompress.integrations.llm import LLMClient

    return LLMClient(provider=config.llm_provider, config=config)


async def compress_fixture(
    fixture: DocumentFixture,
    config: DeepCompressConfig,
) -> CompressionRun:
    """Run the local DeepCompress compression pipeline on one fixture."""
    start = time.perf_counter()
    extracted = fixture.extracted
    optimizer = DTOONOptimizer(include_bbox=False, include_confidence=False)

    protected_facts_with_pages = (
        extract_protected_facts_with_pages(extracted)
        if config.protect_facts
        else {}
    )
    protected_facts = extract_protected_facts(extracted) if config.protect_facts else {}

    optimized_text = await optimizer.optimize_async(
        extracted,
        mode=config.dtoon_mode,
        llm_client=await _get_dtoon_llm_client(config),
    )
    if config.protect_facts:
        optimized_text = append_missing_protected_facts(
            optimized_text,
            protected_facts_with_pages,
        )

    original_text = "\n\n".join(
        page.raw_text.strip()
        for page in extracted.pages
        if page.raw_text and page.raw_text.strip()
    )
    original_count = count_tokens(
        original_text,
        provider=config.token_counter_provider,
        model=config.token_counter_model or config.llm_model,
        api_key=config.llm_api_key or None,
    )
    compressed_count = count_tokens(
        optimized_text,
        provider=config.token_counter_provider,
        model=config.token_counter_model or config.llm_model,
        api_key=config.llm_api_key or None,
    )
    compression_ratio = (
        original_count.count / compressed_count.count
        if compressed_count.count > 0
        else 1.0
    )

    return CompressionRun(
        document_id=extracted.document_id,
        optimized_text=optimized_text,
        original_tokens=original_count.count,
        compressed_tokens=compressed_count.count,
        compression_ratio=compression_ratio,
        processing_time_ms=(time.perf_counter() - start) * 1000,
        protected_facts=protected_facts,
        chunks=chunks_from_optimized_text(extracted.document_id, optimized_text),
    )


def evaluate_mode(
    mode: str,
    fixtures: list[DocumentFixture],
    compression_by_id: dict[str, CompressionRun],
    questions: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Evaluate one RAG mode across the real fixture corpus."""
    fixture_by_id = {fixture.document_id: fixture for fixture in fixtures}
    pricing = _get_llm_pricing(model)

    question_results = []
    context_tokens = 0
    protected_values_total = 0
    protected_values_present = 0
    start = time.perf_counter()

    for fixture in fixtures:
        if mode == "full_text":
            chunks = full_text_chunks(fixture.extracted)
        elif mode == "deepcompress_rag":
            chunks = compression_by_id[fixture.document_id].chunks
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")

        context_text = "\n".join(chunk.text for chunk in chunks)
        context_tokens += count_tokens(context_text, provider="openai", model=model).count

        protected_values = {
            value
            for values in compression_by_id[fixture.document_id].protected_facts.values()
            for value in values
        }
        for value in protected_values:
            protected_values_total += 1
            if value.lower() in context_text.lower():
                protected_values_present += 1

    for question in questions:
        fixture = fixture_by_id[question["document_id"]]
        chunks = (
            full_text_chunks(fixture.extracted)
            if mode == "full_text"
            else compression_by_id[fixture.document_id].chunks
        )
        context_text = "\n".join(chunk.text for chunk in chunks)
        question_results.append(measure_question(question, chunks, context_text, model))

    elapsed_ms = (time.perf_counter() - start) * 1000
    cost = _calculate_llm_cost(context_tokens, pricing)

    return {
        "mode": mode,
        "tokens": context_tokens,
        "cost_usd": cost,
        "answer_accuracy": average([result["answer_accuracy"] for result in question_results]),
        "retrieval_hit_rate": average([result["retrieval_hit"] for result in question_results]),
        "citation_correctness": average(
            [result["citation_correct"] for result in question_results]
        ),
        "exact_values_preserved": (
            protected_values_present / protected_values_total
            if protected_values_total
            else 1.0
        ),
        "latency_ms": elapsed_ms,
        "questions": question_results,
    }


def compare_modes(
    fixtures: list[DocumentFixture],
    compression_by_id: dict[str, CompressionRun],
    questions: list[dict[str, Any]],
    model: str = "gpt-4o",
    dtoon_mode: str = "raw",
) -> dict[str, Any]:
    """Compare full-text RAG against locally compressed DeepCompress output."""
    full_text = evaluate_mode("full_text", fixtures, compression_by_id, questions, model=model)
    deepcompress_rag = evaluate_mode(
        "deepcompress_rag",
        fixtures,
        compression_by_id,
        questions,
        model=model,
    )

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

    compression_summary = []
    for fixture in fixtures:
        run = compression_by_id[fixture.document_id]
        compression_summary.append(
            {
                "document_id": run.document_id,
                "source_file": str(fixture.path),
                "original_tokens": run.original_tokens,
                "compressed_tokens": run.compressed_tokens,
                "compression_ratio": round(run.compression_ratio, 3),
                "processing_time_ms": round(run.processing_time_ms, 2),
                "chunk_count": len(run.chunks),
            }
        )

    return {
        "benchmark": "real_data",
        "library_source": "local_repo",
        "dtoon_mode": dtoon_mode,
        "model": model,
        "documents": compression_summary,
        "modes": [full_text, deepcompress_rag],
        "summary": {
            "token_reduction": token_reduction,
            "cost_reduction": cost_reduction,
        },
    }


async def run_real_benchmark_async(
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    questions_path: Path = DEFAULT_QUESTIONS,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    model: str = "gpt-4o",
    dtoon_mode: str = "raw",
    protect_facts: bool = True,
    llm_provider: str = "openai",
    llm_api_key: str = "",
) -> dict[str, Any]:
    """Run the real-data benchmark and write JSON/Markdown results."""
    fixtures = discover_fixtures(fixtures_dir)
    if not fixtures:
        raise FileNotFoundError(f"No .txt fixtures found in {fixtures_dir}")

    questions = load_json(questions_path)
    config = DeepCompressConfig(
        dtoon_mode=dtoon_mode,  # type: ignore[arg-type]
        protect_facts=protect_facts,
        ocr_device="cpu",
        vector_db_provider="none",
        cache_enabled=False,
        llm_provider=llm_provider,  # type: ignore[arg-type]
        llm_api_key=llm_api_key,
        token_counter_provider="openai",
        token_counter_model=model,
    )
    config.validate_config()

    compression_by_id = {}
    for fixture in fixtures:
        compression_by_id[fixture.document_id] = await compress_fixture(fixture, config)

    results = compare_modes(
        fixtures,
        compression_by_id,
        questions,
        model=model,
        dtoon_mode=dtoon_mode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_benchmark_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_dir / "real_benchmark_results.md").write_text(
        render_markdown_report(results),
        encoding="utf-8",
    )
    return results


def render_markdown_report(results: dict[str, Any]) -> str:
    """Render a Markdown report for the real-data benchmark."""
    lines = [
        "# Real Data Benchmark",
        "",
        f"- Library source: `{results['library_source']}`",
        f"- D-TOON mode: `{results['dtoon_mode']}`",
        f"- Token counter model: `{results['model']}`",
        "",
        "## Compression per document",
        "",
        "| Document | Original tokens | Compressed tokens | Ratio | Chunks | Time |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for document in results["documents"]:
        lines.append(
            "| {document_id} | {original_tokens:,} | {compressed_tokens:,} | "
            "{compression_ratio:.2f}x | {chunk_count} | {processing_time_ms:.2f} ms |".format(
                **document
            )
        )

    lines.extend(
        [
            "",
            "## RAG comparison",
            "",
            markdown_table(results).rstrip(),
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the real-data benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the local DeepCompress benchmark on realistic document fixtures.",
    )
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--dtoon-mode",
        choices=["raw", "structured", "rag"],
        default="raw",
        help="raw works offline; structured/rag require an LLM API key.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic"],
        default="openai",
    )
    parser.add_argument(
        "--llm-api-key",
        default="",
        help="Optional override; otherwise read from OPENAI_API_KEY or ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--no-protect-facts",
        action="store_true",
        help="Disable protected exact-value preservation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    import os

    llm_api_key = args.llm_api_key
    if not llm_api_key and args.dtoon_mode != "raw":
        env_key = "OPENAI_API_KEY" if args.llm_provider == "openai" else "ANTHROPIC_API_KEY"
        llm_api_key = os.getenv(env_key, "")

    results = asyncio.run(
        run_real_benchmark_async(
            fixtures_dir=args.fixtures_dir,
            questions_path=args.questions,
            output_dir=args.output_dir,
            model=args.model,
            dtoon_mode=args.dtoon_mode,
            protect_facts=not args.no_protect_facts,
            llm_provider=args.llm_provider,
            llm_api_key=llm_api_key,
        )
    )

    print(render_markdown_report(results))
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
