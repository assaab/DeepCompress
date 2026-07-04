"""Colab-ready OCR benchmark for the real DeepCompress pipeline.

This benchmark exercises the main library path:

    generated document image/PDF -> DeepSeek-OCR -> DTOONOptimizer -> token metrics

It intentionally avoids pre-extracted text fixtures. The generated dataset has
known exact facts so OCR recall and compressed-output fact preservation can be
measured without sending answers to another evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcompress import DeepCompressConfig, DocumentCompressor  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "fixtures" / "ocr_colab"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ocr_colab"


@dataclass
class SyntheticDocument:
    document_id: str
    title: str
    pages: list[str]
    expected_facts: list[str]


DATASET: list[SyntheticDocument] = [
    SyntheticDocument(
        document_id="loan_packet_2026",
        title="Residential Loan Review Packet",
        pages=[
            """
            RESIDENTIAL LOAN REVIEW PACKET
            Application ID: LN-2026-00419
            Applicant: Jordan Vale
            Co-applicant: Mira Vale
            Property address: 418 Cedar Spring Road, Austin, TX 78704
            Requested loan amount: $486,000
            Purchase price: $610,000
            Down payment: $124,000
            Application date: 2026-03-15

            The application packet includes borrower disclosures, income
            verification, asset statements, debt review, and an underwriting
            decision memo. This synthetic document is intentionally verbose so
            the OCR compression path has repeated context to remove.

            The borrower profile section repeats background narrative:
            Jordan Vale has been employed by Analytical Engines LLC since 2021.
            Mira Vale operates a consulting practice with recurring contracts.
            The subject property will be owner occupied after closing.
            """,
            """
            INCOME SUMMARY AND SUPPORTING DETAILS
            Employer: Analytical Engines LLC
            Payroll income: $17,000 per month
            Freelance consulting income: $3,200 per month
            Total verified monthly income: $20,200

            Income evidence table
            Source                         Monthly Amount      Evidence
            Payroll wages                  $17,000             Pay stubs and W-2
            Consulting contracts           $3,200              Invoices and deposits
            Total verified income          $20,200             Underwriter calculation

            Narrative notes:
            The payroll income was stable across the review period. Consulting
            invoices were averaged over the documented lookback period. The
            underwriter accepted both sources for qualifying income.
            """,
            """
            MONTHLY DEBT AND UNDERWRITING DECISION
            Automobile loan payment: $420 per month
            Credit card minimum payment: $180 per month
            Total monthly debt: $600
            Debt-to-income ratio: 2.97 percent
            Credit score: 740
            Underwriting action: APPROVE SUBJECT TO STANDARD CLOSING CONDITIONS

            Conditions:
            1. Re-verify employment within ten days of closing.
            2. Confirm source of down payment funds.
            3. Confirm hazard insurance binder before final approval.

            Decision memo:
            The file meets income, credit, and collateral requirements. The
            debt ratio is low relative to verified income. The decision remains
            subject to standard closing conditions and fraud checks.
            """,
        ],
        expected_facts=[
            "LN-2026-00419",
            "$486,000",
            "$610,000",
            "$17,000",
            "$3,200",
            "$20,200",
            "$600",
            "2.97 percent",
            "740",
            "APPROVE SUBJECT TO STANDARD CLOSING CONDITIONS",
        ],
    ),
    SyntheticDocument(
        document_id="services_agreement_2026",
        title="Enterprise Services Agreement",
        pages=[
            """
            ENTERPRISE SERVICES AGREEMENT
            Agreement ID: ESA-2026-7781
            Vendor: Northwind Analytics LLC
            Customer: Contoso Retail Group
            Effective date: 2026-03-15
            Initial term: 24 months
            Governing law: State of Washington

            Scope summary:
            The vendor will provide operational analytics dashboards, monthly
            business reviews, anomaly monitoring, and support for retail
            reporting workflows. The parties agree to keep confidential
            business information protected during and after the term.
            """,
            """
            PAYMENT TERMS AND BILLING
            Monthly recurring service fee: $8,500
            Implementation fee: $14,000
            Invoice terms: net 30 from invoice date
            Late payment charge: 1.5% per month
            Vendor billing contact: billing@northwind.example.com
            Customer accounts payable: ap@contoso.example.com

            Fee table
            Item                           Amount              Timing
            Monthly subscription           $8,500              Monthly
            Implementation services        $14,000             One time
            Late charge                     1.5%                Monthly on unpaid balance
            """,
            """
            TERMINATION AND REMEDIES
            Either party may terminate for material breach.
            Cure period: 30 days after written notice
            Confidentiality survival period: 12 months after termination
            Service credit cap: $25,000
            Liability cap: $102,000

            Remedy notes:
            Termination for material breach is permitted only if the breach
            remains uncured after notice. Confidential information must remain
            protected through termination and for the survival period.
            """,
        ],
        expected_facts=[
            "ESA-2026-7781",
            "Northwind Analytics LLC",
            "Contoso Retail Group",
            "2026-03-15",
            "24 months",
            "$8,500",
            "$14,000",
            "net 30",
            "1.5%",
            "30 days",
            "12 months",
            "$25,000",
            "$102,000",
        ],
    ),
]


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def fact_recall(text: str, expected_facts: list[str]) -> tuple[int, int, float, list[str]]:
    normalized = normalize_text(text)
    found = []
    missing = []
    for fact in expected_facts:
        if normalize_text(fact) in normalized:
            found.append(fact)
        else:
            missing.append(fact)

    total = len(expected_facts)
    score = len(found) / total if total else 1.0
    return len(found), total, score, missing


def find_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_page(text: str, title: str, page_number: int):
    from PIL import Image, ImageDraw

    width, height = 1700, 2200
    margin = 110
    line_spacing = 12
    font = find_font(34)
    title_font = find_font(44)
    footer_font = find_font(26)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = margin
    draw.text((margin, y), title, fill="black", font=title_font)
    y += 78
    draw.line((margin, y, width - margin, y), fill="black", width=3)
    y += 48

    for paragraph in textwrap.dedent(text).strip().splitlines():
        if not paragraph.strip():
            y += 30
            continue

        wrapped = textwrap.wrap(paragraph, width=78) or [paragraph]
        for line in wrapped:
            draw.text((margin, y), line, fill="black", font=font)
            y += 42 + line_spacing
            if y > height - 170:
                break
        if y > height - 170:
            break

    draw.line((margin, height - 125, width - margin, height - 125), fill="gray", width=2)
    draw.text(
        (margin, height - 95),
        f"Synthetic OCR benchmark page {page_number}",
        fill="gray",
        font=footer_font,
    )
    return image


def generate_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    """Generate synthetic PNG pages and multi-page PDFs for OCR benchmarking."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for doc in DATASET:
        doc_dir = dataset_dir / doc.document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        images = []
        image_paths = []
        for index, page_text in enumerate(doc.pages, start=1):
            image = render_page(page_text, doc.title, index)
            image_path = doc_dir / f"page_{index:02d}.png"
            image.save(image_path)
            images.append(image)
            image_paths.append(str(image_path))

        pdf_path = doc_dir / f"{doc.document_id}.pdf"
        first, rest = images[0], images[1:]
        first.save(pdf_path, "PDF", resolution=200.0, save_all=True, append_images=rest)

        manifest.append(
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "pdf_path": str(pdf_path),
                "image_paths": image_paths,
                "page_count": len(doc.pages),
                "expected_facts": doc.expected_facts,
            }
        )

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_or_generate_dataset(dataset_dir: Path, regenerate: bool) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    if regenerate or not manifest_path.exists():
        return generate_dataset(dataset_dir)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def document_text(result: Any) -> str:
    return "\n\n".join(
        page.raw_text.strip()
        for page in result.extracted.pages
        if page.raw_text and page.raw_text.strip()
    )


def compact_snippet(text: str, limit: int = 600) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[:limit] + "..."


async def compress_one(
    compressor: DocumentCompressor,
    file_path: str,
    document_id: str,
) -> Any:
    return await compressor.compress(file_path, document_id=document_id)


async def run_ocr_benchmark_async(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv_if_present(REPO_ROOT / ".env")

    manifest = load_or_generate_dataset(args.dataset_dir, regenerate=args.regenerate_dataset)
    if args.limit_docs:
        manifest = manifest[: args.limit_docs]

    llm_api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or ""
    )

    config = DeepCompressConfig(
        dtoon_mode=args.dtoon_mode,
        protect_facts=not args.no_protect_facts,
        ocr_device=args.ocr_device,
        ocr_mode=args.ocr_mode,
        ocr_max_new_tokens=args.ocr_max_new_tokens,
        ocr_batch_size=1,
        vector_db_provider="none",
        cache_enabled=False,
        llm_provider=args.llm_provider,
        llm_api_key=llm_api_key,
        llm_model=args.model,
        token_counter_provider="openai",
        token_counter_model=args.token_counter_model,
    )

    compressor = DocumentCompressor(config)
    documents = []
    started = time.perf_counter()

    for item in manifest:
        if args.input_kind == "pdf":
            targets = [(item["document_id"], item["pdf_path"], item["expected_facts"])]
        else:
            targets = [
                (
                    f"{item['document_id']}_page_{index}",
                    image_path,
                    item["expected_facts"],
                )
                for index, image_path in enumerate(item["image_paths"], start=1)
            ]

        for document_id, file_path, expected_facts in targets:
            result = await compress_one(compressor, file_path, document_id=document_id)
            ocr_text = document_text(result)
            ocr_found, ocr_total, ocr_recall, ocr_missing = fact_recall(
                ocr_text,
                expected_facts,
            )
            compressed_found, compressed_total, compressed_recall, compressed_missing = (
                fact_recall(result.optimized_text, expected_facts)
            )

            documents.append(
                {
                    "document_id": document_id,
                    "source_path": file_path,
                    "page_count": result.extracted.page_count,
                    "ocr_mode": args.ocr_mode,
                    "dtoon_mode": args.dtoon_mode,
                    "original_tokens": result.original_tokens_measured,
                    "compressed_tokens": result.compressed_tokens_measured,
                    "tokens_saved": result.tokens_saved,
                    "compression_ratio": result.compression_ratio_measured,
                    "processing_time_ms": result.processing_time_ms,
                    "ocr_fact_recall": ocr_recall,
                    "ocr_facts_found": ocr_found,
                    "ocr_facts_total": ocr_total,
                    "ocr_missing_facts": ocr_missing,
                    "compressed_fact_recall": compressed_recall,
                    "compressed_facts_found": compressed_found,
                    "compressed_facts_total": compressed_total,
                    "compressed_missing_facts": compressed_missing,
                    "ocr_text_preview": compact_snippet(ocr_text),
                    "optimized_text_preview": compact_snippet(result.optimized_text),
                }
            )

    elapsed_ms = (time.perf_counter() - started) * 1000
    totals = {
        "original_tokens": sum(doc["original_tokens"] for doc in documents),
        "compressed_tokens": sum(doc["compressed_tokens"] for doc in documents),
        "tokens_saved": sum(doc["tokens_saved"] for doc in documents),
        "processing_time_ms": elapsed_ms,
    }
    totals["compression_ratio"] = (
        totals["original_tokens"] / totals["compressed_tokens"]
        if totals["compressed_tokens"]
        else 1.0
    )
    totals["token_reduction"] = (
        1.0 - (totals["compressed_tokens"] / totals["original_tokens"])
        if totals["original_tokens"]
        else 0.0
    )
    totals["ocr_fact_recall"] = average(doc["ocr_fact_recall"] for doc in documents)
    totals["compressed_fact_recall"] = average(
        doc["compressed_fact_recall"] for doc in documents
    )

    return {
        "benchmark": "ocr_colab",
        "input_kind": args.input_kind,
        "dataset_dir": str(args.dataset_dir),
        "dtoon_mode": args.dtoon_mode,
        "ocr_mode": args.ocr_mode,
        "ocr_device": args.ocr_device,
        "token_counter_model": args.token_counter_model,
        "documents": documents,
        "summary": totals,
    }


def average(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def render_markdown_report(results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# OCR Colab Benchmark",
        "",
        f"- Input kind: `{results['input_kind']}`",
        f"- OCR mode: `{results['ocr_mode']}`",
        f"- OCR device: `{results['ocr_device']}`",
        f"- D-TOON mode: `{results['dtoon_mode']}`",
        f"- Token counter model: `{results['token_counter_model']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Original OCR text tokens | {summary['original_tokens']:,} |",
        f"| Compressed output tokens | {summary['compressed_tokens']:,} |",
        f"| Tokens saved | {summary['tokens_saved']:,} |",
        f"| Compression ratio | {summary['compression_ratio']:.2f}x |",
        f"| Token reduction | {summary['token_reduction']:.1%} |",
        f"| OCR exact-fact recall | {summary['ocr_fact_recall']:.0%} |",
        f"| Compressed exact-fact recall | {summary['compressed_fact_recall']:.0%} |",
        f"| End-to-end time | {summary['processing_time_ms'] / 1000:.2f}s |",
        "",
        "## Per Document",
        "",
        (
            "| Document | Pages | Original tokens | Compressed tokens | Ratio | "
            "OCR facts | Compressed facts | Time |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for doc in results["documents"]:
        lines.append(
            "| {document_id} | {page_count} | {original_tokens:,} | "
            "{compressed_tokens:,} | {compression_ratio:.2f}x | "
            "{ocr_fact_recall:.0%} | {compressed_fact_recall:.0%} | "
            "{time:.2f}s |".format(
                time=doc["processing_time_ms"] / 1000,
                **doc,
            )
        )

    lines.extend(["", "## Missing Facts", ""])
    for doc in results["documents"]:
        if not doc["ocr_missing_facts"] and not doc["compressed_missing_facts"]:
            continue
        lines.append(f"### {doc['document_id']}")
        if doc["ocr_missing_facts"]:
            lines.append(
                "- Missing from OCR text: " + ", ".join(doc["ocr_missing_facts"])
            )
        if doc["compressed_missing_facts"]:
            lines.append(
                "- Missing from compressed text: "
                + ", ".join(doc["compressed_missing_facts"])
            )
        lines.append("")

    lines.extend(
        [
            "## Output Previews",
            "",
            "These previews are truncated. Use the JSON output for full metric data.",
            "",
        ]
    )
    for doc in results["documents"]:
        lines.extend(
            [
                f"### {doc['document_id']}",
                "",
                "**OCR text preview**",
                "",
                doc["ocr_text_preview"],
                "",
                "**Optimized text preview**",
                "",
                doc["optimized_text_preview"],
                "",
            ]
        )

    return "\n".join(lines)


def write_charts(results: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    documents = results["documents"]
    labels = [doc["document_id"] for doc in documents]
    original = [doc["original_tokens"] for doc in documents]
    compressed = [doc["compressed_tokens"] for doc in documents]
    ratios = [doc["compression_ratio"] for doc in documents]
    ocr_recall = [doc["ocr_fact_recall"] * 100 for doc in documents]
    compressed_recall = [doc["compressed_fact_recall"] * 100 for doc in documents]

    chart_paths = []

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], original, width, label="Full OCR text")
    ax.bar([i + width / 2 for i in x], compressed, width, label="Compressed output")
    ax.set_title("OCR Text Tokens vs Compressed Tokens")
    ax.set_ylabel("Tokens")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ocr_tokens.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    chart_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, ratios, color="#2f6f73")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title("Compression Ratio by Document")
    ax.set_ylabel("Original tokens / compressed tokens")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output_dir / "ocr_compression_ratio.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    chart_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], ocr_recall, width, label="OCR text")
    ax.bar([i + width / 2 for i in x], compressed_recall, width, label="Compressed")
    ax.set_title("Exact Fact Recall")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ocr_fact_recall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    chart_paths.append(str(path))

    return chart_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real DeepSeek-OCR DeepCompress benchmark for Colab.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--input-kind", choices=["pdf", "images"], default="pdf")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--ocr-device", default="cuda:0")
    parser.add_argument("--ocr-mode", choices=["small", "base", "large"], default="small")
    parser.add_argument("--ocr-max-new-tokens", type=int, default=2048)
    parser.add_argument("--dtoon-mode", choices=["raw", "structured", "rag"], default="raw")
    parser.add_argument("--llm-provider", choices=["openai", "claude"], default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--token-counter-model", default="gpt-4o")
    parser.add_argument("--no-protect-facts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(run_ocr_benchmark_async(args))
    json_path = args.output_dir / "ocr_colab_results.json"
    md_path = args.output_dir / "ocr_colab_results.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = render_markdown_report(results)
    md_path.write_text(markdown, encoding="utf-8")
    chart_paths = write_charts(results, args.output_dir)

    print(markdown)
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")
    if chart_paths:
        print("Charts:")
        for chart_path in chart_paths:
            print(f"- {chart_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
