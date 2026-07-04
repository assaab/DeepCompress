# DeepCompress

📄 Compress OCR-heavy documents into smaller, page-cited context for LLM and RAG workflows.

DeepCompress helps you reduce token usage while keeping important document facts traceable. It is useful for contracts, invoices, financial packets, claims, and other long documents where exact values matter.

## Why Use It?

- ⚡ **Reduce context size** before sending document text to an LLM.
- 🔎 **Keep page-cited output** for retrieval and answer evidence.
- 🧾 **Protect exact values** like IDs, dates, amounts, emails, phone numbers, clauses, and totals.
- 📊 **Measure token savings** with provider-aware token counters.
- 🔐 **Scrub common PII patterns** before LLM analysis when needed.
- 🧪 **Run offline demos and benchmarks** without API keys or external services.

## Installation

```bash
pip install deepcompress
```

Optional extras:

```bash
pip install "deepcompress[gpu]"       # OCR model support
pip install "deepcompress[llm]"       # OpenAI / Anthropic integrations
pip install "deepcompress[vector-db]" # Pinecone / Weaviate integrations
pip install "deepcompress[all]"       # Everything for local development
```

## Quick Start

Compress a document and inspect the measured savings:

```python
import asyncio

from deepcompress import DeepCompressConfig, DocumentCompressor


async def main():
    config = DeepCompressConfig(
        dtoon_mode="raw",
        protect_facts=True,
        vector_db_provider="none",
        cache_enabled=False,
    )

    compressor = DocumentCompressor(config)
    result = await compressor.compress("contract.pdf")

    print(result.optimized_text)
    print(f"Original tokens: {result.original_tokens_measured:,}")
    print(f"Compressed tokens: {result.compressed_tokens_measured:,}")
    print(f"Compression ratio: {result.compression_ratio_measured:.2f}x")
    print(f"Protected facts: {result.protected_facts}")


asyncio.run(main())
```

## Ask A Question

Use the high-level API when you want compression plus an LLM answer:

```python
import asyncio

from deepcompress import compress_and_analyze


async def main():
    result = await compress_and_analyze(
        file="loan_application.pdf",
        query="What is the applicant's total monthly income?",
        llm="openai",
        scrub_pii=True,
    )

    print(result.answer)
    print(f"Tokens saved: {result.tokens_saved:,}")


asyncio.run(main())
```

For LLM-backed usage, set your provider key in the environment:

```bash
# macOS / Linux
export LLM_API_KEY=your-api-key

# Windows PowerShell
$env:LLM_API_KEY="your-api-key"
```

## Common Configuration

```python
from deepcompress import DeepCompressConfig

config = DeepCompressConfig(
    dtoon_mode="raw",              # raw, structured, or rag
    ocr_device="cpu",              # cpu, cuda:0, cuda:1
    token_counter_provider="openai",
    token_counter_model="gpt-4o",
    protect_facts=True,
    cache_enabled=False,
    vector_db_provider="none",
)
```

Useful modes:

| Mode | Best For |
| --- | --- |
| `raw` | Fast compression without an LLM provider |
| `structured` | Cleaner structured summaries using an API-backed LLM |
| `rag` | Retrieval-friendly chunks for RAG pipelines |

## Offline Demo

Run the enterprise RAG demo without API keys, Redis, Pinecone, CUDA, or network access:

```bash
python examples/enterprise_rag_demo/demo.py
```

The demo compresses a synthetic loan document, builds a local in-memory index, answers a question with evidence, and prints token and cost savings.

## Benchmarks

Run the deterministic benchmark:

```bash
python benchmarks/run_benchmark.py
```

The benchmark compares full-text RAG with DeepCompress RAG and reports token usage, cost, answer accuracy, retrieval hit rate, citation correctness, exact-value preservation, and latency.

## What You Get Back

`DocumentCompressor.compress()` returns a compressed document result with:

- `optimized_text` - the compressed D-TOON document text
- `original_tokens_measured` - measured input token count
- `compressed_tokens_measured` - measured compressed token count
- `compression_ratio_measured` - measured compression ratio
- `tokens_saved` - token reduction
- `cost_saved_usd` - estimated GPT-4o-style input savings
- `protected_facts` - exact values preserved during compression
- `processing_time_ms` - total processing time

## Notes And Limits

- Compression quality depends on document type, OCR quality, and selected mode.
- `structured` and `rag` modes require an API-backed LLM provider.
- PII scrubbing is pattern-based and should be reviewed for your data domain.
- Built-in benchmarks use synthetic fixtures; validate results on your own documents before publishing savings claims.

## License

MIT
