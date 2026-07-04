# Real Data Benchmark

This benchmark runs the **local checkout** of DeepCompress against realistic
page-marked enterprise document fixtures. It does not use the published PyPI
package and it does not rely on pre-baked compressed chunks in JSON.

```bash
python benchmarks/run_real_benchmark.py
```

Outputs:

- `benchmarks/results/real_benchmark_results.json`
- `benchmarks/results/real_benchmark_results.md`

Fixtures live in `benchmarks/fixtures/real/`:

- `loan_application.txt`
- `services_agreement.txt`
- `questions.json`

## What this measures

1. **Compression** through the local pipeline:
   - page-marked text fixture -> `ExtractedDocument`
   - `DTOONOptimizer`
   - protected exact-value preservation
   - measured token counts
2. **RAG quality** against the same fixture questions:
   - full-text page chunks vs locally compressed chunks
   - answer accuracy, retrieval hit rate, citation correctness, exact-value preservation

## API and hardware requirements

| Mode | OCR / GPU | LLM API | Vector DB | Notes |
| --- | --- | --- | --- | --- |
| Default `--dtoon-mode raw` | No | No | No | Best offline baseline. Uses local fixtures instead of PDF OCR. |
| `--dtoon-mode rag` | No | Yes | No | Needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Recommended for realistic compression. |
| `--dtoon-mode structured` | No | Yes | No | Same API requirement as `rag`. |
| Full PDF pipeline via `DocumentCompressor` | Yes | Optional | Optional | Requires CUDA GPU, `pip install -e ".[gpu]"`, and DeepSeek-OCR model download. LLM only needed for `structured` / `rag`. |

### LLM APIs

For `--dtoon-mode rag` or `structured`:

- **OpenAI**: set `OPENAI_API_KEY`
- **Anthropic**: set `ANTHROPIC_API_KEY` and pass `--llm-provider anthropic`

Example:

```bash
set OPENAI_API_KEY=sk-...
python benchmarks/run_real_benchmark.py --dtoon-mode rag --llm-provider openai
```

### OCR / GPU

The real-data benchmark intentionally skips OCR so you can measure compression
and RAG behavior on realistic text without downloading models. To benchmark PDF
ingestion itself, use `DocumentCompressor` with:

- NVIDIA GPU with CUDA
- `pip install -e ".[gpu]"`
- Hugging Face access to `deepseek-ai/DeepSeek-OCR`

No LLM is required for OCR-only extraction with `dtoon_mode="raw"`.

### Vector DB

Not required for this benchmark. Retrieval uses the same deterministic lexical
scorer as `benchmarks/rag_benchmark.py`.

## Compare with the synthetic benchmark

```bash
python benchmarks/run_benchmark.py
```

That older benchmark compares hand-authored compressed chunks in
`benchmarks/datasets/synthetic_documents.json`. Use the real-data benchmark when
you want numbers from the actual local DeepCompress code path.

## Colab OCR Benchmark

Use this when you want to test the main DeepSeek-OCR path instead of the
pre-extracted text fixtures:

```bash
python benchmarks/run_ocr_colab_benchmark.py --regenerate-dataset --input-kind pdf --ocr-device cuda:0 --ocr-mode small --dtoon-mode raw
```

Outputs:

- `benchmarks/results/ocr_colab/ocr_colab_results.json`
- `benchmarks/results/ocr_colab/ocr_colab_results.md`
- `benchmarks/results/ocr_colab/ocr_tokens.png`
- `benchmarks/results/ocr_colab/ocr_compression_ratio.png`
- `benchmarks/results/ocr_colab/ocr_fact_recall.png`

Recommended Colab setup:

```bash
apt-get update && apt-get install -y poppler-utils
pip install -e ".[gpu,llm]"
```

The benchmark generates synthetic multi-page PDFs and PNG page images under
`benchmarks/fixtures/ocr_colab/`, runs `DocumentCompressor`, and compares:

- full OCR text tokens vs compressed D-TOON tokens
- compression ratio and tokens saved
- exact-fact recall in OCR output
- exact-fact recall after compression
- per-document runtime

Use `--input-kind images` if PDF conversion is unavailable. Use `--dtoon-mode
rag` or `structured` to test LLM-assisted compression with your `.env` provider
settings.
