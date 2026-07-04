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
