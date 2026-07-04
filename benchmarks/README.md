# DeepCompress RAG Benchmark

This benchmark compares full-text RAG against DeepCompress-style compressed RAG
chunks using deterministic synthetic fixtures.

```bash
python benchmarks/run_benchmark.py
```

Outputs:

- `benchmarks/results/benchmark_results.json`
- `benchmarks/results/benchmark_results.md`

The default benchmark is offline and CI-safe. It does not require API keys,
Redis, Pinecone, Weaviate, CUDA, OCR model downloads, or network access.

Live provider flags are reserved for future work and intentionally fail until
that mode has explicit implementation and tests.

