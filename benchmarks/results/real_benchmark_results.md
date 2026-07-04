# Real Data Benchmark

- Library source: `local_repo`
- D-TOON mode: `rag`
- Token counter model: `gpt-4o`

## Compression per document

| Document | Original tokens | Compressed tokens | Ratio | Chunks | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| loan_application | 413 | 399 | 1.03x | 8 | 6945.09 ms |
| services_agreement | 256 | 404 | 0.63x | 6 | 4543.55 ms |

## RAG comparison

| Test | Full Text RAG | DeepCompress RAG |
| --- | ---: | ---: |
| Tokens | 678 | 803 |
| Cost | $0.007797 | $0.009235 |
| Answer accuracy | 100% | 100% |
| Retrieval hit rate | 100% | 100% |
| Citation correctness | 100% | 100% |
| Exact values preserved | 100% | 100% |
| Latency | 8.06 ms | 6.96 ms |
| Cost reduction | baseline | -18.4% |
