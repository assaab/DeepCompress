# Real Data Benchmark

- Library source: `local_repo`
- D-TOON mode: `raw`
- Token counter model: `gpt-4o`

## Compression per document

| Document | Original tokens | Compressed tokens | Ratio | Chunks | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| loan_application | 413 | 440 | 0.94x | 4 | 13.88 ms |
| services_agreement | 256 | 280 | 0.91x | 3 | 4.23 ms |

## RAG comparison

| Test | Full Text RAG | DeepCompress RAG |
| --- | ---: | ---: |
| Tokens | 678 | 678 |
| Cost | $0.007797 | $0.007797 |
| Answer accuracy | 100% | 100% |
| Retrieval hit rate | 100% | 100% |
| Citation correctness | 100% | 100% |
| Exact values preserved | 100% | 100% |
| Latency | 7.93 ms | 7.41 ms |
| Cost reduction | baseline | 0.0% |
