# Enterprise RAG Demo

This offline demo shows the publishable DeepCompress RAG workflow without API keys,
Redis, Pinecone, CUDA, or network access.

```bash
python examples/enterprise_rag_demo/demo.py
```

Expected output includes:

```text
Answer:
[p2] Income: payroll $17,000/month; freelance $3,200/month; total monthly income $20,200.

Evidence:
[p2] Income: payroll $17,000/month; freelance $3,200/month; total monthly income $20,200.

Tokens:
Original: ...
Compressed: ...
Reduction: ...
Cost saved: ...
```

The numbers are measured from the synthetic fixture text and the configured token
counter. Compression ratios depend on document type, OCR quality, target model,
and compression mode.

