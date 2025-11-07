# EDC Examples

This directory contains example scripts demonstrating various EDC features and use cases.

## Examples

### 1. Quickstart (`quickstart.py`)

Basic document compression and LLM analysis.

```bash
python examples/quickstart.py
```

**What it demonstrates:**
- One-liner API usage
- Basic compression workflow
- Cost savings calculation
- D-TOON output format

---

### 2. Pilot Batch Processing (`pilot.py`)

Process 1,000 loan applications to validate accuracy and cost savings.

```bash
python examples/pilot.py
```

**What it demonstrates:**
- Batch processing with `BatchProcessor`
- Progress tracking
- Cache utilization
- Performance metrics
- Validation checklist

**Prerequisites:**
- Sample documents in `data/pilot/loan-applications/`
- Redis running locally

---

### 3. Production Deployment (`production.py`)

Full production workflow with vector DB indexing.

```bash
python examples/production.py
```

**What it demonstrates:**
- Vector database integration (Pinecone)
- LLM analysis with structured queries
- Risk scoring
- Structured logging with trace IDs
- Error handling

**Prerequisites:**
- AWS S3 access configured
- Pinecone API key in `.env`
- OpenAI API key in `.env`

---

### 4. Advanced Usage (`advanced_usage.py`)

Advanced features and customization options.

```bash
python examples/advanced_usage.py
```

**What it demonstrates:**
- Custom compression settings
- Custom PII patterns
- Semantic search with vector DB
- Detailed cost analysis
- Cache management

---

## Setup

### 1. Install EDC

```bash
pip install -e ".[all]"
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:
- `LLM_API_KEY` - OpenAI API key
- `VECTOR_DB_API_KEY` - Pinecone API key (for production/advanced examples)
- `CACHE_URL` - Redis URL (default: `redis://localhost:6379`)

### 3. Start Redis (for caching)

```bash
docker run -d -p 6379:6379 redis:latest
```

### 4. Prepare Sample Data

For pilot batch processing, create sample documents:

```bash
mkdir -p data/pilot/loan-applications
# Copy your PDF documents here
```

---

## Example Output

### Quickstart Output

```
============================================================
EDC Quickstart Example
============================================================

Compressing and analyzing document...

============================================================
RESULTS
============================================================

Document ID: a3f2b1c4d5e6f7g8

Original tokens: 250,000
Compressed tokens: 4,000
Compression ratio: 62.5x
Tokens saved: 246,000
Cost saved: $2.4600
Processing time: 2347ms
Cache hit: False

--- LLM Answer ---
The key highlights from this document are:
1. Total monthly income: $20,200
2. Employment: Stable (5 years at current employer)
3. Debt-to-income ratio: 28%
4. Credit score: 740
5. Recommendation: APPROVED (confidence: 0.92)

--- Compressed Document (D-TOON) ---
doc{pages,entities}:
  id:loan_app_2024_12345
  pages[50]
    page1{entities[4],tables[1]}:
      bank_name;Chase Bank
      account_holder;John Doe
      ...

============================================================
SUCCESS!
============================================================
```

---

## Troubleshooting

### GPU Issues

If you see CUDA errors:

```bash
# Check GPU availability
nvidia-smi

# Set device explicitly
export OCR_DEVICE=cuda:0
```

### Cache Connection Issues

If Redis connection fails:

```bash
# Check Redis is running
redis-cli ping

# Use local cache instead
export CACHE_ENABLED=False
```

### API Rate Limits

If you hit API rate limits:

```bash
# Reduce batch size
export OCR_BATCH_SIZE=4

# Increase retry delay
export RETRY_DELAY=2.0
```

---

## Next Steps

1. **Read the documentation**: [README.md](../README.md)
2. **Explore the API**: [API Reference](https://edc.readthedocs.io/api)
3. **Deploy to production**: [Deployment Guide](../docs/deployment.md)
4. **Join the community**: [Discord](https://discord.gg/edc)

---

## Support

If you have questions or issues:

- **GitHub Issues**: [Report a bug](https://github.com/your-org/enterprise-doc-compressor/issues)
- **Email**: engineering@yourorg.com
- **Discord**: [Join our community](https://discord.gg/edc)

