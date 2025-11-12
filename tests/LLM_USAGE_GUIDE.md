# 🤖 DeepCompress LLM Usage Guide

## ❓ Your Questions Answered

### 1. Which Model is DeepCompress Using?

**Default Model: `gpt-4o` (GPT-4 Omni)**

This is set in `deepcompress/core/config.py`:
```python
llm_model: str = Field(
    default="gpt-4o",
    description="LLM model identifier",
)
```

### 2. Why Don't I See API Calls in OpenAI Portal?

**Because your current code is NOT calling the OpenAI API!**

Your test code:
```python
compressor = DocumentCompressor(config)
result = await compressor.compress(pdf_path)
```

This **ONLY** does:
- ✅ OCR extraction (local DeepSeek-OCR model)
- ✅ Text compression (D-TOON optimization)
- ❌ **NO LLM/OpenAI API calls**

---

## 🔍 Three Ways to Use DeepCompress

### Method 1: Compression Only (NO LLM)
**When OpenAI API is called:** ❌ NEVER

```python
from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.compressor import DocumentCompressor

config = DeepCompressConfig(
    ocr_mode="small",
    # API key not needed for compression only
)

compressor = DocumentCompressor(config)
result = await compressor.compress("document.pdf")

# Result contains:
# - result.original_tokens
# - result.compressed_tokens
# - result.optimized_text
# But NO LLM answer!
```

**Cost:** FREE (no API calls)
**Use when:** You only want to compress documents for later use

---

### Method 2: Manual Compression + LLM Query (CALLS OpenAI)
**When OpenAI API is called:** ✅ When calling `llm_client.query()`

```python
from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.compressor import DocumentCompressor
from deepcompress.integrations.llm import LLMClient

config = DeepCompressConfig(
    ocr_mode="small",
    llm_api_key="sk-...",  # ⚠️ REQUIRED for LLM
    llm_model="gpt-4o",    # Specify model
)

# Step 1: Compress (no API call)
compressor = DocumentCompressor(config)
compressed = await compressor.compress("document.pdf")

# Step 2: Query LLM (THIS CALLS OpenAI API!)
llm_client = LLMClient(provider="openai", config=config)
await llm_client.initialize()

answer = await llm_client.query(
    context=compressed.optimized_text,
    question="What is this document about?"
)

print(f"Model used: {answer.model}")
print(f"Answer: {answer.text}")
print(f"Tokens used: {answer.tokens_used}")
```

**Cost:** OpenAI API charges apply
**Use when:** You want full control over the compression and query steps

---

### Method 3: One-Liner API (CALLS OpenAI)
**When OpenAI API is called:** ✅ Automatically during `compress_and_analyze()`

```python
from deepcompress import compress_and_analyze, DeepCompressConfig

config = DeepCompressConfig(
    llm_api_key="sk-...",  # ⚠️ REQUIRED
    llm_model="gpt-4o",
)

# All-in-one: compression + LLM query
result = await compress_and_analyze(
    file="document.pdf",
    query="Summarize this document",
    config=config
)

print(f"Answer: {result.answer}")
print(f"Tokens saved: {result.tokens_saved}")
```

**Cost:** OpenAI API charges apply
**Use when:** You want the simplest API

---

## ⚙️ How to Change the Model

### Option 1: Via Config
```python
config = DeepCompressConfig(
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",  # Cheaper!
    llm_temperature=0.0,       # More deterministic
    llm_max_tokens=500,        # Limit response length
)
```

### Option 2: Via Environment Variables
Create a `.env` file:
```bash
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=500
```

Then just:
```python
config = DeepCompressConfig()  # Auto-loads from .env
```

---

## 💰 Available Models & Pricing

| Model | Input Cost | Output Cost | Best For |
|-------|-----------|-------------|----------|
| **gpt-4o** (default) | $0.0025/1K | $0.01/1K | Best quality |
| **gpt-4o-mini** | $0.00015/1K | $0.0006/1K | Cost-effective |
| gpt-4-turbo | $0.01/1K | $0.03/1K | Previous gen |
| gpt-3.5-turbo | $0.0005/1K | $0.0015/1K | Fastest/cheapest |

**Recommendation:** Start with `gpt-4o-mini` for testing, upgrade to `gpt-4o` for production.

---

## 🧪 Testing Scripts

### Check Your Configuration
```bash
python tests/check_config.py
```

### Test Compression Only (No API Calls)
```bash
python tests/install_colab.py
```

### Test With LLM (Will Call OpenAI)
```bash
python tests/test_with_llm.py
```

⚠️ **Remember to set your API key in the script!**

---

## 🔐 How to Get & Set Your API Key

### 1. Get OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)

### 2. Set the Key

**Method A: In Code**
```python
config = DeepCompressConfig(llm_api_key="sk-...")
```

**Method B: Environment Variable**
```bash
export LLM_API_KEY="sk-..."  # Linux/Mac
set LLM_API_KEY=sk-...       # Windows CMD
$env:LLM_API_KEY="sk-..."    # Windows PowerShell
```

**Method C: .env File** (Recommended)
```bash
echo 'LLM_API_KEY=sk-...' > .env
```

---

## 📊 How to Verify API Calls in OpenAI Portal

1. Go to https://platform.openai.com/usage
2. Select "Activity" tab
3. You should see:
   - Model used (e.g., `gpt-4o`)
   - Tokens consumed
   - Timestamp of request
   - Cost

If you don't see any activity:
- ✅ Check you're using Method 2 or 3 (not just compression)
- ✅ Verify API key is correct
- ✅ Make sure you're calling `llm_client.query()` or `compress_and_analyze()`

---

## 🐛 Common Issues

### "Why is my API key empty?"
```python
OPENAI_API_KEY = ""  # ❌ This is empty!
```

You need to actually set it:
```python
OPENAI_API_KEY = "sk-proj-..."  # ✅ Actual key
```

### "I set the API key but no calls appear"
You're probably only calling `compressor.compress()` which doesn't use the LLM.

**Fix:** Use one of these instead:
```python
# Option 1: Manual LLM query
llm_client = LLMClient(provider="openai", config=config)
await llm_client.initialize()
answer = await llm_client.query(compressed.optimized_text, "Your question")

# Option 2: One-liner
result = await compress_and_analyze("doc.pdf", "Your question", config=config)
```

### "How do I know which model it's using?"
```python
# Print the config
config = DeepCompressConfig(llm_api_key="sk-...")
print(f"Using model: {config.llm_model}")

# Or check the LLM response
answer = await llm_client.query(...)
print(f"Model used: {answer.model}")
```

---

## ✅ Quick Checklist

Before running tests with LLM:
- [ ] Have valid OpenAI API key
- [ ] Set `llm_api_key` in config
- [ ] Know which model you're using (check `llm_model`)
- [ ] Using Method 2 or 3 (not just compression)
- [ ] Have credits in OpenAI account

---

## 📝 Summary

| Your Code | LLM Called? | OpenAI API? | Cost |
|-----------|-------------|-------------|------|
| `compressor.compress()` | ❌ No | ❌ No | FREE |
| `llm_client.query()` | ✅ Yes | ✅ Yes | Paid |
| `compress_and_analyze()` | ✅ Yes | ✅ Yes | Paid |

**Default model:** `gpt-4o`
**To change:** Set `llm_model` in config
**To verify:** Check OpenAI usage portal after running Method 2 or 3






