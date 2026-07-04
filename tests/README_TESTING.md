# 🧪 DeepCompress Testing Guide

## 📁 Available Test Scripts

### 1. `check_config.py` - Configuration Inspector
**What it does:** Shows all configuration settings without running any tests

```bash
python tests/check_config.py
```

**Features:**
- ✅ Shows default model (gpt-4o)
- ✅ Lists available models & pricing
- ✅ Displays current config settings
- ✅ No API calls, no costs

---

### 2. `test_compression_kpis_explained.py` - Compression Only
**What it does:** Tests OCR + compression WITHOUT calling LLM

```bash
python tests/test_compression_kpis_explained.py
```

**Features:**
- ✅ OCR extraction
- ✅ D-TOON compression
- ✅ Compression statistics
- ❌ NO OpenAI API calls
- ❌ NO costs

**Use when:** You want to test compression performance only

---

### 3. `test_with_llm.py` - Full LLM Integration Test
**What it does:** Tests compression AND actual OpenAI API calls

```bash
python tests/test_with_llm.py
```

**Features:**
- ✅ OCR extraction
- ✅ D-TOON compression
- ✅ OpenAI API call
- ✅ Shows which model is used
- ✅ Displays actual API costs
- ⚠️ **REQUIRES valid API key**
- ⚠️ **WILL cost money** (usually < $0.01)

**Use when:** You want to test end-to-end with actual LLM queries

---

## 🔑 Setting Up API Key

Before running `test_with_llm.py`, set your API key:

### Method 1: Edit the Script
Open `tests/test_with_llm.py` and replace:
```python
OPENAI_API_KEY = "sk-..."  # Replace with your actual key
```

### Method 2: Environment Variable
```bash
# Linux/Mac
export LLM_API_KEY="sk-..."

# Windows PowerShell
$env:LLM_API_KEY="sk-..."
```

### Method 3: .env File
Create `tests/.env`:
```
LLM_API_KEY=sk-...
```

---

## ❓ FAQ - Your Questions Answered

### Q: Which model is DeepCompress using?
**A:** Default is `gpt-4o` (GPT-4 Omni)

To check: `python tests/check_config.py`

To change:
```python
config = DeepCompressConfig(llm_model="gpt-4o-mini")
```

---

### Q: Why don't I see API calls in OpenAI portal?
**A:** Because `compressor.compress()` doesn't call the LLM!

Only these make API calls:
```python
# Method 1: Manual LLM query
llm_client = LLMClient(provider="openai", config=config)
await llm_client.query(text, question)

# Method 2: All-in-one
await compress_and_analyze(file, query, config=config)
```

---

### Q: How do I verify OpenAI API is being called?
**A:** Run `python tests/test_with_llm.py` then check:
1. Go to https://platform.openai.com/usage
2. Click "Activity" tab
3. You should see the request with model name and tokens

---

### Q: What's the difference between the test scripts?

| Script | OpenAI API? | Costs? | Shows Model? |
|--------|-------------|--------|--------------|
| `check_config.py` | ❌ No | FREE | ✅ Yes |
| `test_compression_kpis_explained.py` | ❌ No | FREE | ❌ No |
| `test_with_llm.py` | ✅ Yes | Paid | ✅ Yes |

---

## 🚀 Quick Start

### Step 1: Check Configuration
```bash
python tests/check_config.py
```
This shows you which model will be used (default: gpt-4o)

### Step 2: Test Compression Only
```bash
python tests/test_compression_kpis_explained.py
```
This tests OCR + compression without API calls (FREE)

### Step 3: Test With LLM
1. Get API key from https://platform.openai.com/api-keys
2. Edit `tests/test_with_llm.py` and add your key
3. Run:
```bash
python tests/test_with_llm.py
```
4. Check OpenAI portal to see the API call

---

## 💰 Cost Expectations

### Compression Only Tests
- **Cost:** $0.00 (FREE)
- No API calls made

### LLM Integration Tests
- **Cost:** ~$0.001 - $0.01 per test
- Depends on:
  - Model used (gpt-4o vs gpt-4o-mini)
  - Document size
  - Query complexity

**Example costs for a 10-page document:**
- With compression: ~$0.002
- Without compression: ~$0.05
- **Savings: 96%**

---

## 📚 Additional Documentation

- **Full LLM Usage Guide:** `tests/LLM_USAGE_GUIDE.md`
- **Main README:** `../README.md`
- **Configuration Reference:** `deepcompress/core/config.py`

---

## 🐛 Troubleshooting

### "No API calls showing in portal"
→ Make sure you're running `test_with_llm.py`, not the compression-only test

### "API key error"
→ Check your key starts with `sk-` and has no spaces

### "CUDA out of memory"
→ Use `ocr_device="cpu"` or `ocr_mode="small"`

### "Which model is being used?"
→ Run `python tests/check_config.py` or check the LLM response

---

## ✅ Summary

| Goal | Script | API Calls? |
|------|--------|------------|
| Check model settings | `check_config.py` | ❌ No |
| Test compression | `test_compression_kpis_explained.py` | ❌ No |
| Test full pipeline | `test_with_llm.py` | ✅ Yes |
| Learn about LLM usage | Read `LLM_USAGE_GUIDE.md` | - |

**Key Takeaway:** 
- `compressor.compress()` = NO API calls
- `llm_client.query()` = ACTUAL API calls
- Default model = `gpt-4o`








