"""
DeepCompress Compression KPIs Test
===================================

⚠️ IMPORTANT: This test ONLY does compression - NO LLM API calls!

What this test does:
  ✅ Downloads a PDF
  ✅ Runs OCR extraction (local DeepSeek-OCR model)
  ✅ Compresses content (D-TOON optimization)
  ✅ Shows compression statistics

What this test does NOT do:
  ❌ Call OpenAI API
  ❌ Query any LLM
  ❌ Cost you money

The llm_api_key is set in config but NEVER USED because we only call
compressor.compress(), which doesn't use the LLM.

To actually test LLM integration, see: tests/test_with_llm.py
"""

import asyncio
from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.compressor import DocumentCompressor
import requests
import tempfile
import os


async def test_compression_kpis():
    """
    Test document compression without LLM.
    This will NOT show up in your OpenAI usage portal.
    """
    
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print("=" * 70)
    print("📊 DeepCompress - Compression KPIs")
    print("=" * 70)
    print("\n⚠️  NOTE: This test does NOT call OpenAI API!")
    print("    It only tests local OCR + compression.")
    print("    For LLM testing, see: tests/test_with_llm.py\n")
    
    print("[1/3] Downloading test PDF...")
    response = requests.get(pdf_url)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(response.content)
        pdf_path = f.name
        print(f"✓ PDF saved: {pdf_path}")
    
    try:
        # API key is set but NOT used in this test
        OPENAI_API_KEY = ""  # Empty is fine for compression-only
        
        print("\n[2/3] Configuring DeepCompress...")
        config = DeepCompressConfig(
            ocr_mode="small",
            ocr_device="cuda:0",
            enable_flash_attention=False,
            use_vector_db=False,
            vector_db_provider='none',
            # This API key is configured but NEVER used in compress()
            llm_api_key=OPENAI_API_KEY
        )
        print("✓ Configuration ready")
        
        print("\n[3/3] Processing document...")
        print("  → Running OCR extraction (local model)")
        print("  → Compressing content (D-TOON optimization)")
        print("  → NO LLM API calls will be made!")
        
        # THIS IS THE KEY LINE:
        # compressor.compress() ONLY does OCR + compression
        # It does NOT call OpenAI API
        compressor = DocumentCompressor(config)
        result = await compressor.compress(pdf_path)
        
        print("  ✓ Processing complete!")
        
        # Display all the KPIs
        print("\n" + "=" * 70)
        print(f"📄 RESULTS FOR: {os.path.basename(pdf_path)}")
        print("=" * 70)
        
        print("\n📈 COMPRESSION STATISTICS:")
        print(f"  Original tokens:        {result.original_tokens:>12,}")
        print(f"  Compressed tokens:      {result.compressed_tokens:>12,}")
        print(f"  Tokens saved:           {result.tokens_saved:>12,}")
        print(f"  Compression ratio:      {result.compression_ratio:>12.2f}x")
        
        print("\n💰 COST ANALYSIS:")
        print("  (These are PROJECTED savings if you were to use these tokens with an LLM)")
        
        # Calculate costs using GPT-4o pricing (projected, not actual)
        cost_per_1k = 0.0025  # $0.0025 per 1K input tokens (GPT-4o)
        original_cost = (result.original_tokens / 1000) * cost_per_1k
        compressed_cost = (result.compressed_tokens / 1000) * cost_per_1k
        cost_saved = original_cost - compressed_cost
        
        print(f"  Original cost:          ${original_cost:>12.4f}")
        print(f"  Compressed cost:        ${compressed_cost:>12.4f}")
        print(f"  Cost saved (USD):       ${cost_saved:>12.4f}")
        
        if original_cost > 0:
            savings_pct = (cost_saved / original_cost) * 100
            print(f"  Savings percentage:     {savings_pct:>12.1f}%")
        
        print("\n⏱️  PERFORMANCE:")
        print(f"  Processing time:        {result.processing_time_ms:>12.0f}ms")
        
        if result.processing_time_ms > 0:
            throughput = result.original_tokens / (result.processing_time_ms / 1000)
            print(f"  Throughput:             {throughput:>12.0f} tokens/sec")
        
        print(f"  Cache hit:              {str(result.cache_hit):>12}")
        print(f"  Document ID:            {result.document_id}")
        
        print("\n📝 COMPRESSED OUTPUT PREVIEW:")
        print("-" * 70)
        preview_len = min(400, len(result.optimized_text))
        print(result.optimized_text[:preview_len])
        if len(result.optimized_text) > preview_len:
            remaining = len(result.optimized_text) - preview_len
            print(f"\n... (+ {remaining:,} more characters)")
        print("-" * 70)
        
        print(f"\n📊 Total compressed text: {len(result.optimized_text):,} characters")
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Compression KPIs displayed!")
        print("=" * 70)
        print("\n💡 WHAT HAPPENED:")
        print("  ✅ OCR extracted text from PDF (local DeepSeek-OCR)")
        print("  ✅ D-TOON compressed the text")
        print("  ❌ NO OpenAI API was called")
        print("  ❌ NO charges incurred")
        print("\n📚 TO TEST LLM INTEGRATION:")
        print("  Run: python tests/test_with_llm.py")
        print("  (This WILL call OpenAI and show in your portal)")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
            print(f"\n🧹 Cleaned up temporary file")


# Run the test
if __name__ == "__main__":
    success = asyncio.run(test_compression_kpis())
    print(f"\n{'✅ TEST PASSED!' if success else '❌ TEST FAILED'}")






