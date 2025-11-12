"""
Test DeepCompress with actual LLM API calls
This version WILL call OpenAI API and show up in your portal
"""
import asyncio
from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.compressor import DocumentCompressor
from deepcompress.integrations.llm import LLMClient
import requests
import tempfile
import os


async def test_compression_with_llm():
    """
    Test that actually calls OpenAI API
    """
    
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print("=" * 70)
    print("📊 DeepCompress - WITH LLM API Testing")
    print("=" * 70)
    
    print("\n[1/4] Downloading test PDF...")
    response = requests.get(pdf_url)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(response.content)
        pdf_path = f.name
        print(f"✓ PDF saved: {pdf_path}")
    
    try:
        # PUT YOUR ACTUAL API KEY HERE
        OPENAI_API_KEY = "sk-..."  # Replace with your actual key
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
            print("\n❌ ERROR: Please set your OPENAI_API_KEY in the script!")
            print("   Get one at: https://platform.openai.com/api-keys")
            return False

        print("\n[2/4] Configuring DeepCompress...")
        config = DeepCompressConfig(
            ocr_mode="small",
            ocr_device="cuda:0",
            enable_flash_attention=False,
            use_vector_db=False,
            vector_db_provider='none',
            llm_api_key=OPENAI_API_KEY,
            llm_model="gpt-4o",  # Explicitly set model
            llm_temperature=0.1,
            llm_max_tokens=500,
            # OCR generation limits to prevent hallucinations
            ocr_max_new_tokens=1024,  # Reduced from default 2048
            ocr_temperature=0.0,  # Greedy decoding for more accuracy
            ocr_repetition_penalty=1.5,  # Strong penalty against repetition
            # Batch processing for better performance
            ocr_batch_size=4,  # Process 4 pages concurrently
        )
        print(f"✓ Configuration ready")
        print(f"  → LLM Provider: openai")
        print(f"  → LLM Model: {config.llm_model}")
        print(f"  → Temperature: {config.llm_temperature}")
        print(f"  → Batch Size: {config.ocr_batch_size} pages/batch")
        
        print("\n[3/4] Compressing document (NO LLM call yet)...")
        compressor = DocumentCompressor(config)
        compressed = await compressor.compress(pdf_path)
        print("  ✓ Compression complete!")
        print(f"  → Compressed to {compressed.compressed_tokens} tokens")
        
        print("\n[4/4] Querying LLM (THIS CALLS OPENAI API)...")
        llm_client = LLMClient(provider="openai", config=config)
        await llm_client.initialize()
        
        # This is where the actual OpenAI API call happens!
        query = "What is this document about? Provide a brief summary."
        print(f"  → Question: {query}")
        print(f"  → Sending to OpenAI {config.llm_model}...")
        
        llm_response = await llm_client.query(
            context=compressed.optimized_text,
            question=query
        )
        
        print("  ✓ LLM response received!")
        
        # Display results
        print("\n" + "=" * 70)
        print("📄 RESULTS")
        print("=" * 70)
        
        print("\n📊 COMPRESSION STATS:")
        print(f"  Original tokens:        {compressed.original_tokens:>12,}")
        print(f"  Compressed tokens:      {compressed.compressed_tokens:>12,}")
        print(f"  Tokens saved:           {compressed.tokens_saved:>12,}")
        print(f"  Compression ratio:      {compressed.compression_ratio:>12.2f}x")
        
        print("\n🤖 LLM RESPONSE:")
        print("-" * 70)
        print(f"Model used:     {llm_response.model}")
        print(f"Tokens used:    {llm_response.tokens_used}")
        print(f"Response time:  {llm_response.response_time_ms:.0f}ms")
        print("\nAnswer:")
        print(llm_response.text)
        print("-" * 70)
        
        print("\n💰 COST ANALYSIS:")
        # GPT-4o pricing
        input_cost_per_1k = 0.0025  # $0.0025 per 1K input tokens
        output_cost_per_1k = 0.01   # $0.01 per 1K output tokens
        
        prompt_tokens = llm_response.metadata.get('prompt_tokens', 0)
        completion_tokens = llm_response.metadata.get('completion_tokens', 0)
        
        input_cost = (prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (completion_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        print(f"  Prompt tokens:          {prompt_tokens:>12,}")
        print(f"  Completion tokens:      {completion_tokens:>12,}")
        print(f"  Input cost:             ${input_cost:>12.6f}")
        print(f"  Output cost:            ${output_cost:>12.6f}")
        print(f"  Total API cost:         ${total_cost:>12.6f}")
        
        # Calculate what it would have cost without compression
        uncompressed_input_cost = (compressed.original_tokens / 1000) * input_cost_per_1k
        savings = uncompressed_input_cost - input_cost
        
        print(f"\n  Without compression:    ${uncompressed_input_cost:>12.6f}")
        print(f"  Cost saved:             ${savings:>12.6f}")
        if uncompressed_input_cost > 0:
            savings_pct = (savings / uncompressed_input_cost) * 100
            print(f"  Savings:                {savings_pct:>12.1f}%")
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! OpenAI API was called - check your portal!")
        print("   Go to: https://platform.openai.com/usage")
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


async def test_compression_only():
    """
    Original test - NO LLM calls, just compression
    """
    
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print("\n\n" + "=" * 70)
    print("📊 DeepCompress - Compression ONLY (No LLM)")
    print("=" * 70)
    
    print("\n[1/2] Downloading test PDF...")
    response = requests.get(pdf_url)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(response.content)
        pdf_path = f.name
    
    try:
        print("\n[2/2] Compressing document...")
        config = DeepCompressConfig(
            ocr_mode="small",
            ocr_device="cuda:0",
            enable_flash_attention=False,
            use_vector_db=False,
            vector_db_provider='none',
            # OCR generation limits to prevent hallucinations
            ocr_max_new_tokens=1024,
            ocr_temperature=0.0,
            ocr_repetition_penalty=1.5,
            # Batch processing
            ocr_batch_size=4,
        )
        
        compressor = DocumentCompressor(config)
        result = await compressor.compress(pdf_path)
        
        print("\n✅ Compression complete (NO OpenAI API calls made)")
        print(f"  → Compressed to {result.compressed_tokens} tokens")
        print(f"  → Compression ratio: {result.compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False
        
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


async def main():
    """
    Run both tests to show the difference
    """
    print("\n🔬 DeepCompress Testing Suite\n")
    
    # Test 1: Compression only (no LLM)
    success1 = await test_compression_only()
    
    # Test 2: With LLM API calls
    success2 = await test_compression_with_llm()
    
    print("\n\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"Compression only test:  {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"LLM integration test:   {'✅ PASSED' if success2 else '❌ FAILED'}")
    print("\n💡 KEY TAKEAWAY:")
    print("  - compressor.compress() = NO API calls (just OCR + compression)")
    print("  - llm_client.query() = ACTUAL OpenAI API call")
    print("  - compress_and_analyze() = Both compression + LLM query")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())


