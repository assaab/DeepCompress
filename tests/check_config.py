"""
Quick script to check DeepCompress configuration and model settings
"""
from deepcompress.core.config import DeepCompressConfig


def show_config():
    """Display all LLM-related configuration"""
    
    print("=" * 70)
    print("⚙️  DeepCompress Configuration Inspector")
    print("=" * 70)
    
    # Default config
    print("\n📋 DEFAULT CONFIGURATION:")
    config = DeepCompressConfig()
    
    print(f"\n🤖 LLM Settings:")
    print(f"  Provider:       {config.llm_provider}")
    print(f"  Model:          {config.llm_model}")
    print(f"  Max tokens:     {config.llm_max_tokens}")
    print(f"  Temperature:    {config.llm_temperature}")
    print(f"  API key set:    {'Yes (hidden)' if config.llm_api_key else 'No'}")
    
    print(f"\n🔍 OCR Settings:")
    print(f"  Model:          {config.ocr_model}")
    print(f"  Mode:           {config.ocr_mode}")
    print(f"  Device:         {config.ocr_device}")
    print(f"  Batch size:     {config.ocr_batch_size}")
    
    print(f"\n💾 Cache Settings:")
    print(f"  Enabled:        {config.cache_enabled}")
    print(f"  URL:            {config.cache_url}")
    print(f"  TTL:            {config.cache_ttl}s")
    
    print(f"\n🗄️  Vector DB Settings:")
    print(f"  Provider:       {config.vector_db_provider}")
    print(f"  Index:          {config.vector_db_index_name}")
    
    # Custom config example
    print("\n" + "=" * 70)
    print("📝 CUSTOM CONFIGURATION EXAMPLE:")
    print("=" * 70)
    
    custom_config = DeepCompressConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",  # Cheaper model
        llm_api_key="sk-your-key-here",
        llm_temperature=0.0,  # More deterministic
        llm_max_tokens=1000,
        ocr_mode="small",
    )
    
    print(f"\n🤖 Custom LLM Settings:")
    print(f"  Provider:       {custom_config.llm_provider}")
    print(f"  Model:          {custom_config.llm_model}")
    print(f"  Max tokens:     {custom_config.llm_max_tokens}")
    print(f"  Temperature:    {custom_config.llm_temperature}")
    
    # Available models
    print("\n" + "=" * 70)
    print("📚 AVAILABLE OPENAI MODELS:")
    print("=" * 70)
    
    models = [
        ("gpt-4o", "$0.0025/1K in", "$0.01/1K out", "Default - Best quality"),
        ("gpt-4o-mini", "$0.00015/1K in", "$0.0006/1K out", "Cheaper, faster"),
        ("gpt-4-turbo", "$0.01/1K in", "$0.03/1K out", "Previous gen"),
        ("gpt-3.5-turbo", "$0.0005/1K in", "$0.0015/1K out", "Cheapest, fastest"),
    ]
    
    print("\nModel              Input Cost      Output Cost     Notes")
    print("-" * 70)
    for model, input_cost, output_cost, note in models:
        print(f"{model:18} {input_cost:15} {output_cost:15} {note}")
    
    print("\n💡 To use a different model, set llm_model in your config:")
    print('   config = DeepCompressConfig(llm_model="gpt-4o-mini")')
    
    print("\n" + "=" * 70)
    print("✅ Configuration check complete!")
    print("=" * 70)


if __name__ == "__main__":
    show_config()






