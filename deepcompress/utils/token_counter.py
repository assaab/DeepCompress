"""
Token counting utilities for supported LLM providers.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCount:
    """Token counting result with provenance."""

    count: int
    provider: str
    model: str
    is_estimated: bool


def count_tokens(
    text: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str or None = None,
) -> TokenCount:
    """
    Count tokens for text using the requested provider/model when available.

    Falls back to a rough character-based estimate when the provider SDK,
    tokenizer, API key, or model tokenizer is unavailable.
    """
    normalized_provider = provider.lower()
    text = text or ""

    if not text:
        return TokenCount(
            count=0,
            provider=normalized_provider,
            model=model,
            is_estimated=False,
        )

    if normalized_provider == "openai":
        return _count_openai_tokens(text, model)
    if normalized_provider in {"claude", "anthropic"}:
        return _count_anthropic_tokens(text, normalized_provider, model, api_key)
    if normalized_provider in {"llama", "huggingface", "hf"}:
        return _count_huggingface_tokens(text, normalized_provider, model)

    return _estimated_count(text, normalized_provider, model)


def _count_openai_tokens(text: str, model: str) -> TokenCount:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        return TokenCount(
            count=len(encoding.encode(text)),
            provider="openai",
            model=model,
            is_estimated=False,
        )
    except Exception:
        return _estimated_count(text, "openai", model)


def _count_anthropic_tokens(
    text: str,
    provider: str,
    model: str,
    api_key: str or None,
) -> TokenCount:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _estimated_count(text, provider, model)

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        result = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return TokenCount(
            count=result.input_tokens,
            provider=provider,
            model=model,
            is_estimated=False,
        )
    except Exception:
        return _estimated_count(text, provider, model)


def _count_huggingface_tokens(text: str, provider: str, model: str) -> TokenCount:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return TokenCount(
            count=len(tokenizer.encode(text)),
            provider=provider,
            model=model,
            is_estimated=False,
        )
    except Exception:
        return _estimated_count(text, provider, model)


def _estimated_count(text: str, provider: str, model: str) -> TokenCount:
    count = max(1, (len(text) + 3) // 4) if text else 0
    return TokenCount(
        count=count,
        provider=provider,
        model=model,
        is_estimated=True,
    )
