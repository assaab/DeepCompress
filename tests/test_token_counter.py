import sys
from types import SimpleNamespace

from deepcompress.utils.token_counter import count_tokens


def test_openai_token_counter_uses_tiktoken(monkeypatch):
    class FakeEncoding:
        def encode(self, text):
            return text.split()

    fake_tiktoken = SimpleNamespace(
        encoding_for_model=lambda model: FakeEncoding(),
        get_encoding=lambda name: FakeEncoding(),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    result = count_tokens("one two three", provider="openai", model="gpt-4o")

    assert result.count == 3
    assert result.provider == "openai"
    assert result.model == "gpt-4o"
    assert result.is_estimated is False


def test_claude_without_api_key_is_estimated(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = count_tokens("abcd efgh", provider="claude", model="claude-3-5-sonnet")

    assert result.count > 0
    assert result.provider == "claude"
    assert result.is_estimated is True


def test_empty_text_returns_zero_tokens():
    result = count_tokens("", provider="openai", model="gpt-4o")

    assert result.count == 0
    assert result.is_estimated is False
