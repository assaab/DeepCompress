"""
LLM integrations (OpenAI, Claude, Llama).
"""

import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import LLMError
from deepcompress.models.response import LLMResponse


class LLMClient:
    """
    Multi-provider LLM client supporting OpenAI, Claude, and Llama.

    Features:
    - Async API calls
    - Automatic retries with exponential backoff
    - Token usage tracking
    - Response time monitoring
    """

    def __init__(self, provider: str, config: DeepCompressConfig) -> None:
        self.provider = provider.lower()
        self.config = config
        self._client: Any = None

    async def initialize(self) -> None:
        """Initialize LLM client."""
        if self.provider == "openai":
            await self._init_openai()
        elif self.provider == "claude":
            await self._init_claude()
        elif self.provider == "llama":
            await self._init_llama()
        else:
            raise LLMError(
                f"Unsupported LLM provider: {self.provider}",
                details={"supported": ["openai", "claude", "llama"]},
            )

    async def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.config.llm_api_key)
        except ImportError:
            raise LLMError(
                "OpenAI not installed. Install with: pip install edc[llm]"
            )

    async def _init_claude(self) -> None:
        """Initialize Claude client."""
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.config.llm_api_key)
        except ImportError:
            raise LLMError(
                "Anthropic not installed. Install with: pip install edc[llm]"
            )

    async def _init_llama(self) -> None:
        """Initialize Llama client (placeholder for local deployment)."""
        raise LLMError("Llama integration not yet implemented")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def query(
        self,
        context: str,
        question: str,
        system_prompt: str  or None = None,
    ) -> LLMResponse:
        """
        Query LLM with compressed document context.

        Args:
            context: Compressed document (D-TOON format)
            question: Question to ask
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with answer and metadata

        Example:
            >>> client = LLMClient("openai", config)
            >>> response = await client.query(
            ...     context="doc{...}",
            ...     question="What is the total income?"
            ... )
            >>> print(response.text)
        """
        if self._client is None:
            await self.initialize()

        start_time = time.time()

        try:
            if self.provider == "openai":
                response = await self._query_openai(context, question, system_prompt)
            elif self.provider == "claude":
                response = await self._query_claude(context, question, system_prompt)
            else:
                raise LLMError(f"Unsupported provider: {self.provider}")

            processing_time_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                text=response["text"],
                tokens_used=response["tokens_used"],
                processing_time_ms=processing_time_ms,
                model=response["model"],
                metadata=response.get("metadata", {}),
            )

        except Exception as e:
            raise LLMError(
                f"LLM query failed: {str(e)}",
                details={"provider": self.provider, "error": str(e)},
            )

    async def _query_openai(
        self,
        context: str,
        question: str,
        system_prompt: str  or None,
    ) -> dict[str, Any]:
        """Query OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are analyzing a compressed document. Provide accurate, concise answers based on the provided data.",
            })

        messages.append({
            "role": "user",
            "content": f"Document data:\n{context}\n\nQuestion: {question}",
        })

        response = await self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )

        return {
            "text": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "model": response.model,
            "metadata": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    async def _query_claude(
        self,
        context: str,
        question: str,
        system_prompt: str  or None,
    ) -> dict[str, Any]:
        """Query Claude API."""
        if system_prompt is None:
            system_prompt = "You are analyzing a compressed document. Provide accurate, concise answers based on the provided data."

        message = await self._client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Document data:\n{context}\n\nQuestion: {question}",
                }
            ],
        )

        return {
            "text": message.content[0].text,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "model": message.model,
            "metadata": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }

    async def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self._client is None:
            await self.initialize()

        try:
            if self.provider == "openai":
                response = await self._client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text,
                )
                return response.data[0].embedding
            else:
                raise LLMError(
                    f"Embeddings not supported for provider: {self.provider}"
                )

        except Exception as e:
            raise LLMError(
                f"Embedding generation failed: {str(e)}",
                details={"provider": self.provider, "error": str(e)},
            )

