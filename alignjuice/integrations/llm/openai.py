"""
OpenAI LLM backend for AlignJuice.

Provides integration with OpenAI API (and compatible APIs).
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator

import httpx

from alignjuice.integrations.llm.base import LLMBackend, LLMResponse
from alignjuice.core.registry import register_llm


@register_llm("openai")
class OpenAIBackend(LLMBackend):
    """
    OpenAI backend for cloud LLM inference.

    Supports OpenAI API and compatible endpoints (Azure, local proxies, etc.).
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text using OpenAI completions API."""
        # Use chat completions API (more widely supported)
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Generate text for multiple prompts."""
        # OpenAI doesn't have native batching for chat, process sequentially
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat-style generation using OpenAI chat completions API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})

            return LLMResponse(
                text=choice.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "id": data.get("id"),
                },
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            raise RuntimeError(f"OpenAI API error ({e.response.status_code}): {error_body}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenAI API connection error: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream generated text token by token."""
        messages = [{"role": "user", "content": prompt}]
        yield from self.stream_chat(messages, **kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completions token by token."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        import json
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenAI streaming error: {e}") from e

    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async generation."""
        messages = [{"role": "user", "content": prompt}]
        return await self.achat(messages, **kwargs)

    async def achat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Async chat completion."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                choice = data.get("choices", [{}])[0]
                usage = data.get("usage", {})

                return LLMResponse(
                    text=choice.get("message", {}).get("content", ""),
                    model=data.get("model", self.model),
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                )
            except httpx.HTTPError as e:
                raise RuntimeError(f"OpenAI async API error: {e}") from e

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = self._client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPError:
            return []

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            response = self._client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
