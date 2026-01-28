"""
Ollama LLM backend for AlignJuice.

Provides integration with locally running Ollama models.
"""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from alignjuice.integrations.llm.base import LLMBackend, LLMResponse
from alignjuice.core.registry import register_llm


@register_llm("ollama")
class OllamaBackend(LLMBackend):
    """
    Ollama backend for local LLM inference.

    Requires Ollama to be running locally (default: http://localhost:11434).
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "phi3:medium",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text using Ollama API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                text=data.get("response", ""),
                model=self.model,
                usage={
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                },
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "eval_duration": data.get("eval_duration"),
                },
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}") from e

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Generate text for multiple prompts sequentially."""
        # Ollama doesn't support native batching, so we process sequentially
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream generated text token by token."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama streaming error: {e}") from e

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat-style generation using Ollama chat API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=self.model,
                usage={
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                },
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama chat API error: {e}") from e

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError:
            return []

    def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
