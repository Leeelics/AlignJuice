"""
Base LLM backend interface for AlignJuice.

Provides abstract base class for all LLM integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator


@dataclass
class LLMResponse:
    """Response from LLM generation."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All LLM integrations (Ollama, vLLM, OpenAI, Anthropic) should inherit from this.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """
        Initialize LLM backend.

        Args:
            model: Model name/identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse with generated text
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of LLMResponse objects
        """
        pass

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Stream generated text token by token.

        Default implementation falls back to non-streaming generate.
        Override in subclasses for true streaming support.
        """
        response = self.generate(prompt, **kwargs)
        yield response.text

    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Async version of generate.

        Default implementation runs sync version.
        Override in subclasses for true async support.
        """
        return self.generate(prompt, **kwargs)

    async def abatch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """
        Async version of batch_generate.

        Default implementation runs sync version.
        Override in subclasses for true async support.
        """
        return self.batch_generate(prompts, **kwargs)

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """
        Async streaming generation.

        Default implementation falls back to non-streaming.
        """
        response = await self.agenerate(prompt, **kwargs)
        yield response.text

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse with assistant's reply
        """
        # Default: convert to single prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, **kwargs)

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def health_check(self) -> bool:
        """Check if the backend is available and working."""
        try:
            response = self.generate("Hello", max_tokens=5)
            return len(response.text) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, temperature={self.temperature})"
