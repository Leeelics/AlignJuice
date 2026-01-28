"""
LLM factory module for AlignJuice.

Provides factory functions for creating LLM backends with fallback support.
"""

from __future__ import annotations

from typing import Any

from alignjuice.integrations.llm.base import LLMBackend, LLMResponse
from alignjuice.core.registry import Registry
from alignjuice.config.schema import LLMConfig


def create_llm(config: LLMConfig | dict[str, Any]) -> LLMBackend:
    """
    Create an LLM backend from configuration.

    Args:
        config: LLMConfig object or dict with backend configuration

    Returns:
        Configured LLMBackend instance
    """
    if isinstance(config, dict):
        config = LLMConfig(**config)

    backend_name = config.backend
    llm_cls = Registry.get_llm(backend_name)

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    llm = llm_cls(**kwargs)

    # Wrap with fallback if configured
    if config.fallback_backend:
        fallback_cls = Registry.get_llm(config.fallback_backend)
        fallback_kwargs = kwargs.copy()
        # Fallback might need different model name
        if config.fallback_backend == "openai":
            fallback_kwargs["model"] = "gpt-4o-mini"
        fallback = fallback_cls(**fallback_kwargs)
        return FallbackLLM(primary=llm, fallback=fallback)

    return llm


def get_llm(
    backend: str = "ollama",
    model: str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """
    Get an LLM backend by name.

    Args:
        backend: Backend name (ollama, openai, anthropic, vllm)
        model: Model name (uses default if not specified)
        **kwargs: Additional backend parameters

    Returns:
        LLMBackend instance
    """
    llm_cls = Registry.get_llm(backend)

    # Default models per backend
    default_models = {
        "ollama": "phi3:medium",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "vllm": "meta-llama/Llama-2-7b-hf",
    }

    if model is None:
        model = default_models.get(backend, "default")

    return llm_cls(model=model, **kwargs)


class FallbackLLM(LLMBackend):
    """
    LLM wrapper that falls back to secondary backend on failure.
    """

    name = "fallback"

    def __init__(self, primary: LLMBackend, fallback: LLMBackend):
        # Don't call super().__init__ as we're wrapping other backends
        self.primary = primary
        self.fallback = fallback
        self.model = f"{primary.model} (fallback: {fallback.model})"
        self.temperature = primary.temperature
        self.max_tokens = primary.max_tokens

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate with fallback on failure."""
        try:
            return self.primary.generate(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed ({e}), falling back to {self.fallback.name}")
            return self.fallback.generate(prompt, **kwargs)

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Batch generate with fallback on failure."""
        try:
            return self.primary.batch_generate(prompts, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed ({e}), falling back to {self.fallback.name}")
            return self.fallback.batch_generate(prompts, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat with fallback on failure."""
        try:
            return self.primary.chat(messages, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed ({e}), falling back to {self.fallback.name}")
            return self.fallback.chat(messages, **kwargs)

    def health_check(self) -> bool:
        """Check if at least one backend is healthy."""
        return self.primary.health_check() or self.fallback.health_check()
