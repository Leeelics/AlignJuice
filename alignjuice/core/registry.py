"""
Registry module for AlignJuice.

Provides a centralized registry for operators, metrics, LLM backends, and embeddings.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """
    Centralized registry for framework components.

    Supports registration of:
    - Operators (data processing functions)
    - Metrics (quality measurement functions)
    - LLM backends (language model interfaces)
    - Embedding backends (vector embedding interfaces)
    """

    _instance: Registry | None = None
    _operators: dict[str, type] = {}
    _metrics: dict[str, type] = {}
    _llms: dict[str, type] = {}
    _embeddings: dict[str, type] = {}

    def __new__(cls) -> Registry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_operator(cls, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register an operator class.

        Usage:
            @register_operator("semantic_dedup")
            class SemanticDedup(Operator):
                ...
        """
        def decorator(operator_cls: type[T]) -> type[T]:
            cls._operators[name] = operator_cls
            return operator_cls
        return decorator

    @classmethod
    def register_metric(cls, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a metric class.

        Usage:
            @register_metric("knowledge_density")
            class KnowledgeDensity(Metric):
                ...
        """
        def decorator(metric_cls: type[T]) -> type[T]:
            cls._metrics[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def register_llm(cls, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register an LLM backend class.

        Usage:
            @register_llm("ollama")
            class OllamaBackend(LLMBackend):
                ...
        """
        def decorator(llm_cls: type[T]) -> type[T]:
            cls._llms[name] = llm_cls
            return llm_cls
        return decorator

    @classmethod
    def register_embedding(cls, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register an embedding backend class.

        Usage:
            @register_embedding("sentence_transformers")
            class SentenceTransformersBackend(EmbeddingBackend):
                ...
        """
        def decorator(embedding_cls: type[T]) -> type[T]:
            cls._embeddings[name] = embedding_cls
            return embedding_cls
        return decorator

    @classmethod
    def get_operator(cls, name: str) -> type:
        """Get operator class by name."""
        if name not in cls._operators:
            raise KeyError(f"Operator '{name}' not found. Available: {list(cls._operators.keys())}")
        return cls._operators[name]

    @classmethod
    def get_metric(cls, name: str) -> type:
        """Get metric class by name."""
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name]

    @classmethod
    def get_llm(cls, name: str) -> type:
        """Get LLM backend class by name."""
        if name not in cls._llms:
            raise KeyError(f"LLM backend '{name}' not found. Available: {list(cls._llms.keys())}")
        return cls._llms[name]

    @classmethod
    def get_embedding(cls, name: str) -> type:
        """Get embedding backend class by name."""
        if name not in cls._embeddings:
            raise KeyError(f"Embedding backend '{name}' not found. Available: {list(cls._embeddings.keys())}")
        return cls._embeddings[name]

    @classmethod
    def list_operators(cls) -> list[str]:
        """List all registered operators."""
        return list(cls._operators.keys())

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metrics."""
        return list(cls._metrics.keys())

    @classmethod
    def list_llms(cls) -> list[str]:
        """List all registered LLM backends."""
        return list(cls._llms.keys())

    @classmethod
    def list_embeddings(cls) -> list[str]:
        """List all registered embedding backends."""
        return list(cls._embeddings.keys())

    @classmethod
    def create_operator(cls, name: str, **kwargs: Any) -> Any:
        """Create an operator instance by name."""
        operator_cls = cls.get_operator(name)
        return operator_cls(**kwargs)

    @classmethod
    def create_metric(cls, name: str, **kwargs: Any) -> Any:
        """Create a metric instance by name."""
        metric_cls = cls.get_metric(name)
        return metric_cls(**kwargs)

    @classmethod
    def create_llm(cls, name: str, **kwargs: Any) -> Any:
        """Create an LLM backend instance by name."""
        llm_cls = cls.get_llm(name)
        return llm_cls(**kwargs)

    @classmethod
    def create_embedding(cls, name: str, **kwargs: Any) -> Any:
        """Create an embedding backend instance by name."""
        embedding_cls = cls.get_embedding(name)
        return embedding_cls(**kwargs)


# Convenience decorators at module level
def register_operator(name: str) -> Callable[[type[T]], type[T]]:
    """Register an operator class."""
    return Registry.register_operator(name)


def register_metric(name: str) -> Callable[[type[T]], type[T]]:
    """Register a metric class."""
    return Registry.register_metric(name)


def register_llm(name: str) -> Callable[[type[T]], type[T]]:
    """Register an LLM backend class."""
    return Registry.register_llm(name)


def register_embedding(name: str) -> Callable[[type[T]], type[T]]:
    """Register an embedding backend class."""
    return Registry.register_embedding(name)
