"""
Embedding factory module for AlignJuice.

Provides factory functions for creating embedding backends.
"""

from __future__ import annotations

from typing import Any

from alignjuice.integrations.embeddings.base import EmbeddingBackend
from alignjuice.core.registry import Registry
from alignjuice.config.schema import EmbeddingConfig


def create_embedding(config: EmbeddingConfig | dict[str, Any]) -> EmbeddingBackend:
    """
    Create an embedding backend from configuration.

    Args:
        config: EmbeddingConfig object or dict with backend configuration

    Returns:
        Configured EmbeddingBackend instance
    """
    if isinstance(config, dict):
        config = EmbeddingConfig(**config)

    backend_name = config.backend

    # Map config backend names to registry names
    backend_map = {
        "sentence_transformers": "sentence_transformers",
        "openai": "openai_embeddings",
    }
    registry_name = backend_map.get(backend_name, backend_name)

    embedding_cls = Registry.get_embedding(registry_name)

    kwargs: dict[str, Any] = {
        "model": config.model,
        "batch_size": config.batch_size,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key

    return embedding_cls(**kwargs)


def get_embedding(
    backend: str = "sentence_transformers",
    model: str | None = None,
    **kwargs: Any,
) -> EmbeddingBackend:
    """
    Get an embedding backend by name.

    Args:
        backend: Backend name (sentence_transformers, openai_embeddings)
        model: Model name (uses default if not specified)
        **kwargs: Additional backend parameters

    Returns:
        EmbeddingBackend instance
    """
    # Map common names
    backend_map = {
        "sentence_transformers": "sentence_transformers",
        "st": "sentence_transformers",
        "huggingface": "huggingface",
        "hf": "huggingface",
        "openai": "openai_embeddings",
        "openai_embeddings": "openai_embeddings",
    }
    registry_name = backend_map.get(backend, backend)

    embedding_cls = Registry.get_embedding(registry_name)

    # Default models per backend
    default_models = {
        "sentence_transformers": "all-MiniLM-L6-v2",
        "huggingface": "all-MiniLM-L6-v2",
        "openai_embeddings": "text-embedding-3-small",
    }

    if model is None:
        model = default_models.get(registry_name, "all-MiniLM-L6-v2")

    return embedding_cls(model=model, **kwargs)
