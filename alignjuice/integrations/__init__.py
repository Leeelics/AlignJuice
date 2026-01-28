"""Integrations module for AlignJuice."""

from alignjuice.integrations.llm import (
    LLMBackend,
    LLMResponse,
    OllamaBackend,
    OpenAIBackend,
    create_llm,
    get_llm,
)
from alignjuice.integrations.embeddings import (
    EmbeddingBackend,
    EmbeddingResult,
    SentenceTransformersBackend,
    OpenAIEmbeddingBackend,
    create_embedding,
    get_embedding,
)

__all__ = [
    # LLM
    "LLMBackend",
    "LLMResponse",
    "OllamaBackend",
    "OpenAIBackend",
    "create_llm",
    "get_llm",
    # Embeddings
    "EmbeddingBackend",
    "EmbeddingResult",
    "SentenceTransformersBackend",
    "OpenAIEmbeddingBackend",
    "create_embedding",
    "get_embedding",
]
