"""Embeddings integrations module for AlignJuice."""

from alignjuice.integrations.embeddings.base import EmbeddingBackend, EmbeddingResult
from alignjuice.integrations.embeddings.sentence_transformers import (
    SentenceTransformersBackend,
    HuggingFaceBackend,
)
from alignjuice.integrations.embeddings.openai_embeddings import OpenAIEmbeddingBackend
from alignjuice.integrations.embeddings.factory import create_embedding, get_embedding

__all__ = [
    "EmbeddingBackend",
    "EmbeddingResult",
    "SentenceTransformersBackend",
    "HuggingFaceBackend",
    "OpenAIEmbeddingBackend",
    "create_embedding",
    "get_embedding",
]
