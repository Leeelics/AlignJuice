"""
Base embedding backend interface for AlignJuice.

Provides abstract base class for all embedding integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: np.ndarray  # Shape: (n_texts, embedding_dim)
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else len(self.embeddings)

    @property
    def count(self) -> int:
        """Get number of embeddings."""
        return self.embeddings.shape[0] if len(self.embeddings.shape) > 1 else 1


class EmbeddingBackend(ABC):
    """
    Abstract base class for embedding backends.

    All embedding integrations (sentence-transformers, OpenAI) should inherit from this.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize embedding backend.

        Args:
            model: Model name/identifier
            batch_size: Batch size for embedding
            normalize: Whether to L2-normalize embeddings
            **kwargs: Additional backend-specific parameters
        """
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.extra_params = kwargs
        self._dimension: int | None = None

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings array
        """
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            1D numpy array of embedding
        """
        result = self.embed([text])
        return result.embeddings[0]

    def embed_batched(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Generate embeddings in batches.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingResult with all embeddings
        """
        if len(texts) <= self.batch_size:
            return self.embed(texts)

        all_embeddings = []
        total_tokens = 0

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Embedding", unit="batch")
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i : i + self.batch_size]
            result = self.embed(batch)
            all_embeddings.append(result.embeddings)
            total_tokens += result.usage.get("total_tokens", 0)

        combined = np.vstack(all_embeddings)

        return EmbeddingResult(
            embeddings=combined,
            model=self.model,
            usage={"total_tokens": total_tokens},
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy loaded)."""
        if self._dimension is None:
            # Generate a test embedding to get dimension
            result = self.embed(["test"])
            self._dimension = result.dimension
        return self._dimension

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (n, d)
            embeddings2: Second set of embeddings (m, d)

        Returns:
            Similarity matrix (n, m)
        """
        # Normalize if not already
        if self.normalize:
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            embeddings1 = embeddings1 / (norm1 + 1e-9)
            embeddings2 = embeddings2 / (norm2 + 1e-9)

        return np.dot(embeddings1, embeddings2.T)

    def find_similar(
        self,
        query: str | np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find most similar items in corpus.

        Args:
            query: Query text or embedding
            corpus_embeddings: Corpus embeddings (n, d)
            top_k: Number of results to return

        Returns:
            Tuple of (indices, scores)
        """
        if isinstance(query, str):
            query_emb = self.embed_single(query)
        else:
            query_emb = query

        # Compute similarities
        similarities = self.similarity(
            query_emb.reshape(1, -1),
            corpus_embeddings,
        )[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        return top_indices, top_scores

    def health_check(self) -> bool:
        """Check if the backend is available and working."""
        try:
            result = self.embed(["test"])
            return result.embeddings.shape[0] == 1
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, batch_size={self.batch_size})"
