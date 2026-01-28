"""
Sentence Transformers embedding backend for AlignJuice.

Provides local embedding generation using sentence-transformers library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alignjuice.integrations.embeddings.base import EmbeddingBackend, EmbeddingResult
from alignjuice.core.registry import register_embedding


@register_embedding("sentence_transformers")
class SentenceTransformersBackend(EmbeddingBackend):
    """
    Sentence Transformers backend for local embedding generation.

    Uses the sentence-transformers library for efficient local embeddings.
    """

    name = "sentence_transformers"

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Sentence Transformers backend.

        Args:
            model: Model name from HuggingFace or local path
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            device: Device to use (cuda, cpu, mps, or None for auto)
            **kwargs: Additional parameters passed to SentenceTransformer
        """
        super().__init__(model, batch_size, normalize, **kwargs)
        self.device = device
        self._model = None

    @property
    def model_instance(self) -> Any:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

            self._model = SentenceTransformer(
                self.model,
                device=self.device,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

        return self._model

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings using sentence-transformers."""
        model = self.model_instance

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model,
            usage={"total_tokens": sum(len(t.split()) for t in texts)},
            metadata={"device": str(model.device)},
        )

    def embed_with_pooling(
        self,
        texts: list[str],
        pooling: str = "mean",
    ) -> EmbeddingResult:
        """
        Generate embeddings with custom pooling strategy.

        Args:
            texts: Texts to embed
            pooling: Pooling strategy (mean, max, cls)

        Returns:
            EmbeddingResult
        """
        # For sentence-transformers, pooling is handled by the model
        # This method is for compatibility with other backends
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model_instance  # This sets _dimension
        return self._dimension

    def health_check(self) -> bool:
        """Check if model can be loaded and used."""
        try:
            _ = self.model_instance
            result = self.embed(["test"])
            return result.embeddings.shape[0] == 1
        except Exception:
            return False


@register_embedding("huggingface")
class HuggingFaceBackend(SentenceTransformersBackend):
    """Alias for SentenceTransformersBackend."""

    name = "huggingface"
