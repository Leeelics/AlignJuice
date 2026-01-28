"""
Semantic deduplication operator for AlignJuice.

Uses embedding similarity to identify and remove near-duplicate samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("semantic_dedup")
class SemanticDedup(Operator):
    """
    Semantic deduplication using embedding similarity.

    Removes samples that are semantically similar above a threshold.
    Uses cosine similarity between embeddings.
    """

    name = "semantic_dedup"

    def __init__(
        self,
        threshold: float = 0.95,
        embedding_backend: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        field: str = "instruction",
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize semantic deduplication operator.

        Args:
            threshold: Similarity threshold (0-1). Samples above this are duplicates.
            embedding_backend: Embedding backend to use
            embedding_model: Embedding model name
            field: Field to use for similarity (instruction, output, or both)
            batch_size: Batch size for embedding generation
        """
        super().__init__(
            threshold=threshold,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            field=field,
            batch_size=batch_size,
            **kwargs,
        )
        self.threshold = threshold
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.field = field
        self.batch_size = batch_size
        self._embedder = None

    @property
    def embedder(self) -> Any:
        """Lazy load embedder."""
        if self._embedder is None:
            from alignjuice.integrations.embeddings import get_embedding
            self._embedder = get_embedding(
                backend=self.embedding_backend,
                model=self.embedding_model,
                batch_size=self.batch_size,
            )
        return self._embedder

    def _get_text(self, sample: AlignmentSample) -> str:
        """Get text to embed from sample."""
        if self.field == "instruction":
            return sample.instruction
        elif self.field == "output":
            return sample.output
        elif self.field == "both":
            return f"{sample.instruction} {sample.output}"
        else:
            return sample.instruction

    def __call__(self, data: DataContainer) -> DataContainer:
        """Remove semantic duplicates from data."""
        if len(data) == 0:
            return data

        # Get texts to embed
        texts = [self._get_text(s) for s in data]

        # Generate embeddings
        result = self.embedder.embed_batched(texts, show_progress=True)
        embeddings = result.embeddings

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        # Find duplicates using greedy selection
        keep_indices = self._greedy_dedup(embeddings)

        # Keep only non-duplicate samples
        kept_samples = [data[i] for i in keep_indices]
        removed_count = len(data) - len(kept_samples)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept_samples),
            "removed_count": removed_count,
            "dedup_rate": removed_count / len(data) if len(data) > 0 else 0,
            "threshold": self.threshold,
        }

        return DataContainer(
            samples=kept_samples,
            provenance=data.provenance + [
                f"semantic_dedup (threshold={self.threshold}): {len(data)} -> {len(kept_samples)}"
            ],
        )

    def _greedy_dedup(self, embeddings: np.ndarray) -> list[int]:
        """
        Greedy deduplication: keep first occurrence, remove similar ones.

        Args:
            embeddings: Normalized embeddings (n, d)

        Returns:
            List of indices to keep
        """
        n = len(embeddings)
        keep_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep_mask[i]:
                continue

            # Compute similarity with remaining samples
            similarities = np.dot(embeddings[i], embeddings[i + 1 :].T)

            # Mark similar samples as duplicates
            duplicate_indices = np.where(similarities >= self.threshold)[0] + i + 1
            keep_mask[duplicate_indices] = False

        return list(np.where(keep_mask)[0])

    def find_duplicates(self, data: DataContainer) -> list[tuple[int, int, float]]:
        """
        Find all duplicate pairs in data (for analysis).

        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        texts = [self._get_text(s) for s in data]
        result = self.embedder.embed_batched(texts, show_progress=True)
        embeddings = result.embeddings

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Find pairs above threshold
        duplicates = []
        n = len(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.threshold:
                    duplicates.append((i, j, float(similarity_matrix[i, j])))

        return duplicates
