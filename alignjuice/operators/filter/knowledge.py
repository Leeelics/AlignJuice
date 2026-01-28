"""
Knowledge density filter operator for AlignJuice.

Filters and scores samples based on knowledge density using embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("knowledge_filter")
class KnowledgeFilter(Operator):
    """
    Filter and rank samples by knowledge density.

    Uses embedding similarity to reference corpus to estimate knowledge content.
    """

    name = "knowledge_filter"

    def __init__(
        self,
        top_k: int = 1000,
        min_score: float | None = None,
        embedding_backend: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        reference_embeddings: np.ndarray | None = None,
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize knowledge filter.

        Args:
            top_k: Number of top samples to keep (by knowledge score)
            min_score: Minimum knowledge score (alternative to top_k)
            embedding_backend: Embedding backend to use
            embedding_model: Embedding model name
            reference_embeddings: Pre-computed reference corpus embeddings
            batch_size: Batch size for embedding generation
        """
        super().__init__(
            top_k=top_k,
            min_score=min_score,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            batch_size=batch_size,
            **kwargs,
        )
        self.top_k = top_k
        self.min_score = min_score
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.reference_embeddings = reference_embeddings
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

    def compute_knowledge_score(self, sample: AlignmentSample, embedding: np.ndarray) -> float:
        """
        Compute knowledge density score for a sample.

        Combines multiple signals:
        1. Information entropy (lexical diversity)
        2. Semantic richness (embedding norm)
        3. Reference similarity (if reference corpus provided)

        Args:
            sample: Sample to score
            embedding: Pre-computed embedding for the sample

        Returns:
            Knowledge score between 0 and 1
        """
        scores = []

        # 1. Information entropy (lexical diversity)
        text = f"{sample.instruction} {sample.output}"
        words = text.lower().split()
        if len(words) > 0:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            probs = np.array(list(word_freq.values())) / len(words)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            # Normalize entropy (typical range 0-10)
            entropy_score = min(1.0, entropy / 8.0)
            scores.append(entropy_score)

        # 2. Content length score (longer = potentially more knowledge)
        content_length = len(sample.output)
        length_score = min(1.0, content_length / 1000)
        scores.append(length_score)

        # 3. Unique concept density
        unique_words = len(set(words))
        if len(words) > 0:
            unique_ratio = unique_words / len(words)
            scores.append(unique_ratio)

        # 4. Reference similarity (if available)
        if self.reference_embeddings is not None:
            # Compute max similarity to reference corpus
            similarities = np.dot(self.reference_embeddings, embedding)
            max_sim = float(np.max(similarities))
            scores.append(max_sim)

        return sum(scores) / len(scores) if scores else 0.5

    def __call__(self, data: DataContainer) -> DataContainer:
        """Filter and rank samples by knowledge density."""
        if len(data) == 0:
            return data

        # Generate embeddings for all samples
        texts = [f"{s.instruction} {s.output}" for s in data]
        result = self.embedder.embed_batched(texts, show_progress=True)
        embeddings = result.embeddings

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        # Compute knowledge scores
        scores = []
        for i, sample in enumerate(data):
            score = self.compute_knowledge_score(sample, embeddings[i])
            sample.metadata["knowledge_score"] = score
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select samples
        if self.min_score is not None:
            selected_indices = [i for i, s in scores if s >= self.min_score]
        else:
            selected_indices = [i for i, _ in scores[: self.top_k]]

        kept_samples = [data[i] for i in selected_indices]
        all_scores = [s for _, s in scores]

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept_samples),
            "filtered_count": len(data) - len(kept_samples),
            "top_k": self.top_k,
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "min_kept_score": min(s for i, s in scores if i in selected_indices) if selected_indices else 0,
            "max_score": max(all_scores) if all_scores else 0,
        }

        return DataContainer(
            samples=kept_samples,
            provenance=data.provenance + [
                f"knowledge_filter (top_k={self.top_k}): {len(data)} -> {len(kept_samples)}"
            ],
        )

    def set_reference_corpus(self, texts: list[str]) -> None:
        """
        Set reference corpus for knowledge scoring.

        Args:
            texts: List of reference texts (e.g., Wikipedia chunks)
        """
        result = self.embedder.embed_batched(texts, show_progress=True)
        embeddings = result.embeddings

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.reference_embeddings = embeddings / (norms + 1e-9)
