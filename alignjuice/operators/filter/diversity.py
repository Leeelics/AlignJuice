"""
Diversity filter operator for AlignJuice.

Ensures diverse sample selection using DaaR-style diversity operators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("diversity_filter")
class DiversityFilter(Operator):
    """
    Diversity-aware sample selection.

    Uses embedding-based clustering and selection to ensure diverse coverage.
    Implements DaaR (Diversity-aware Ranking) style selection.
    """

    name = "diversity_filter"

    def __init__(
        self,
        target_count: int = 1000,
        min_diversity_score: float = 0.3,
        embedding_backend: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        selection_method: str = "maxmin",
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize diversity filter.

        Args:
            target_count: Target number of samples to select
            min_diversity_score: Minimum diversity score for selection
            embedding_backend: Embedding backend to use
            embedding_model: Embedding model name
            selection_method: Selection method (maxmin, kmeans, random)
            batch_size: Batch size for embedding generation
        """
        super().__init__(
            target_count=target_count,
            min_diversity_score=min_diversity_score,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            selection_method=selection_method,
            batch_size=batch_size,
            **kwargs,
        )
        self.target_count = target_count
        self.min_diversity_score = min_diversity_score
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.selection_method = selection_method
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

    def __call__(self, data: DataContainer) -> DataContainer:
        """Select diverse samples from data."""
        if len(data) == 0:
            return data

        if len(data) <= self.target_count:
            return data

        # Generate embeddings
        texts = [f"{s.instruction} {s.output}" for s in data]
        result = self.embedder.embed_batched(texts, show_progress=True)
        embeddings = result.embeddings

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        # Select diverse samples
        if self.selection_method == "maxmin":
            selected_indices = self._maxmin_selection(embeddings)
        elif self.selection_method == "kmeans":
            selected_indices = self._kmeans_selection(embeddings)
        else:
            selected_indices = self._random_selection(len(data))

        # Compute diversity scores
        for i in selected_indices:
            data[i].metadata["diversity_selected"] = True

        kept_samples = [data[i] for i in selected_indices]

        # Compute overall diversity metric
        diversity_score = self._compute_diversity(embeddings[selected_indices])

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept_samples),
            "target_count": self.target_count,
            "selection_method": self.selection_method,
            "diversity_score": diversity_score,
        }

        return DataContainer(
            samples=kept_samples,
            provenance=data.provenance + [
                f"diversity_filter ({self.selection_method}): {len(data)} -> {len(kept_samples)}"
            ],
        )

    def _maxmin_selection(self, embeddings: np.ndarray) -> list[int]:
        """
        MaxMin diversity selection.

        Iteratively selects the sample that maximizes minimum distance to selected set.
        """
        n = len(embeddings)
        target = min(self.target_count, n)

        # Start with random sample
        selected = [np.random.randint(n)]
        remaining = set(range(n)) - set(selected)

        while len(selected) < target and remaining:
            # Compute distances from remaining to selected
            selected_embs = embeddings[selected]
            remaining_list = list(remaining)
            remaining_embs = embeddings[remaining_list]

            # Similarity matrix (remaining x selected)
            similarities = np.dot(remaining_embs, selected_embs.T)

            # Max similarity to any selected (we want to minimize this)
            max_similarities = np.max(similarities, axis=1)

            # Select the one with minimum max similarity (most different)
            best_idx = np.argmin(max_similarities)
            best_sample = remaining_list[best_idx]

            selected.append(best_sample)
            remaining.remove(best_sample)

        return selected

    def _kmeans_selection(self, embeddings: np.ndarray) -> list[int]:
        """
        K-means based diversity selection.

        Clusters samples and selects representatives from each cluster.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback to maxmin if sklearn not available
            return self._maxmin_selection(embeddings)

        n = len(embeddings)
        n_clusters = min(self.target_count, n)

        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Select sample closest to each centroid
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_embs = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(closest_idx)

        return selected

    def _random_selection(self, n: int) -> list[int]:
        """Random selection (baseline)."""
        target = min(self.target_count, n)
        return list(np.random.choice(n, target, replace=False))

    def _compute_diversity(self, embeddings: np.ndarray) -> float:
        """
        Compute diversity score for a set of embeddings.

        Uses average pairwise distance as diversity metric.
        """
        if len(embeddings) < 2:
            return 1.0

        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Get upper triangle (excluding diagonal)
        n = len(embeddings)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_indices]

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(pairwise_sims)
        return float(1.0 - avg_similarity)

    def compute_category_diversity(self, data: DataContainer) -> dict[str, int]:
        """Compute category distribution for diversity analysis."""
        categories: dict[str, int] = {}
        for sample in data:
            cat = sample.category
            categories[cat] = categories.get(cat, 0) + 1
        return categories
