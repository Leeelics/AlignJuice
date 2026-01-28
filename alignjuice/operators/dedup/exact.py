"""
Exact deduplication operator for AlignJuice.

Uses hash-based matching to identify and remove exact duplicates.
"""

from __future__ import annotations

import hashlib
from typing import Any

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("exact_dedup")
class ExactDedup(Operator):
    """
    Exact deduplication using hash matching.

    Removes samples with identical content based on hash comparison.
    Much faster than semantic dedup but only catches exact matches.
    """

    name = "exact_dedup"

    def __init__(
        self,
        field: str = "instruction",
        normalize: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize exact deduplication operator.

        Args:
            field: Field to use for deduplication (instruction, output, both, all)
            normalize: Whether to normalize text before hashing (lowercase, strip whitespace)
        """
        super().__init__(field=field, normalize=normalize, **kwargs)
        self.field = field
        self.normalize = normalize

    def _get_hash_text(self, sample: AlignmentSample) -> str:
        """Get text to hash from sample."""
        if self.field == "instruction":
            text = sample.instruction
        elif self.field == "output":
            text = sample.output
        elif self.field == "both":
            text = f"{sample.instruction}|||{sample.output}"
        elif self.field == "all":
            text = f"{sample.instruction}|||{sample.input}|||{sample.output}"
        else:
            text = sample.instruction

        if self.normalize:
            text = text.lower().strip()
            # Normalize whitespace
            text = " ".join(text.split())

        return text

    def _compute_hash(self, text: str) -> str:
        """Compute hash of text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def __call__(self, data: DataContainer) -> DataContainer:
        """Remove exact duplicates from data."""
        seen_hashes: set[str] = set()
        kept_samples: list[AlignmentSample] = []

        for sample in data:
            text = self._get_hash_text(sample)
            hash_value = self._compute_hash(text)

            if hash_value not in seen_hashes:
                seen_hashes.add(hash_value)
                kept_samples.append(sample)

        removed_count = len(data) - len(kept_samples)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept_samples),
            "removed_count": removed_count,
            "dedup_rate": removed_count / len(data) if len(data) > 0 else 0,
            "unique_hashes": len(seen_hashes),
        }

        return DataContainer(
            samples=kept_samples,
            provenance=data.provenance + [
                f"exact_dedup (field={self.field}): {len(data)} -> {len(kept_samples)}"
            ],
        )

    def find_duplicates(self, data: DataContainer) -> dict[str, list[int]]:
        """
        Find all duplicate groups in data (for analysis).

        Returns:
            Dict mapping hash to list of sample indices
        """
        hash_to_indices: dict[str, list[int]] = {}

        for i, sample in enumerate(data):
            text = self._get_hash_text(sample)
            hash_value = self._compute_hash(text)

            if hash_value not in hash_to_indices:
                hash_to_indices[hash_value] = []
            hash_to_indices[hash_value].append(i)

        # Return only groups with duplicates
        return {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}
