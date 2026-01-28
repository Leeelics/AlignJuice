"""
Quality filter operator for AlignJuice.

Filters samples based on quality scores.
"""

from __future__ import annotations

from typing import Any, Callable

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import FilterOperator


@register_operator("quality_filter")
class QualityFilter(FilterOperator):
    """
    Filter samples based on quality score.

    Uses heuristic quality metrics or model-based scoring.
    """

    name = "quality_filter"

    def __init__(
        self,
        threshold: float = 0.8,
        scorer: str = "heuristic",
        min_instruction_length: int = 10,
        max_instruction_length: int = 2000,
        min_output_length: int = 20,
        max_output_length: int = 8000,
        **kwargs: Any,
    ):
        """
        Initialize quality filter.

        Args:
            threshold: Minimum quality score to keep (0-1)
            scorer: Scoring method (heuristic, model)
            min_instruction_length: Minimum instruction character length
            max_instruction_length: Maximum instruction character length
            min_output_length: Minimum output character length
            max_output_length: Maximum output character length
        """
        super().__init__(
            threshold=threshold,
            scorer=scorer,
            min_instruction_length=min_instruction_length,
            max_instruction_length=max_instruction_length,
            min_output_length=min_output_length,
            max_output_length=max_output_length,
            **kwargs,
        )
        self.threshold = threshold
        self.scorer = scorer
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length

    def compute_quality_score(self, sample: AlignmentSample) -> float:
        """
        Compute quality score for a sample.

        Args:
            sample: Sample to score

        Returns:
            Quality score between 0 and 1
        """
        if self.scorer == "heuristic":
            return self._heuristic_score(sample)
        else:
            return self._heuristic_score(sample)

    def _heuristic_score(self, sample: AlignmentSample) -> float:
        """Compute heuristic quality score."""
        scores = []

        # Length checks
        instr_len = len(sample.instruction)
        output_len = len(sample.output)

        # Instruction length score
        if instr_len < self.min_instruction_length:
            scores.append(0.3)
        elif instr_len > self.max_instruction_length:
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Output length score
        if output_len < self.min_output_length:
            scores.append(0.3)
        elif output_len > self.max_output_length:
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Content quality checks
        instruction = sample.instruction.lower()
        output = sample.output.lower()

        # Check for repetition in output
        words = output.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            scores.append(min(1.0, unique_ratio * 1.5))
        else:
            scores.append(0.8)

        # Check for common low-quality patterns
        low_quality_patterns = [
            "i cannot", "i can't", "as an ai",
            "i don't have access", "i'm not able",
            "sorry, but", "i apologize",
        ]
        has_low_quality = any(p in output for p in low_quality_patterns)
        scores.append(0.5 if has_low_quality else 1.0)

        # Check instruction has question mark or imperative
        has_question = "?" in sample.instruction
        has_imperative = any(
            sample.instruction.lower().startswith(w)
            for w in ["write", "create", "explain", "describe", "list", "how", "what", "why"]
        )
        scores.append(1.0 if (has_question or has_imperative) else 0.7)

        # Average all scores
        return sum(scores) / len(scores)

    def should_keep(self, sample: AlignmentSample) -> bool:
        """Determine if sample meets quality threshold."""
        score = self.compute_quality_score(sample)
        # Store score in metadata for later analysis
        sample.metadata["quality_score"] = score
        return score >= self.threshold

    def __call__(self, data: DataContainer) -> DataContainer:
        """Filter samples based on quality score."""
        kept = []
        scores = []

        for sample in data:
            score = self.compute_quality_score(sample)
            sample.metadata["quality_score"] = score
            scores.append(score)

            if score >= self.threshold:
                kept.append(sample)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept),
            "filtered_count": len(data) - len(kept),
            "filter_rate": (len(data) - len(kept)) / len(data) if len(data) > 0 else 0,
            "threshold": self.threshold,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }

        return DataContainer(
            samples=kept,
            provenance=data.provenance + [
                f"quality_filter (threshold={self.threshold}): {len(data)} -> {len(kept)}"
            ],
        )
