"""
Base operator class for AlignJuice.

Provides abstract base class for all data processing operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alignjuice.core.data_container import DataContainer, AlignmentSample


class Operator(ABC):
    """
    Abstract base class for all operators.

    Operators are the building blocks of data processing pipelines.
    Each operator takes a DataContainer and returns a transformed DataContainer.
    """

    name: str = "base_operator"

    def __init__(self, **kwargs: Any):
        """Initialize operator with optional parameters."""
        self.params = kwargs
        self._metrics: dict[str, Any] = {}

    @abstractmethod
    def __call__(self, data: DataContainer) -> DataContainer:
        """
        Process data and return transformed DataContainer.

        Args:
            data: Input DataContainer

        Returns:
            Transformed DataContainer
        """
        pass

    def process_sample(self, sample: AlignmentSample) -> AlignmentSample | None:
        """
        Process a single sample.

        Override this for sample-level operations.
        Return None to filter out the sample.

        Args:
            sample: Input sample

        Returns:
            Transformed sample or None to filter
        """
        return sample

    @property
    def metrics(self) -> dict[str, Any]:
        """Return metrics collected during processing."""
        return self._metrics.copy()

    def explain(self) -> str:
        """Explain what this operator does."""
        doc = self.__doc__ or "No description available"
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}: {doc.strip().split(chr(10))[0]}\nParameters: {params_str or 'none'}"

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class FilterOperator(Operator):
    """Base class for filtering operators."""

    @abstractmethod
    def should_keep(self, sample: AlignmentSample) -> bool:
        """
        Determine if a sample should be kept.

        Args:
            sample: Sample to evaluate

        Returns:
            True to keep, False to filter out
        """
        pass

    def __call__(self, data: DataContainer) -> DataContainer:
        """Filter samples based on should_keep predicate."""
        kept = []
        filtered_count = 0

        for sample in data:
            if self.should_keep(sample):
                kept.append(sample)
            else:
                filtered_count += 1

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept),
            "filtered_count": filtered_count,
            "filter_rate": filtered_count / len(data) if len(data) > 0 else 0,
        }

        return DataContainer(
            samples=kept,
            provenance=data.provenance + [f"{self.name}: {len(data)} -> {len(kept)}"],
        )


class TransformOperator(Operator):
    """Base class for transformation operators."""

    @abstractmethod
    def transform(self, sample: AlignmentSample) -> AlignmentSample:
        """
        Transform a single sample.

        Args:
            sample: Sample to transform

        Returns:
            Transformed sample
        """
        pass

    def __call__(self, data: DataContainer) -> DataContainer:
        """Transform all samples."""
        transformed = []

        for sample in data:
            transformed.append(self.transform(sample))

        self._metrics = {
            "input_count": len(data),
            "output_count": len(transformed),
        }

        return DataContainer(
            samples=transformed,
            provenance=data.provenance + [f"{self.name}: transformed {len(transformed)} samples"],
        )
