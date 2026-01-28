"""
Base stage class for AlignJuice pipeline stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alignjuice.core.data_container import DataContainer
from alignjuice.operators.base import Operator


class BaseStage(ABC):
    """
    Base class for pipeline stages.

    Each stage represents a major processing step in the alignment data pipeline.
    """

    name: str = "base_stage"

    def __init__(self, operators: list[Operator] | None = None, **kwargs: Any):
        """
        Initialize stage.

        Args:
            operators: List of operators to apply in this stage
            **kwargs: Additional stage configuration
        """
        self.operators = operators or []
        self.config = kwargs
        self._metrics: dict[str, Any] = {}

    @abstractmethod
    def process(self, data: DataContainer) -> DataContainer:
        """
        Process data through this stage.

        Args:
            data: Input DataContainer

        Returns:
            Processed DataContainer
        """
        pass

    def validate_input(self, data: DataContainer) -> bool:
        """Validate input data before processing."""
        return len(data) > 0

    def validate_output(self, data: DataContainer) -> bool:
        """Validate output data after processing."""
        return len(data) > 0

    @property
    def metrics(self) -> dict[str, Any]:
        """Return metrics collected during processing."""
        return self._metrics.copy()

    def _apply_operators(self, data: DataContainer) -> DataContainer:
        """Apply all operators in sequence."""
        current_data = data
        for operator in self.operators:
            current_data = operator(current_data)
            # Collect operator metrics
            self._metrics[f"op_{operator.name}"] = operator.metrics
        return current_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operators={len(self.operators)})"
