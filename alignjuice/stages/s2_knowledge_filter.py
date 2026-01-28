"""
Stage 2: Knowledge filtering and LLM synthesis stage.

Filters by knowledge density and synthesizes improvements for low-knowledge samples.
"""

from __future__ import annotations

from typing import Any

from alignjuice.core.data_container import DataContainer
from alignjuice.stages.base import BaseStage
from alignjuice.operators.base import Operator


class KnowledgeFilterStage(BaseStage):
    """
    Stage 2: Knowledge density filtering + LLM synthesis.

    Applies:
    - Embedding-based knowledge density scoring
    - LLM synthesis for low-knowledge samples
    - Rank and select top-k by knowledge score

    Input: ~1500 samples from Stage 1
    Output: ~1000 high knowledge density samples
    """

    name = "s2_knowledge_filter"

    def __init__(
        self,
        operators: list[Operator] | None = None,
        top_k: int = 1000,
        synthesis_enabled: bool = True,
        synthesis_threshold: float = 0.5,
        llm_backend: str = "ollama",
        llm_model: str = "phi3:medium",
        embedding_backend: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ):
        """
        Initialize knowledge filter stage.

        Args:
            operators: Custom operators to apply
            top_k: Number of top samples to keep
            synthesis_enabled: Whether to synthesize low-knowledge samples
            synthesis_threshold: Threshold below which to synthesize
            llm_backend: LLM backend for synthesis
            llm_model: LLM model for synthesis
            embedding_backend: Embedding backend for knowledge scoring
            embedding_model: Embedding model for knowledge scoring
        """
        super().__init__(operators, **kwargs)
        self.top_k = top_k
        self.synthesis_enabled = synthesis_enabled
        self.synthesis_threshold = synthesis_threshold
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model

        if not self.operators:
            self._init_default_operators()

    def _init_default_operators(self) -> None:
        """Initialize default operators for this stage."""
        from alignjuice.operators.filter import KnowledgeFilter
        from alignjuice.operators.transform import LLMSynthesis

        operators = []

        # Knowledge scoring and filtering
        operators.append(KnowledgeFilter(
            top_k=self.top_k * 2,  # Keep more initially for synthesis
            embedding_backend=self.embedding_backend,
            embedding_model=self.embedding_model,
        ))

        # LLM synthesis for low-knowledge samples
        if self.synthesis_enabled:
            operators.append(LLMSynthesis(
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
                synthesis_style="textbook",
                target_samples="low_knowledge",
                quality_threshold=self.synthesis_threshold,
            ))

        # Final knowledge filter to get top_k
        operators.append(KnowledgeFilter(
            top_k=self.top_k,
            embedding_backend=self.embedding_backend,
            embedding_model=self.embedding_model,
        ))

        self.operators = operators

    def process(self, data: DataContainer) -> DataContainer:
        """Process data through knowledge filter stage."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data for Knowledge Filter stage")

        input_count = len(data)
        result = self._apply_operators(data)

        self._metrics.update({
            "stage": self.name,
            "input_count": input_count,
            "output_count": len(result),
            "top_k": self.top_k,
            "synthesis_enabled": self.synthesis_enabled,
        })

        return result
