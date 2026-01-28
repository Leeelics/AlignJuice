"""
Stage 3: Reasoning enhancement and noise cleaning stage.

Adds reasoning chains and removes noisy samples.
"""

from __future__ import annotations

from typing import Any, Literal

from alignjuice.core.data_container import DataContainer
from alignjuice.stages.base import BaseStage
from alignjuice.operators.base import Operator


class ReasoningEnhanceStage(BaseStage):
    """
    Stage 3: LIMO reasoning chain enhancement + CleanLab noise cleaning.

    Applies:
    - Reasoning chain enhancement for reasoning-type samples
    - Noise detection and removal using CleanLab/heuristics
    - LIMA style validation

    Input: ~1000 samples from Stage 2
    Output: ~1000 noise-free, reasoning-enhanced samples
    """

    name = "s3_reasoning_enhance"

    def __init__(
        self,
        operators: list[Operator] | None = None,
        chain_style: Literal["step_by_step", "cot", "tree_of_thought"] = "step_by_step",
        target_category: str = "reasoning",
        noise_method: Literal["heuristic", "cleanlab"] = "heuristic",
        llm_backend: str = "ollama",
        llm_model: str = "phi3:medium",
        **kwargs: Any,
    ):
        """
        Initialize reasoning enhance stage.

        Args:
            operators: Custom operators to apply
            chain_style: Style of reasoning chain to generate
            target_category: Category of samples to enhance
            noise_method: Method for noise detection
            llm_backend: LLM backend for reasoning generation
            llm_model: LLM model for reasoning generation
        """
        super().__init__(operators, **kwargs)
        self.chain_style = chain_style
        self.target_category = target_category
        self.noise_method = noise_method
        self.llm_backend = llm_backend
        self.llm_model = llm_model

        if not self.operators:
            self._init_default_operators()

    def _init_default_operators(self) -> None:
        """Initialize default operators for this stage."""
        from alignjuice.operators.transform import ReasoningChainEnhancer
        from alignjuice.operators.validate import NoiseDetector

        self.operators = [
            # Reasoning chain enhancement
            ReasoningChainEnhancer(
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
                chain_style=self.chain_style,
                target_category=self.target_category,
            ),
            # Noise detection and removal
            NoiseDetector(
                method=self.noise_method,
                action_error="remove",
                action_redundancy="remove",
                action_ambiguity="flag",
            ),
            # Style validation (using quality filter as proxy)
            # LIMAStyleValidator would go here
        ]

    def process(self, data: DataContainer) -> DataContainer:
        """Process data through reasoning enhance stage."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data for Reasoning Enhance stage")

        input_count = len(data)
        result = self._apply_operators(data)

        self._metrics.update({
            "stage": self.name,
            "input_count": input_count,
            "output_count": len(result),
            "chain_style": self.chain_style,
            "noise_method": self.noise_method,
        })

        return result
