"""
LLM synthesis operator for AlignJuice.

Uses LLM to synthesize or enhance sample outputs.
"""

from __future__ import annotations

from typing import Any, Literal

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("llm_synthesis")
class LLMSynthesis(Operator):
    """
    LLM-based synthesis for enhancing or generating outputs.

    Can be used to:
    - Enhance low-quality outputs
    - Generate textbook-style explanations
    - Improve response structure
    """

    name = "llm_synthesis"

    SYNTHESIS_PROMPTS = {
        "textbook": """You are an expert educator. Rewrite the following response in a clear,
textbook-style format that is educational and well-structured.

Original instruction: {instruction}
Original response: {output}

Provide an improved, educational response:""",

        "enhance": """Improve the following response to be more helpful, accurate, and well-structured.
Keep the same general meaning but enhance clarity and completeness.

Instruction: {instruction}
Original response: {output}

Improved response:""",

        "expand": """Expand the following response with more details and examples while maintaining accuracy.

Instruction: {instruction}
Original response: {output}

Expanded response:""",
    }

    def __init__(
        self,
        llm_backend: str = "ollama",
        llm_model: str = "phi3:medium",
        synthesis_style: Literal["textbook", "enhance", "expand"] = "textbook",
        target_samples: Literal["all", "low_quality", "low_knowledge"] = "low_quality",
        quality_threshold: float = 0.7,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """
        Initialize LLM synthesis operator.

        Args:
            llm_backend: LLM backend to use
            llm_model: LLM model name
            synthesis_style: Style of synthesis (textbook, enhance, expand)
            target_samples: Which samples to synthesize (all, low_quality, low_knowledge)
            quality_threshold: Threshold below which to synthesize (for low_quality mode)
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            llm_backend=llm_backend,
            llm_model=llm_model,
            synthesis_style=synthesis_style,
            target_samples=target_samples,
            quality_threshold=quality_threshold,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.synthesis_style = synthesis_style
        self.target_samples = target_samples
        self.quality_threshold = quality_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    @property
    def llm(self) -> Any:
        """Lazy load LLM."""
        if self._llm is None:
            from alignjuice.integrations.llm import get_llm
            self._llm = get_llm(
                backend=self.llm_backend,
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    def should_synthesize(self, sample: AlignmentSample) -> bool:
        """Determine if a sample should be synthesized."""
        if self.target_samples == "all":
            return True
        elif self.target_samples == "low_quality":
            score = sample.metadata.get("quality_score", 1.0)
            return score < self.quality_threshold
        elif self.target_samples == "low_knowledge":
            score = sample.metadata.get("knowledge_score", 1.0)
            return score < self.quality_threshold
        return False

    def synthesize(self, sample: AlignmentSample) -> AlignmentSample:
        """Synthesize improved output for a sample."""
        prompt_template = self.SYNTHESIS_PROMPTS.get(
            self.synthesis_style,
            self.SYNTHESIS_PROMPTS["enhance"],
        )

        prompt = prompt_template.format(
            instruction=sample.instruction,
            output=sample.output,
        )

        response = self.llm.generate(prompt)
        new_output = response.text.strip()

        # Create new sample with synthesized output
        return AlignmentSample(
            id=sample.id,
            instruction=sample.instruction,
            input=sample.input,
            output=new_output,
            category=sample.category,
            metadata={
                **sample.metadata,
                "synthesized": True,
                "synthesis_style": self.synthesis_style,
                "original_output": sample.output,
            },
        )

    def __call__(self, data: DataContainer) -> DataContainer:
        """Apply LLM synthesis to qualifying samples."""
        results = []
        synthesized_count = 0

        for sample in data:
            if self.should_synthesize(sample):
                try:
                    new_sample = self.synthesize(sample)
                    results.append(new_sample)
                    synthesized_count += 1
                except Exception as e:
                    # Keep original on failure
                    sample.metadata["synthesis_error"] = str(e)
                    results.append(sample)
            else:
                results.append(sample)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(results),
            "synthesized_count": synthesized_count,
            "synthesis_rate": synthesized_count / len(data) if len(data) > 0 else 0,
            "synthesis_style": self.synthesis_style,
            "target_samples": self.target_samples,
        }

        return DataContainer(
            samples=results,
            provenance=data.provenance + [
                f"llm_synthesis ({self.synthesis_style}): synthesized {synthesized_count}/{len(data)}"
            ],
        )
