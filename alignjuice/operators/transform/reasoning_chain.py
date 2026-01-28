"""
Reasoning chain enhancer operator for AlignJuice.

Adds step-by-step reasoning chains to reasoning-type samples.
"""

from __future__ import annotations

from typing import Any, Literal

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("reasoning_chain")
class ReasoningChainEnhancer(Operator):
    """
    Enhance samples with explicit reasoning chains.

    Implements LIMO-style reasoning chain generation for reasoning tasks.
    """

    name = "reasoning_chain"

    CHAIN_PROMPTS = {
        "step_by_step": """Analyze the following question and provide a step-by-step reasoning process,
then give the final answer.

Question: {instruction}
{input_text}

Think through this step by step:
1. First, let me understand what is being asked...
2. Then, I'll identify the key information...
3. Next, I'll work through the logic...
4. Finally, I'll arrive at the answer...

Provide your step-by-step reasoning followed by the final answer:""",

        "cot": """Let's solve this problem step by step.

Problem: {instruction}
{input_text}

Let me think about this carefully:""",

        "tree_of_thought": """Consider multiple approaches to solve this problem.

Problem: {instruction}
{input_text}

Approach 1: ...
Approach 2: ...
Best approach and solution:""",
    }

    def __init__(
        self,
        llm_backend: str = "ollama",
        llm_model: str = "phi3:medium",
        chain_style: Literal["step_by_step", "cot", "tree_of_thought"] = "step_by_step",
        target_category: str = "reasoning",
        max_steps: int = 5,
        verify_logic: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """
        Initialize reasoning chain enhancer.

        Args:
            llm_backend: LLM backend to use
            llm_model: LLM model name
            chain_style: Style of reasoning chain
            target_category: Category of samples to enhance
            max_steps: Maximum reasoning steps
            verify_logic: Whether to verify logical consistency
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            llm_backend=llm_backend,
            llm_model=llm_model,
            chain_style=chain_style,
            target_category=target_category,
            max_steps=max_steps,
            verify_logic=verify_logic,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.chain_style = chain_style
        self.target_category = target_category
        self.max_steps = max_steps
        self.verify_logic = verify_logic
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

    def should_enhance(self, sample: AlignmentSample) -> bool:
        """Determine if a sample should be enhanced with reasoning chain."""
        # Check category
        if self.target_category != "all" and sample.category != self.target_category:
            return False

        # Check if already has reasoning chain
        if sample.metadata.get("has_reasoning_chain"):
            return False

        # Check if output already contains step-by-step reasoning
        output_lower = sample.output.lower()
        reasoning_indicators = [
            "step 1", "first,", "let me think",
            "let's break", "to solve this",
        ]
        has_reasoning = any(ind in output_lower for ind in reasoning_indicators)

        return not has_reasoning

    def generate_reasoning_chain(self, sample: AlignmentSample) -> str:
        """Generate reasoning chain for a sample."""
        prompt_template = self.CHAIN_PROMPTS.get(
            self.chain_style,
            self.CHAIN_PROMPTS["step_by_step"],
        )

        input_text = f"\nAdditional context: {sample.input}" if sample.input else ""

        prompt = prompt_template.format(
            instruction=sample.instruction,
            input_text=input_text,
        )

        response = self.llm.generate(prompt)
        return response.text.strip()

    def verify_reasoning(self, original_output: str, enhanced_output: str) -> bool:
        """Verify that enhanced output is logically consistent."""
        if not self.verify_logic:
            return True

        # Simple verification: check that key conclusions are preserved
        # Extract potential answer patterns from original
        original_lower = original_output.lower()
        enhanced_lower = enhanced_output.lower()

        # Check for numerical answers
        import re
        original_numbers = set(re.findall(r'\b\d+\.?\d*\b', original_lower))
        enhanced_numbers = set(re.findall(r'\b\d+\.?\d*\b', enhanced_lower))

        # If original had specific numbers, enhanced should have them too
        if original_numbers:
            overlap = len(original_numbers & enhanced_numbers) / len(original_numbers)
            if overlap < 0.5:
                return False

        return True

    def enhance(self, sample: AlignmentSample) -> AlignmentSample:
        """Enhance a sample with reasoning chain."""
        enhanced_output = self.generate_reasoning_chain(sample)

        # Verify logical consistency
        if not self.verify_reasoning(sample.output, enhanced_output):
            # Keep original if verification fails
            sample.metadata["reasoning_verification_failed"] = True
            return sample

        return AlignmentSample(
            id=sample.id,
            instruction=sample.instruction,
            input=sample.input,
            output=enhanced_output,
            category=sample.category,
            metadata={
                **sample.metadata,
                "has_reasoning_chain": True,
                "chain_style": self.chain_style,
                "original_output": sample.output,
            },
        )

    def __call__(self, data: DataContainer) -> DataContainer:
        """Apply reasoning chain enhancement to qualifying samples."""
        results = []
        enhanced_count = 0

        for sample in data:
            if self.should_enhance(sample):
                try:
                    new_sample = self.enhance(sample)
                    if new_sample.metadata.get("has_reasoning_chain"):
                        enhanced_count += 1
                    results.append(new_sample)
                except Exception as e:
                    sample.metadata["reasoning_error"] = str(e)
                    results.append(sample)
            else:
                results.append(sample)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(results),
            "enhanced_count": enhanced_count,
            "enhancement_rate": enhanced_count / len(data) if len(data) > 0 else 0,
            "chain_style": self.chain_style,
            "target_category": self.target_category,
        }

        return DataContainer(
            samples=results,
            provenance=data.provenance + [
                f"reasoning_chain ({self.chain_style}): enhanced {enhanced_count}/{len(data)}"
            ],
        )
