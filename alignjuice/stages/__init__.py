"""Stages module for AlignJuice."""

from alignjuice.stages.base import BaseStage
from alignjuice.stages.s1_data_juicer import DataJuicerStage
from alignjuice.stages.s2_knowledge_filter import KnowledgeFilterStage
from alignjuice.stages.s3_reasoning_enhance import ReasoningEnhanceStage
from alignjuice.stages.s4_sandbox_eval import SandboxEvalStage

__all__ = [
    "BaseStage",
    "DataJuicerStage",
    "KnowledgeFilterStage",
    "ReasoningEnhanceStage",
    "SandboxEvalStage",
    "get_stage_class",
]

# Stage registry
_STAGE_REGISTRY = {
    "s1_data_juicer": DataJuicerStage,
    "data_juicer": DataJuicerStage,
    "s2_knowledge_filter": KnowledgeFilterStage,
    "knowledge_filter": KnowledgeFilterStage,
    "s3_reasoning_enhance": ReasoningEnhanceStage,
    "reasoning_enhance": ReasoningEnhanceStage,
    "s4_sandbox_eval": SandboxEvalStage,
    "sandbox_eval": SandboxEvalStage,
}


def get_stage_class(name: str) -> type:
    """Get stage class by name."""
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Stage '{name}' not found. Available: {list(_STAGE_REGISTRY.keys())}")
    return _STAGE_REGISTRY[name]
