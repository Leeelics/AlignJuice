"""
Configuration schema module for AlignJuice.

Provides Pydantic models for pipeline configuration.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class OperatorConfig(BaseModel):
    """Configuration for a single operator."""

    name: str = Field(..., description="Operator name (must be registered)")
    enabled: bool = Field(default=True, description="Whether this operator is enabled")
    params: dict[str, Any] = Field(default_factory=dict, description="Operator parameters")

    model_config = {"extra": "allow"}


class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""

    name: str = Field(..., description="Stage name")
    operators: list[OperatorConfig] = Field(default_factory=list, description="List of operators")
    input_validation: bool = Field(default=True, description="Validate input data")
    output_validation: bool = Field(default=True, description="Validate output data")

    model_config = {"extra": "allow"}


class LLMConfig(BaseModel):
    """Configuration for LLM backend."""

    backend: Literal["ollama", "vllm", "openai", "anthropic"] = Field(
        default="ollama",
        description="LLM backend to use",
    )
    model: str = Field(default="phi3:medium", description="Model name/identifier")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")
    fallback_backend: str | None = Field(
        default=None,
        description="Fallback backend if primary fails",
    )
    api_key: str | None = Field(default=None, description="API key (for cloud backends)")
    base_url: str | None = Field(default=None, description="Base URL for API")

    model_config = {"extra": "allow"}


class EmbeddingConfig(BaseModel):
    """Configuration for embedding backend."""

    backend: Literal["sentence_transformers", "openai"] = Field(
        default="sentence_transformers",
        description="Embedding backend to use",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name/identifier",
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for embedding")
    api_key: str | None = Field(default=None, description="API key (for cloud backends)")

    model_config = {"extra": "allow"}


class OutputConfig(BaseModel):
    """Configuration for stage output."""

    format: Literal["jsonl", "parquet", "csv"] = Field(
        default="jsonl",
        description="Output format",
    )
    path: str = Field(..., description="Output file path")
    target_count: int | None = Field(default=None, description="Target sample count")

    model_config = {"extra": "allow"}


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    name: str = Field(default="alignjuice_pipeline", description="Pipeline name")
    version: str = Field(default="1.0.0", description="Pipeline version")
    stages: list[StageConfig] = Field(default_factory=list, description="Pipeline stages")

    # Global settings
    checkpoint_enabled: bool = Field(default=True, description="Enable checkpointing")
    checkpoint_dir: str = Field(default=".checkpoints", description="Checkpoint directory")
    parallel_workers: int = Field(default=4, gt=0, description="Number of parallel workers")

    # Data settings
    input_format: Literal["jsonl", "parquet", "csv"] = Field(
        default="jsonl",
        description="Input data format",
    )
    output_format: Literal["jsonl", "parquet"] = Field(
        default="jsonl",
        description="Output data format",
    )

    # Default thresholds
    dedup_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Semantic deduplication threshold",
    )
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Quality score threshold",
    )
    target_samples: int = Field(default=1000, gt=0, description="Target number of samples")

    # LLM and embedding configs
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration",
    )

    model_config = {"extra": "allow"}

    @field_validator("stages", mode="before")
    @classmethod
    def validate_stages(cls, v: Any) -> list[StageConfig]:
        """Ensure stages is a list of StageConfig."""
        if isinstance(v, list):
            return [
                StageConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v


# Stage-specific configurations

class DataJuicerStageConfig(StageConfig):
    """Configuration for Data-Juicer stage."""

    name: str = "s1_data_juicer"
    recipes: list[str] = Field(
        default=["daily", "factual", "reasoning", "creative"],
        description="Data-Juicer recipes to load",
    )
    output: OutputConfig | None = None


class KnowledgeFilterStageConfig(StageConfig):
    """Configuration for knowledge filtering stage."""

    name: str = "s2_knowledge_filter"
    reference_corpus: str = Field(
        default="wikipedia_chunks",
        description="Reference corpus for knowledge scoring",
    )
    top_k: int = Field(default=1000, gt=0, description="Number of top samples to keep")
    output: OutputConfig | None = None


class ReasoningEnhanceStageConfig(StageConfig):
    """Configuration for reasoning enhancement stage."""

    name: str = "s3_reasoning_enhance"
    chain_style: Literal["step_by_step", "cot", "tree_of_thought"] = Field(
        default="step_by_step",
        description="Reasoning chain style",
    )
    max_steps: int = Field(default=5, gt=0, description="Maximum reasoning steps")
    output: OutputConfig | None = None


class SandboxEvalStageConfig(StageConfig):
    """Configuration for sandbox evaluation stage."""

    name: str = "s4_sandbox_eval"
    eval_tasks: list[str] = Field(
        default=["helpfulness", "harmlessness", "honesty"],
        description="Evaluation tasks",
    )
    sample_size: int = Field(default=100, gt=0, description="Sample size for evaluation")
    report_path: str = Field(
        default="reports/quality_report.html",
        description="Report output path",
    )
    output: OutputConfig | None = None
