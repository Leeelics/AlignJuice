"""
AlignJuice - A modular framework for managing, creating, and evaluating high-quality alignment data.

Usage:
    from alignjuice import AlignJuice, DataContainer
    from alignjuice.operators import SemanticDedup, QualityFilter

    # Initialize framework
    aj = AlignJuice(config="configs/default.yaml")

    # Load and process data
    data = aj.load("data/raw.jsonl")
    result = aj.run_pipeline(data)

    # Interactive exploration
    data.describe()
    data.sample(5).show()
"""

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.pipeline import Pipeline
from alignjuice.core.registry import Registry, register_operator, register_metric, register_llm
from alignjuice.config.schema import PipelineConfig, StageConfig, OperatorConfig

__version__ = "0.1.0"
__all__ = [
    # Core
    "AlignJuice",
    "DataContainer",
    "AlignmentSample",
    "Pipeline",
    # Registry
    "Registry",
    "register_operator",
    "register_metric",
    "register_llm",
    # Config
    "PipelineConfig",
    "StageConfig",
    "OperatorConfig",
]


class AlignJuice:
    """Main entry point for the AlignJuice framework."""

    def __init__(self, config: str | PipelineConfig | None = None):
        """
        Initialize AlignJuice framework.

        Args:
            config: Path to YAML config file or PipelineConfig object
        """
        self.config = self._load_config(config)
        self.pipeline = Pipeline(self.config) if self.config else None
        self._registry = Registry()

    def _load_config(self, config: str | PipelineConfig | None) -> PipelineConfig | None:
        """Load configuration from file or use provided config."""
        if config is None:
            return None
        if isinstance(config, PipelineConfig):
            return config
        if isinstance(config, str):
            from alignjuice.config.loader import load_config
            return load_config(config)
        raise ValueError(f"Invalid config type: {type(config)}")

    def load(self, path: str) -> DataContainer:
        """
        Load data from file.

        Args:
            path: Path to data file (JSONL, Parquet, or CSV)

        Returns:
            DataContainer with loaded samples
        """
        from alignjuice.io.readers import read_data
        return read_data(path)

    def run_pipeline(self, data: DataContainer) -> DataContainer:
        """
        Run the full pipeline on data.

        Args:
            data: Input DataContainer

        Returns:
            Processed DataContainer with results
        """
        if self.pipeline is None:
            raise ValueError("No pipeline configured. Provide a config when initializing.")
        return self.pipeline.run(data)

    def run_stage(self, stage_name: str, data: DataContainer) -> DataContainer:
        """
        Run a single stage on data.

        Args:
            stage_name: Name of the stage to run
            data: Input DataContainer

        Returns:
            Processed DataContainer
        """
        if self.pipeline is None:
            raise ValueError("No pipeline configured. Provide a config when initializing.")
        return self.pipeline.run_stage(stage_name, data)

    @property
    def registry(self) -> Registry:
        """Access the global registry for operators, metrics, and LLMs."""
        return self._registry
