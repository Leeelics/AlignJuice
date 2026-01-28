"""
Stage 1: Data-Juicer integration stage.

Handles initial data processing using Data-Juicer CLI wrapper.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from alignjuice.core.data_container import DataContainer
from alignjuice.stages.base import BaseStage
from alignjuice.operators.base import Operator


class DataJuicerStage(BaseStage):
    """
    Stage 1: Data-Juicer based processing.

    Applies:
    - DaaR diversity operators
    - Semantic deduplication
    - Quality score filtering

    Input: ~3000 rough-screened candidates
    Output: ~1500 deduplicated + high quality samples
    """

    name = "s1_data_juicer"

    def __init__(
        self,
        operators: list[Operator] | None = None,
        recipes: list[str] | None = None,
        use_cli: bool = False,
        data_juicer_path: str | None = None,
        target_count: int = 1500,
        **kwargs: Any,
    ):
        """
        Initialize Data-Juicer stage.

        Args:
            operators: Custom operators to apply
            recipes: Data-Juicer recipe names to load
            use_cli: Whether to use Data-Juicer CLI (vs built-in operators)
            data_juicer_path: Path to Data-Juicer installation
            target_count: Target number of samples after processing
        """
        super().__init__(operators, **kwargs)
        self.recipes = recipes or ["daily", "factual", "reasoning", "creative"]
        self.use_cli = use_cli
        self.data_juicer_path = data_juicer_path
        self.target_count = target_count

        # Initialize default operators if none provided
        if not self.operators:
            self._init_default_operators()

    def _init_default_operators(self) -> None:
        """Initialize default operators for this stage."""
        from alignjuice.operators.dedup import ExactDedup, SemanticDedup
        from alignjuice.operators.filter import QualityFilter, DiversityFilter

        self.operators = [
            ExactDedup(field="instruction"),
            SemanticDedup(threshold=0.95),
            QualityFilter(threshold=0.8),
            DiversityFilter(target_count=self.target_count),
        ]

    def process(self, data: DataContainer) -> DataContainer:
        """Process data through Data-Juicer stage."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data for Data-Juicer stage")

        input_count = len(data)

        if self.use_cli and self.data_juicer_path:
            # Use Data-Juicer CLI
            result = self._process_with_cli(data)
        else:
            # Use built-in operators
            result = self._apply_operators(data)

        self._metrics.update({
            "stage": self.name,
            "input_count": input_count,
            "output_count": len(result),
            "reduction_rate": 1 - len(result) / input_count if input_count > 0 else 0,
        })

        return result

    def _process_with_cli(self, data: DataContainer) -> DataContainer:
        """Process data using Data-Juicer CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input data
            input_path = tmpdir / "input.jsonl"
            data.to_jsonl(input_path)

            # Create config for Data-Juicer
            config = self._create_dj_config(tmpdir)
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                import yaml
                yaml.dump(config, f)

            # Run Data-Juicer
            output_path = tmpdir / "output.jsonl"
            cmd = [
                "python", "-m", "data_juicer.core.executor",
                "--config", str(config_path),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
                if result.returncode != 0:
                    print(f"Data-Juicer warning: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Data-Juicer timed out, using fallback")
                return self._apply_operators(data)
            except FileNotFoundError:
                print("Data-Juicer not found, using fallback")
                return self._apply_operators(data)

            # Read output
            if output_path.exists():
                return DataContainer.from_jsonl(output_path)
            else:
                # Fallback to built-in operators
                return self._apply_operators(data)

    def _create_dj_config(self, tmpdir: Path) -> dict[str, Any]:
        """Create Data-Juicer configuration."""
        return {
            "dataset_path": str(tmpdir / "input.jsonl"),
            "export_path": str(tmpdir / "output.jsonl"),
            "process": [
                {"deduplicator": {"method": "exact"}},
                {"deduplicator": {"method": "semantic", "threshold": 0.95}},
                {"filter": {"quality_score_threshold": 0.8}},
            ],
        }
