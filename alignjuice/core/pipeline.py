"""
Pipeline module for AlignJuice.

Provides pipeline orchestration with checkpointing and stage management.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from alignjuice.core.data_container import DataContainer
from alignjuice.core.registry import Registry

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Operator(ABC):
    """Base class for all operators."""

    name: str = "base_operator"

    @abstractmethod
    def __call__(self, data: DataContainer) -> DataContainer:
        """Process data and return transformed DataContainer."""
        pass

    @property
    def metrics(self) -> dict[str, Any]:
        """Return metrics collected during processing."""
        return {}

    def explain(self) -> str:
        """Explain what this operator does."""
        return f"{self.__class__.__name__}: {self.__doc__ or 'No description'}"


class Stage(ABC):
    """Base class for pipeline stages."""

    name: str = "base_stage"

    def __init__(self, operators: list[Operator] | None = None):
        self.operators = operators or []
        self._metrics: dict[str, Any] = {}

    @abstractmethod
    def process(self, data: DataContainer) -> DataContainer:
        """Process data through this stage."""
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


class Checkpointer:
    """Handles pipeline checkpointing for recovery."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage_name: str, data: DataContainer, metrics: dict[str, Any] | None = None) -> Path:
        """Save checkpoint after a stage completes."""
        checkpoint_path = self.checkpoint_dir / f"{stage_name}.jsonl"
        data.to_jsonl(checkpoint_path)

        # Save metadata
        meta_path = self.checkpoint_dir / f"{stage_name}.meta.json"
        meta = {
            "stage": stage_name,
            "sample_count": len(data),
            "timestamp": time.time(),
            "provenance": data.provenance,
            "metrics": metrics or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return checkpoint_path

    def load(self, stage_name: str) -> tuple[DataContainer, dict[str, Any]]:
        """Load checkpoint for a stage."""
        checkpoint_path = self.checkpoint_dir / f"{stage_name}.jsonl"
        meta_path = self.checkpoint_dir / f"{stage_name}.meta.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = DataContainer.from_jsonl(checkpoint_path)

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return data, meta

    def exists(self, stage_name: str) -> bool:
        """Check if checkpoint exists for a stage."""
        return (self.checkpoint_dir / f"{stage_name}.jsonl").exists()

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.jsonl"):
            checkpoints.append(path.stem)
        return sorted(checkpoints)

    def clear(self) -> None:
        """Clear all checkpoints."""
        for path in self.checkpoint_dir.glob("*"):
            path.unlink()


class PipelineResult:
    """Result of a pipeline run."""

    def __init__(
        self,
        data: DataContainer,
        metrics: dict[str, Any],
        stage_metrics: dict[str, dict[str, Any]],
        elapsed_time: float,
    ):
        self.data = data
        self.metrics = metrics
        self.stage_metrics = stage_metrics
        self.elapsed_time = elapsed_time

    def report(self) -> None:
        """Display pipeline result report."""
        if RICH_AVAILABLE:
            self._rich_report()
        else:
            self._text_report()

    def _rich_report(self) -> None:
        """Display report using rich."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Summary panel
        summary = f"""
[bold]Pipeline Execution Summary[/bold]

Total Samples: {len(self.data)}
Elapsed Time: {self.elapsed_time:.2f}s
Stages Executed: {len(self.stage_metrics)}
        """
        console.print(Panel(summary, title="Results", border_style="green"))

        # Stage metrics table
        if self.stage_metrics:
            table = Table(title="Stage Metrics", show_header=True)
            table.add_column("Stage", style="cyan")
            table.add_column("Input", style="yellow")
            table.add_column("Output", style="green")
            table.add_column("Time (s)", style="magenta")

            for stage_name, metrics in self.stage_metrics.items():
                table.add_row(
                    stage_name,
                    str(metrics.get("input_count", "N/A")),
                    str(metrics.get("output_count", "N/A")),
                    f"{metrics.get('elapsed_time', 0):.2f}",
                )

            console.print(table)

    def _text_report(self) -> None:
        """Display plain text report."""
        print("\n=== Pipeline Execution Summary ===")
        print(f"Total Samples: {len(self.data)}")
        print(f"Elapsed Time: {self.elapsed_time:.2f}s")
        print(f"Stages Executed: {len(self.stage_metrics)}")

        if self.stage_metrics:
            print("\n--- Stage Metrics ---")
            for stage_name, metrics in self.stage_metrics.items():
                print(f"  {stage_name}:")
                print(f"    Input: {metrics.get('input_count', 'N/A')}")
                print(f"    Output: {metrics.get('output_count', 'N/A')}")
                print(f"    Time: {metrics.get('elapsed_time', 0):.2f}s")

    def save(self, path: str | Path) -> None:
        """Save result data to file."""
        self.data.to_jsonl(path)


class Pipeline:
    """
    Pipeline orchestrator for running multi-stage data processing.

    Supports:
    - Sequential stage execution
    - Checkpointing and recovery
    - Progress tracking
    - Metrics collection
    """

    def __init__(self, config: Any = None):
        """
        Initialize pipeline.

        Args:
            config: PipelineConfig object or None for manual stage setup
        """
        self.config = config
        self.stages: list[Stage] = []
        self.checkpointer: Checkpointer | None = None
        self._metrics: dict[str, Any] = {}
        self._stage_metrics: dict[str, dict[str, Any]] = {}

        if config:
            self._build_from_config(config)

    def _build_from_config(self, config: Any) -> None:
        """Build pipeline from configuration."""
        from alignjuice.config.schema import PipelineConfig

        if not isinstance(config, PipelineConfig):
            raise TypeError(f"Expected PipelineConfig, got {type(config)}")

        # Setup checkpointer
        if config.checkpoint_enabled:
            self.checkpointer = Checkpointer(config.checkpoint_dir)

        # Build stages from config
        for stage_config in config.stages:
            stage = self._create_stage(stage_config)
            self.stages.append(stage)

    def _create_stage(self, stage_config: Any) -> Stage:
        """Create a stage from configuration."""
        from alignjuice.stages import get_stage_class

        stage_cls = get_stage_class(stage_config.name)
        operators = []

        for op_config in stage_config.operators:
            if op_config.enabled:
                op_cls = Registry.get_operator(op_config.name)
                operators.append(op_cls(**op_config.params))

        return stage_cls(operators=operators)

    def add_stage(self, stage: Stage) -> Pipeline:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def run(self, data: DataContainer, resume_from: str | None = None) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            data: Input DataContainer
            resume_from: Stage name to resume from (uses checkpoint)

        Returns:
            PipelineResult with processed data and metrics
        """
        start_time = time.time()
        current_data = data
        start_idx = 0

        # Resume from checkpoint if specified
        if resume_from and self.checkpointer:
            if self.checkpointer.exists(resume_from):
                current_data, _ = self.checkpointer.load(resume_from)
                # Find the index of the next stage
                for i, stage in enumerate(self.stages):
                    if stage.name == resume_from:
                        start_idx = i + 1
                        break

        # Run stages with progress tracking
        if RICH_AVAILABLE:
            current_data = self._run_with_progress(current_data, start_idx)
        else:
            current_data = self._run_simple(current_data, start_idx)

        elapsed_time = time.time() - start_time

        return PipelineResult(
            data=current_data,
            metrics=self._metrics,
            stage_metrics=self._stage_metrics,
            elapsed_time=elapsed_time,
        )

    def _run_simple(self, data: DataContainer, start_idx: int = 0) -> DataContainer:
        """Run pipeline without progress display."""
        current_data = data

        for i, stage in enumerate(self.stages[start_idx:], start=start_idx):
            print(f"Running stage: {stage.name}")
            stage_start = time.time()
            input_count = len(current_data)

            # Validate input
            if not stage.validate_input(current_data):
                raise ValueError(f"Stage {stage.name} input validation failed")

            # Process
            current_data = stage.process(current_data)

            # Validate output
            if not stage.validate_output(current_data):
                raise ValueError(f"Stage {stage.name} output validation failed")

            # Record metrics
            stage_elapsed = time.time() - stage_start
            self._stage_metrics[stage.name] = {
                "input_count": input_count,
                "output_count": len(current_data),
                "elapsed_time": stage_elapsed,
                **stage.metrics,
            }

            # Checkpoint
            if self.checkpointer:
                self.checkpointer.save(stage.name, current_data, self._stage_metrics[stage.name])

            print(f"  Completed: {input_count} -> {len(current_data)} samples ({stage_elapsed:.2f}s)")

        return current_data

    def _run_with_progress(self, data: DataContainer, start_idx: int = 0) -> DataContainer:
        """Run pipeline with rich progress display."""
        current_data = data
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                "[cyan]Pipeline Progress",
                total=len(self.stages) - start_idx,
            )

            for i, stage in enumerate(self.stages[start_idx:], start=start_idx):
                stage_task = progress.add_task(
                    f"[yellow]{stage.name}",
                    total=100,
                )

                stage_start = time.time()
                input_count = len(current_data)

                # Validate input
                if not stage.validate_input(current_data):
                    raise ValueError(f"Stage {stage.name} input validation failed")

                progress.update(stage_task, completed=30)

                # Process
                current_data = stage.process(current_data)

                progress.update(stage_task, completed=80)

                # Validate output
                if not stage.validate_output(current_data):
                    raise ValueError(f"Stage {stage.name} output validation failed")

                # Record metrics
                stage_elapsed = time.time() - stage_start
                self._stage_metrics[stage.name] = {
                    "input_count": input_count,
                    "output_count": len(current_data),
                    "elapsed_time": stage_elapsed,
                    **stage.metrics,
                }

                # Checkpoint
                if self.checkpointer:
                    self.checkpointer.save(stage.name, current_data, self._stage_metrics[stage.name])

                progress.update(stage_task, completed=100)
                progress.update(overall_task, advance=1)

        return current_data

    def run_stage(self, stage_name: str, data: DataContainer) -> DataContainer:
        """Run a single stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.process(data)
        raise ValueError(f"Stage '{stage_name}' not found")

    def status(self) -> dict[str, Any]:
        """Get pipeline status including checkpoint info."""
        status = {
            "stages": [s.name for s in self.stages],
            "checkpoints": [],
        }

        if self.checkpointer:
            status["checkpoints"] = self.checkpointer.list_checkpoints()

        return status
