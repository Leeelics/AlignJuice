"""
Data container module for AlignJuice.

Provides unified data abstraction with Jupyter-friendly display capabilities.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


CategoryType = Literal["daily", "factual", "reasoning", "creative", "other"]


@dataclass
class AlignmentSample:
    """A single alignment data sample."""

    id: str
    instruction: str
    input: str
    output: str
    category: CategoryType = "other"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentSample:
        """Create sample from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            instruction=data.get("instruction", ""),
            input=data.get("input", ""),
            output=data.get("output", ""),
            category=data.get("category", "other"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        instr_preview = self.instruction[:50] + "..." if len(self.instruction) > 50 else self.instruction
        return f"AlignmentSample(id={self.id!r}, instruction={instr_preview!r}, category={self.category!r})"


class DataContainer:
    """
    Unified data container for alignment samples.

    Supports:
    - Jupyter-friendly display (rich tables, visualizations)
    - Lazy loading and streaming
    - Metrics tracking
    - Processing history (provenance)
    """

    def __init__(
        self,
        samples: list[AlignmentSample] | None = None,
        provenance: list[str] | None = None,
    ):
        """
        Initialize DataContainer.

        Args:
            samples: List of AlignmentSample objects
            provenance: List of processing steps applied to this data
        """
        self._samples: list[AlignmentSample] = samples or []
        self._provenance: list[str] = provenance or []
        self._metrics_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[AlignmentSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int | slice) -> AlignmentSample | DataContainer:
        if isinstance(idx, slice):
            return DataContainer(
                samples=self._samples[idx],
                provenance=self._provenance.copy(),
            )
        return self._samples[idx]

    @property
    def samples(self) -> list[AlignmentSample]:
        """Access underlying samples."""
        return self._samples

    @property
    def provenance(self) -> list[str]:
        """Get processing history."""
        return self._provenance.copy()

    def add_provenance(self, step: str) -> DataContainer:
        """Add a processing step to provenance and return self."""
        self._provenance.append(step)
        return self

    # ==================== Data Loading ====================

    @classmethod
    def from_jsonl(cls, path: str | Path) -> DataContainer:
        """Load data from JSONL file."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(AlignmentSample.from_dict(data))
        return cls(samples=samples, provenance=[f"loaded from {path}"])

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> DataContainer:
        """Create from list of dictionaries."""
        samples = [AlignmentSample.from_dict(d) for d in data]
        return cls(samples=samples, provenance=["created from list"])

    def to_jsonl(self, path: str | Path) -> None:
        """Save data to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    def to_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [s.to_dict() for s in self._samples]

    # ==================== Data Operations ====================

    def filter(self, predicate: Callable[[AlignmentSample], bool]) -> DataContainer:
        """Filter samples by predicate function."""
        filtered = [s for s in self._samples if predicate(s)]
        return DataContainer(
            samples=filtered,
            provenance=self._provenance + [f"filtered: {len(self)} -> {len(filtered)}"],
        )

    def map(self, func: Callable[[AlignmentSample], AlignmentSample]) -> DataContainer:
        """Apply function to each sample."""
        mapped = [func(s) for s in self._samples]
        return DataContainer(
            samples=mapped,
            provenance=self._provenance + ["mapped"],
        )

    def sample(self, n: int, seed: int | None = None) -> DataContainer:
        """Random sample n items."""
        if seed is not None:
            random.seed(seed)
        n = min(n, len(self._samples))
        sampled = random.sample(self._samples, n)
        return DataContainer(
            samples=sampled,
            provenance=self._provenance + [f"sampled {n} items"],
        )

    def head(self, n: int = 10) -> DataContainer:
        """Get first n samples."""
        return DataContainer(
            samples=self._samples[:n],
            provenance=self._provenance + [f"head {n}"],
        )

    def tail(self, n: int = 10) -> DataContainer:
        """Get last n samples."""
        return DataContainer(
            samples=self._samples[-n:],
            provenance=self._provenance + [f"tail {n}"],
        )

    def concat(self, other: DataContainer) -> DataContainer:
        """Concatenate with another DataContainer."""
        return DataContainer(
            samples=self._samples + other._samples,
            provenance=self._provenance + [f"concatenated with {len(other)} samples"],
        )

    def diff(self, other: DataContainer) -> DataContainer:
        """Find samples in self but not in other (by id)."""
        other_ids = {s.id for s in other._samples}
        diff_samples = [s for s in self._samples if s.id not in other_ids]
        return DataContainer(
            samples=diff_samples,
            provenance=[f"diff: {len(diff_samples)} samples unique to first container"],
        )

    # ==================== Statistics ====================

    def describe(self) -> dict[str, Any]:
        """Get statistical summary of the data."""
        if not self._samples:
            return {"count": 0}

        categories = {}
        instruction_lengths = []
        output_lengths = []

        for s in self._samples:
            categories[s.category] = categories.get(s.category, 0) + 1
            instruction_lengths.append(len(s.instruction))
            output_lengths.append(len(s.output))

        stats = {
            "count": len(self._samples),
            "categories": categories,
            "instruction_length": {
                "min": min(instruction_lengths),
                "max": max(instruction_lengths),
                "mean": sum(instruction_lengths) / len(instruction_lengths),
            },
            "output_length": {
                "min": min(output_lengths),
                "max": max(output_lengths),
                "mean": sum(output_lengths) / len(output_lengths),
            },
            "provenance_steps": len(self._provenance),
        }

        # Display if in interactive environment
        self._display_stats(stats)
        return stats

    def _display_stats(self, stats: dict[str, Any]) -> None:
        """Display statistics using rich if available."""
        if not RICH_AVAILABLE:
            return

        console = Console()

        # Main stats table
        table = Table(title="Data Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Samples", str(stats["count"]))
        table.add_row("Processing Steps", str(stats["provenance_steps"]))

        if stats["count"] > 0:
            table.add_row(
                "Instruction Length",
                f"min={stats['instruction_length']['min']}, "
                f"max={stats['instruction_length']['max']}, "
                f"mean={stats['instruction_length']['mean']:.1f}"
            )
            table.add_row(
                "Output Length",
                f"min={stats['output_length']['min']}, "
                f"max={stats['output_length']['max']}, "
                f"mean={stats['output_length']['mean']:.1f}"
            )

        console.print(table)

        # Category distribution
        if stats.get("categories"):
            cat_table = Table(title="Category Distribution", show_header=True)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            cat_table.add_column("Percentage", style="yellow")

            total = stats["count"]
            for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
                pct = count / total * 100
                cat_table.add_row(cat, str(count), f"{pct:.1f}%")

            console.print(cat_table)

    # ==================== Display Methods ====================

    def show(self, max_width: int = 80) -> None:
        """Display samples in a formatted table."""
        if not RICH_AVAILABLE:
            for s in self._samples:
                print(s)
            return

        console = Console()
        table = Table(title=f"Alignment Samples ({len(self)} items)", show_header=True)
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Category", style="cyan", max_width=10)
        table.add_column("Instruction", style="green", max_width=max_width // 3)
        table.add_column("Output", style="yellow", max_width=max_width // 3)

        for sample in self._samples[:20]:  # Limit display
            instr = sample.instruction[:100] + "..." if len(sample.instruction) > 100 else sample.instruction
            output = sample.output[:100] + "..." if len(sample.output) > 100 else sample.output
            table.add_row(
                sample.id[:10] + "..." if len(sample.id) > 10 else sample.id,
                sample.category,
                instr,
                output,
            )

        if len(self._samples) > 20:
            table.add_row("...", "...", f"({len(self._samples) - 20} more)", "...")

        console.print(table)

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        rows = []
        for i, s in enumerate(self._samples[:20]):
            instr = s.instruction[:100] + "..." if len(s.instruction) > 100 else s.instruction
            output = s.output[:100] + "..." if len(s.output) > 100 else s.output
            rows.append(f"""
                <tr>
                    <td>{s.id[:10]}...</td>
                    <td><span style="color: #0066cc">{s.category}</span></td>
                    <td>{instr}</td>
                    <td>{output}</td>
                </tr>
            """)

        if len(self._samples) > 20:
            rows.append(f'<tr><td colspan="4">... ({len(self._samples) - 20} more samples)</td></tr>')

        return f"""
        <div style="max-height: 500px; overflow-y: auto;">
            <h4>DataContainer: {len(self._samples)} samples</h4>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th>ID</th>
                        <th>Category</th>
                        <th>Instruction</th>
                        <th>Output</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def __repr__(self) -> str:
        return f"DataContainer(samples={len(self._samples)}, provenance_steps={len(self._provenance)})"

    # ==================== Metrics ====================

    def compute_metrics(self, metrics: Sequence[Any]) -> dict[str, Any]:
        """
        Compute metrics on the data.

        Args:
            metrics: List of Metric objects to compute

        Returns:
            Dictionary of metric name -> values
        """
        results = {}
        for metric in metrics:
            name = getattr(metric, "name", metric.__class__.__name__)
            results[name] = metric.compute(self)
        self._metrics_cache.update(results)
        return results

    @property
    def metrics(self) -> dict[str, Any]:
        """Access cached metrics."""
        return self._metrics_cache.copy()

    # ==================== Visualization ====================

    def plot_distribution(self, field: str = "category") -> None:
        """Plot distribution of a field."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        if field == "category":
            categories = {}
            for s in self._samples:
                categories[s.category] = categories.get(s.category, 0) + 1

            plt.figure(figsize=(10, 6))
            plt.bar(categories.keys(), categories.values(), color="steelblue")
            plt.xlabel("Category")
            plt.ylabel("Count")
            plt.title("Category Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif field == "instruction_length":
            lengths = [len(s.instruction) for s in self._samples]
            plt.figure(figsize=(10, 6))
            plt.hist(lengths, bins=50, color="steelblue", edgecolor="white")
            plt.xlabel("Instruction Length (chars)")
            plt.ylabel("Count")
            plt.title("Instruction Length Distribution")
            plt.tight_layout()
            plt.show()

        elif field == "output_length":
            lengths = [len(s.output) for s in self._samples]
            plt.figure(figsize=(10, 6))
            plt.hist(lengths, bins=50, color="steelblue", edgecolor="white")
            plt.xlabel("Output Length (chars)")
            plt.ylabel("Count")
            plt.title("Output Length Distribution")
            plt.tight_layout()
            plt.show()

    # ==================== Conversion to Polars ====================

    def to_polars(self) -> Any:
        """Convert to Polars DataFrame."""
        if not POLARS_AVAILABLE:
            raise ImportError("polars not installed. Run: pip install polars")

        data = {
            "id": [s.id for s in self._samples],
            "instruction": [s.instruction for s in self._samples],
            "input": [s.input for s in self._samples],
            "output": [s.output for s in self._samples],
            "category": [s.category for s in self._samples],
        }
        return pl.DataFrame(data)

    @classmethod
    def from_polars(cls, df: Any) -> DataContainer:
        """Create from Polars DataFrame."""
        samples = []
        for row in df.iter_rows(named=True):
            samples.append(AlignmentSample(
                id=row.get("id", str(uuid.uuid4())),
                instruction=row.get("instruction", ""),
                input=row.get("input", ""),
                output=row.get("output", ""),
                category=row.get("category", "other"),
                metadata=row.get("metadata", {}),
            ))
        return cls(samples=samples, provenance=["created from Polars DataFrame"])
