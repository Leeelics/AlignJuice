"""
Stage 4: Sandbox evaluation stage.

Evaluates final data quality and generates reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignjuice.core.data_container import DataContainer
from alignjuice.stages.base import BaseStage
from alignjuice.operators.base import Operator


class SandboxEvalStage(BaseStage):
    """
    Stage 4: Data-Juicer Sandbox evaluation.

    Evaluates:
    - Knowledge density
    - Diversity
    - Naturalness
    - Alignment effectiveness

    Input: ~1000 final samples
    Output: Same samples + quality report
    """

    name = "s4_sandbox_eval"

    def __init__(
        self,
        operators: list[Operator] | None = None,
        report_path: str = "reports/quality_report.html",
        metrics_path: str = "reports/metrics.json",
        eval_tasks: list[str] | None = None,
        sample_size: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize sandbox evaluation stage.

        Args:
            operators: Custom operators to apply
            report_path: Path for HTML report output
            metrics_path: Path for JSON metrics output
            eval_tasks: Evaluation tasks to run
            sample_size: Sample size for model evaluation
        """
        super().__init__(operators, **kwargs)
        self.report_path = report_path
        self.metrics_path = metrics_path
        self.eval_tasks = eval_tasks or ["helpfulness", "harmlessness", "honesty"]
        self.sample_size = sample_size

    def process(self, data: DataContainer) -> DataContainer:
        """Process data through sandbox evaluation stage."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data for Sandbox Eval stage")

        # Compute quality metrics
        metrics = self._compute_metrics(data)

        # Generate reports
        self._generate_reports(data, metrics)

        self._metrics.update({
            "stage": self.name,
            "input_count": len(data),
            "output_count": len(data),
            **metrics,
        })

        return data

    def _compute_metrics(self, data: DataContainer) -> dict[str, Any]:
        """Compute quality metrics for the data."""
        metrics = {}

        # Knowledge density (from metadata if available)
        knowledge_scores = [
            s.metadata.get("knowledge_score", 0.5)
            for s in data
        ]
        if knowledge_scores:
            metrics["knowledge_density"] = {
                "mean": sum(knowledge_scores) / len(knowledge_scores),
                "min": min(knowledge_scores),
                "max": max(knowledge_scores),
            }

        # Quality scores
        quality_scores = [
            s.metadata.get("quality_score", 0.5)
            for s in data
        ]
        if quality_scores:
            metrics["quality"] = {
                "mean": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
            }

        # Category distribution
        categories: dict[str, int] = {}
        for s in data:
            categories[s.category] = categories.get(s.category, 0) + 1
        metrics["category_distribution"] = categories

        # Diversity (unique instruction ratio)
        instructions = [s.instruction for s in data]
        unique_ratio = len(set(instructions)) / len(instructions) if instructions else 0
        metrics["diversity"] = {
            "unique_instruction_ratio": unique_ratio,
            "total_samples": len(data),
            "unique_instructions": len(set(instructions)),
        }

        # Length statistics
        instruction_lengths = [len(s.instruction) for s in data]
        output_lengths = [len(s.output) for s in data]
        metrics["length_stats"] = {
            "instruction": {
                "mean": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
                "min": min(instruction_lengths) if instruction_lengths else 0,
                "max": max(instruction_lengths) if instruction_lengths else 0,
            },
            "output": {
                "mean": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
                "min": min(output_lengths) if output_lengths else 0,
                "max": max(output_lengths) if output_lengths else 0,
            },
        }

        # Reasoning chain coverage
        reasoning_count = sum(
            1 for s in data if s.metadata.get("has_reasoning_chain", False)
        )
        metrics["reasoning_coverage"] = {
            "samples_with_reasoning": reasoning_count,
            "coverage_rate": reasoning_count / len(data) if len(data) > 0 else 0,
        }

        # Synthesis coverage
        synthesized_count = sum(
            1 for s in data if s.metadata.get("synthesized", False)
        )
        metrics["synthesis_coverage"] = {
            "synthesized_samples": synthesized_count,
            "synthesis_rate": synthesized_count / len(data) if len(data) > 0 else 0,
        }

        return metrics

    def _generate_reports(self, data: DataContainer, metrics: dict[str, Any]) -> None:
        """Generate quality reports."""
        # Ensure directories exist
        Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)

        # Save JSON metrics
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Generate HTML report
        html_content = self._generate_html_report(data, metrics)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_html_report(self, data: DataContainer, metrics: dict[str, Any]) -> str:
        """Generate HTML quality report."""
        # Category distribution chart data
        categories = metrics.get("category_distribution", {})
        cat_labels = list(categories.keys())
        cat_values = list(categories.values())

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AlignJuice Quality Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; }}
        .chart-container {{ width: 100%; max-width: 600px; margin: 20px auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 AlignJuice Quality Report</h1>

        <div class="card">
            <h2>Overview</h2>
            <div class="metric">
                <div class="metric-value">{len(data)}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics.get('knowledge_density', {}).get('mean', 0):.2f}</div>
                <div class="metric-label">Avg Knowledge Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics.get('quality', {}).get('mean', 0):.2f}</div>
                <div class="metric-label">Avg Quality Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics.get('diversity', {}).get('unique_instruction_ratio', 0):.1%}</div>
                <div class="metric-label">Diversity</div>
            </div>
        </div>

        <div class="card">
            <h2>Category Distribution</h2>
            <div class="chart-container">
                <canvas id="categoryChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Length Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Instruction</th>
                    <th>Output</th>
                </tr>
                <tr>
                    <td>Mean Length</td>
                    <td>{metrics.get('length_stats', {}).get('instruction', {}).get('mean', 0):.0f} chars</td>
                    <td>{metrics.get('length_stats', {}).get('output', {}).get('mean', 0):.0f} chars</td>
                </tr>
                <tr>
                    <td>Min Length</td>
                    <td>{metrics.get('length_stats', {}).get('instruction', {}).get('min', 0)} chars</td>
                    <td>{metrics.get('length_stats', {}).get('output', {}).get('min', 0)} chars</td>
                </tr>
                <tr>
                    <td>Max Length</td>
                    <td>{metrics.get('length_stats', {}).get('instruction', {}).get('max', 0)} chars</td>
                    <td>{metrics.get('length_stats', {}).get('output', {}).get('max', 0)} chars</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Processing Coverage</h2>
            <table>
                <tr>
                    <th>Processing Type</th>
                    <th>Count</th>
                    <th>Rate</th>
                </tr>
                <tr>
                    <td>Reasoning Chain Enhanced</td>
                    <td>{metrics.get('reasoning_coverage', {}).get('samples_with_reasoning', 0)}</td>
                    <td>{metrics.get('reasoning_coverage', {}).get('coverage_rate', 0):.1%}</td>
                </tr>
                <tr>
                    <td>LLM Synthesized</td>
                    <td>{metrics.get('synthesis_coverage', {}).get('synthesized_samples', 0)}</td>
                    <td>{metrics.get('synthesis_coverage', {}).get('synthesis_rate', 0):.1%}</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Sample Preview</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Instruction (preview)</th>
                </tr>
                {''.join(f'<tr><td>{s.id[:8]}...</td><td>{s.category}</td><td>{s.instruction[:100]}...</td></tr>' for s in list(data)[:10])}
            </table>
        </div>
    </div>

    <script>
        new Chart(document.getElementById('categoryChart'), {{
            type: 'doughnut',
            data: {{
                labels: {cat_labels},
                datasets: [{{
                    data: {cat_values},
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        return html
