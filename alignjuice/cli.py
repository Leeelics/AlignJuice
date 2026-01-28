"""
CLI module for AlignJuice.

Provides command-line interface for pipeline execution.
"""

from __future__ import annotations

import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AlignJuice - High-quality alignment data management framework."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to pipeline config file")
@click.option("--input", "-i", "input_path", required=True, help="Input data file")
@click.option("--output", "-o", "output_path", help="Output data file")
@click.option("--resume", "-r", help="Resume from checkpoint stage")
def run(config: str, input_path: str, output_path: str | None, resume: str | None):
    """Run the full pipeline."""
    from alignjuice import AlignJuice
    from alignjuice.io import read_data, write_data

    click.echo(f"Loading config from {config}")
    aj = AlignJuice(config=config)

    click.echo(f"Loading data from {input_path}")
    data = read_data(input_path)
    click.echo(f"Loaded {len(data)} samples")

    click.echo("Running pipeline...")
    result = aj.pipeline.run(data, resume_from=resume)

    result.report()

    if output_path:
        write_data(result.data, output_path)
        click.echo(f"Saved {len(result.data)} samples to {output_path}")


@main.command("run-stage")
@click.argument("stage_name")
@click.option("--input", "-i", "input_path", required=True, help="Input data file")
@click.option("--output", "-o", "output_path", help="Output data file")
@click.option("--config", "-c", help="Stage config file")
def run_stage(stage_name: str, input_path: str, output_path: str | None, config: str | None):
    """Run a single pipeline stage."""
    from alignjuice.io import read_data, write_data
    from alignjuice.stages import get_stage_class

    click.echo(f"Loading data from {input_path}")
    data = read_data(input_path)
    click.echo(f"Loaded {len(data)} samples")

    click.echo(f"Running stage: {stage_name}")
    stage_cls = get_stage_class(stage_name)
    stage = stage_cls()

    result = stage.process(data)
    click.echo(f"Stage complete: {len(data)} -> {len(result)} samples")

    if output_path:
        write_data(result, output_path)
        click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "-i", "input_path", required=True, help="Input data file")
@click.option("--report", "-r", "report_dir", default="reports", help="Report output directory")
def evaluate(input_path: str, report_dir: str):
    """Evaluate data quality and generate reports."""
    from alignjuice.io import read_data
    from alignjuice.stages import SandboxEvalStage

    click.echo(f"Loading data from {input_path}")
    data = read_data(input_path)
    click.echo(f"Loaded {len(data)} samples")

    click.echo("Running evaluation...")
    stage = SandboxEvalStage(
        report_path=f"{report_dir}/quality_report.html",
        metrics_path=f"{report_dir}/metrics.json",
    )
    stage.process(data)

    click.echo(f"Reports saved to {report_dir}/")


@main.command("validate-config")
@click.argument("config_path")
def validate_config(config_path: str):
    """Validate a pipeline configuration file."""
    from alignjuice.config import load_config

    try:
        config = load_config(config_path)
        click.echo(f"✓ Config valid: {config.name} v{config.version}")
        click.echo(f"  Stages: {len(config.stages)}")
        for stage in config.stages:
            click.echo(f"    - {stage.name}: {len(stage.operators)} operators")
    except Exception as e:
        click.echo(f"✗ Config invalid: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--checkpoint-dir", "-d", default=".checkpoints", help="Checkpoint directory")
def status(checkpoint_dir: str):
    """Show pipeline status and checkpoints."""
    from alignjuice.core.pipeline import Checkpointer

    checkpointer = Checkpointer(checkpoint_dir)
    checkpoints = checkpointer.list_checkpoints()

    if not checkpoints:
        click.echo("No checkpoints found.")
        return

    click.echo(f"Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        click.echo(f"  - {cp}")


@main.command()
@click.option("--checkpoint-dir", "-d", default=".checkpoints", help="Checkpoint directory")
@click.confirmation_option(prompt="Are you sure you want to clear all checkpoints?")
def clear_checkpoints(checkpoint_dir: str):
    """Clear all pipeline checkpoints."""
    from alignjuice.core.pipeline import Checkpointer

    checkpointer = Checkpointer(checkpoint_dir)
    checkpointer.clear()
    click.echo("Checkpoints cleared.")


if __name__ == "__main__":
    main()
