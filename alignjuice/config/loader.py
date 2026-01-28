"""
Configuration loader module for AlignJuice.

Handles loading and saving YAML/JSON configuration files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from alignjuice.config.schema import PipelineConfig


def load_config(path: str | Path) -> PipelineConfig:
    """
    Load pipeline configuration from file.

    Args:
        path: Path to YAML or JSON config file

    Returns:
        PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")

    # Expand environment variables in string values
    data = _expand_env_vars(data)

    return PipelineConfig(**data)


def save_config(config: PipelineConfig, path: str | Path) -> None:
    """
    Save pipeline configuration to file.

    Args:
        config: PipelineConfig object
        path: Output path (YAML or JSON based on extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(exclude_none=True)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Expand ${VAR} or $VAR patterns
        return os.path.expandvars(data)
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """
    Merge multiple config dictionaries.

    Later configs override earlier ones.
    """
    result: dict[str, Any] = {}

    for config in configs:
        _deep_merge(result, config)

    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
