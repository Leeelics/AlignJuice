"""Data writers for AlignJuice."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from alignjuice.core.data_container import DataContainer


class BaseWriter(ABC):
    """Base class for data writers."""

    @abstractmethod
    def write(self, data: DataContainer, path: str | Path) -> None:
        """Write data to file."""
        pass


class JSONLWriter(BaseWriter):
    """Writer for JSONL files."""

    def write(self, data: DataContainer, path: str | Path) -> None:
        """Write DataContainer to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


class ParquetWriter(BaseWriter):
    """Writer for Parquet files."""

    def write(self, data: DataContainer, path: str | Path) -> None:
        """Write DataContainer to Parquet file."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars required for Parquet support. Run: pip install polars")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = data.to_polars()
        df.write_parquet(path)


def write_data(data: DataContainer, path: str | Path) -> None:
    """
    Write data to file, auto-detecting format.

    Args:
        data: DataContainer to write
        path: Output path
    """
    path = Path(path)
    suffix = path.suffix.lower()

    writers = {
        ".jsonl": JSONLWriter(),
        ".json": JSONLWriter(),
        ".parquet": ParquetWriter(),
    }

    writer = writers.get(suffix)
    if writer is None:
        raise ValueError(f"Unsupported file format: {suffix}")

    writer.write(data, path)
