"""Data readers for AlignJuice."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from alignjuice.core.data_container import DataContainer, AlignmentSample


class BaseReader(ABC):
    """Base class for data readers."""

    @abstractmethod
    def read(self, path: str | Path) -> DataContainer:
        """Read data from file."""
        pass


class JSONLReader(BaseReader):
    """Reader for JSONL files."""

    def read(self, path: str | Path) -> DataContainer:
        """Read JSONL file into DataContainer."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(AlignmentSample.from_dict(data))
        return DataContainer(samples=samples, provenance=[f"loaded from {path}"])


class ParquetReader(BaseReader):
    """Reader for Parquet files."""

    def read(self, path: str | Path) -> DataContainer:
        """Read Parquet file into DataContainer."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars required for Parquet support. Run: pip install polars")

        df = pl.read_parquet(path)
        return DataContainer.from_polars(df)


class CSVReader(BaseReader):
    """Reader for CSV files."""

    def read(self, path: str | Path) -> DataContainer:
        """Read CSV file into DataContainer."""
        try:
            import polars as pl
        except ImportError:
            # Fallback to standard library
            import csv
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(AlignmentSample.from_dict(row))
            return DataContainer(samples=samples, provenance=[f"loaded from {path}"])

        df = pl.read_csv(path)
        return DataContainer.from_polars(df)


def read_data(path: str | Path) -> DataContainer:
    """
    Read data from file, auto-detecting format.

    Args:
        path: Path to data file

    Returns:
        DataContainer with loaded data
    """
    path = Path(path)
    suffix = path.suffix.lower()

    readers = {
        ".jsonl": JSONLReader(),
        ".json": JSONLReader(),
        ".parquet": ParquetReader(),
        ".csv": CSVReader(),
    }

    reader = readers.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported file format: {suffix}")

    return reader.read(path)
