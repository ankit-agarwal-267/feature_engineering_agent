import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Dict, Union
from fe_agent.ingestion.base import AbstractDataLoader
from fe_agent.profiler.semantic_types import RawDataFrame

class ParquetLoader(AbstractDataLoader):
    def supports(self, source: str | dict) -> bool:
        if isinstance(source, str):
            return source.lower().endswith(('.parquet', '.pq'))
        return False

    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        if not isinstance(source, str):
            raise ValueError("ParquetLoader requires a file path string.")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        backend = kwargs.get('backend', 'pandas')
        
        if backend == 'pandas':
            return pd.read_parquet(path, **{k: v for k, v in kwargs.items() if k != 'backend'})
        elif backend == 'polars':
            return pl.read_parquet(path, **{k: v for k, v in kwargs.items() if k != 'backend'})
        else:
            raise ValueError(f"Unsupported backend: {backend}")
