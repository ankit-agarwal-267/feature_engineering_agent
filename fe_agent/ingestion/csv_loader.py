import csv
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Dict, Union
from fe_agent.ingestion.base import AbstractDataLoader
from fe_agent.profiler.semantic_types import RawDataFrame

class CSVLoader(AbstractDataLoader):
    def supports(self, source: str | dict) -> bool:
        if isinstance(source, str):
            return source.lower().endswith(('.csv', '.tsv', '.txt'))
        return False

    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        if not isinstance(source, str):
            raise ValueError("CSVLoader requires a file path string.")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Auto-detect delimiter
        delimiter = kwargs.get('sep') or kwargs.get('delimiter')
        if not delimiter:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    sample = f.read(8192)
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
            except Exception:
                delimiter = ',' # Fallback to comma

        backend = kwargs.get('backend', 'pandas')
        
        # Date/time heuristic
        datetime_keywords = {'date', 'time', '_at', '_on', 'timestamp', 'dt'}
        
        if backend == 'pandas':
            # Pre-read to get columns for datetime heuristic
            df_preview = pd.read_csv(path, sep=delimiter, nrows=0)
            parse_dates = [col for col in df_preview.columns if any(kw in col.lower() for kw in datetime_keywords)]
            return pd.read_csv(path, sep=delimiter, parse_dates=parse_dates, **{k: v for k, v in kwargs.items() if k not in ('backend', 'sep', 'delimiter')})
        elif backend == 'polars':
            # Polars doesn't have a direct 'parse_dates' list like pandas in read_csv
            # but we can try auto-parsing.
            return pl.read_csv(path, separator=delimiter, try_parse_dates=True, **{k: v for k, v in kwargs.items() if k not in ('backend', 'separator', 'delimiter')})
        else:
            raise ValueError(f"Unsupported backend: {backend}")
