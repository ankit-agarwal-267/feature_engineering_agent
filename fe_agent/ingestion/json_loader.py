import pandas as pd
import polars as pl
import json
from pathlib import Path
from typing import Any, Dict, Union, List
from fe_agent.ingestion.base import AbstractDataLoader
from fe_agent.profiler.semantic_types import RawDataFrame

class JSONLoader(AbstractDataLoader):
    def supports(self, source: str | dict) -> bool:
        if isinstance(source, str):
            return source.lower().endswith(('.json'))
        return False

    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        if not isinstance(source, str):
            raise ValueError("JSONLoader requires a file path string.")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        backend = kwargs.get('backend', 'pandas')
        
        # Section 5.2: Flattens one level of nesting
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            # Already records-oriented
            flattened_data = [self._flatten_one_level(item) for item in data]
        elif isinstance(data, dict):
            # Might be columnar or nested records
            # Simplified: just convert to pandas/polars and let them handle basic structures
            flattened_data = data
        else:
            flattened_data = data

        if backend == 'pandas':
            return pd.DataFrame(flattened_data, **{k: v for k, v in kwargs.items() if k != 'backend'})
        elif backend == 'polars':
            if isinstance(flattened_data, list):
                return pl.DataFrame(flattened_data, **{k: v for k, v in kwargs.items() if k != 'backend'})
            else:
                return pl.from_dict(flattened_data, **{k: v for k, v in kwargs.items() if k != 'backend'})
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _flatten_one_level(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flattens a dictionary by one level.
        """
        if not isinstance(d, dict):
            return d
        
        new_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    new_d[f"{k}_{sub_k}"] = sub_v
            else:
                new_d[k] = v
        return new_d
