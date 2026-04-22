import pandas as pd
import polars as pl
from typing import Any, Dict, Union
from fe_agent.ingestion.base import AbstractDataLoader
from fe_agent.profiler.semantic_types import RawDataFrame

class DictLoader(AbstractDataLoader):
    def supports(self, source: str | dict) -> bool:
        return isinstance(source, dict) and "data" in source

    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        if not isinstance(source, dict):
            raise ValueError("DictLoader requires a dictionary.")

        data = source.get("data")
        format_ = source.get("format", "records") # default to records
        
        backend = kwargs.get('backend', 'pandas')
        
        if backend == 'pandas':
            if format_ == "records":
                return pd.DataFrame(data)
            elif format_ == "columns":
                return pd.DataFrame.from_dict(data)
            else:
                return pd.DataFrame(data)
        elif backend == 'polars':
            if format_ == "records":
                return pl.DataFrame(data)
            elif format_ == "columns":
                return pl.from_dict(data)
            else:
                return pl.DataFrame(data)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
