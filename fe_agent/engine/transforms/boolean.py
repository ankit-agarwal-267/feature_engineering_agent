import pandas as pd
import polars as pl
from typing import Any, List, Optional, Union, Dict
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig
from fe_agent.decisions.decision_log import DecisionRecord

class BooleanTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def apply_bool_cast(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        """
        Condition: SemanticType == BOOLEAN.
        """
        if profile.semantic_type != SemanticType.BOOLEAN:
            return None
            
        if isinstance(col_data, pd.Series):
            # Map standard booleans if they are strings
            if col_data.dtype == 'object' or col_data.dtype == 'string':
                bool_map = {
                    'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0, '1': 1, '0': 0,
                    'active': 1, 'inactive': 0
                }
                # Apply map after lowercasing
                return col_data.astype(str).str.lower().map(bool_map).fillna(0).astype('int8')
            return col_data.astype('int8')
        else:
            # Polars boolean cast
            if col_data.dtype == pl.String:
                # String map logic
                # For now simplified: cast to boolean if possible, then to int8
                return col_data.str.to_lowercase().replace({
                    'true': '1', 'false': '0', 'yes': '1', 'no': '0', 'y': '1', 'n': '0', '1': '1', '0': '0',
                    'active': '1', 'inactive': '0'
                }).cast(pl.Int8)
            return col_data.cast(pl.Int8)
