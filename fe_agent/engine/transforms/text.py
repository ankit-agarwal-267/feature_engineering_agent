import pandas as pd
import polars as pl
from typing import Any, List, Optional, Union, Dict
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig
from fe_agent.decisions.decision_log import DecisionRecord

class TextTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def extract_features(self, col_data: Any, profile: ColumnProfile) -> Optional[RawDataFrame]:
        """
        Condition: SemanticType == TEXT.
        """
        if profile.semantic_type != SemanticType.TEXT:
            return None
            
        if isinstance(col_data, pd.Series):
            res = pd.DataFrame(index=col_data.index)
            col_str = col_data.astype(str)
            
            res[f"{profile.name}_char_count"] = col_str.str.len().astype('Int32')
            res[f"{profile.name}_word_count"] = col_str.str.split().str.len().astype('Int32')
            res[f"{profile.name}_uppercase_ratio"] = (col_str.str.findall(r'[A-Z]').str.len() / col_str.str.len()).astype('float32')
            
            return res
        else:
            # Polars extract text features
            col_str = col_data.cast(pl.String)
            
            new_cols = [
                col_str.str.len_chars().alias(f"{profile.name}_char_count").cast(pl.Int32),
                col_str.str.split(by=" ").list.len().alias(f"{profile.name}_word_count").cast(pl.Int32)
            ]
            
            # Simplified for uppercase ratio
            # res[f"{profile.name}_uppercase_ratio"] = ...
            
            return pl.DataFrame(new_cols)
