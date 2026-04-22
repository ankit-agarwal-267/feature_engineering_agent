import pandas as pd
import polars as pl
import numpy as np
from typing import Any, List, Optional, Union, Dict
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig
from fe_agent.decisions.decision_log import DecisionRecord

class DateTimeTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def extract_features(self, col_data: Any, profile: ColumnProfile) -> Optional[RawDataFrame]:
        """
        Condition: SemanticType == DATETIME.
        """
        if profile.semantic_type != SemanticType.DATETIME:
            return None
            
        if isinstance(col_data, pd.Series):
            res = pd.DataFrame(index=col_data.index)
            # Ensure it's datetime
            dt = pd.to_datetime(col_data, errors='coerce')
            
            # Verify if conversion actually succeeded
            if dt.isna().all():
                return None
            
            # Cast to datetime specifically if it worked
            dt = pd.to_datetime(dt)
            
            res[f"{profile.name}_year"] = dt.dt.year
            res[f"{profile.name}_month"] = dt.dt.month
            res[f"{profile.name}_day"] = dt.dt.day
            res[f"{profile.name}_dayofweek"] = dt.dt.dayofweek
            res[f"{profile.name}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            
            # Cyclical (Section 7.3)
            # Use explicit conversion to numeric, safe from int/str errors
            def to_f(s): return pd.to_numeric(s.fillna(0), errors='coerce')
            
            res[f"{profile.name}_month_sin"] = np.sin(2 * np.pi * to_f(dt.dt.month) / 12)
            res[f"{profile.name}_month_cos"] = np.cos(2 * np.pi * to_f(dt.dt.month) / 12)
            res[f"{profile.name}_dow_sin"] = np.sin(2 * np.pi * to_f(dt.dt.dayofweek) / 7)
            res[f"{profile.name}_dow_cos"] = np.cos(2 * np.pi * to_f(dt.dt.dayofweek) / 7)
            
            return res.astype('float32')
        else:
            # Polars extract features
            # Polars requires the column to be of type Date or Datetime
            # It should have been parsed at ingestion
            dt = col_data
            
            new_cols = [
                dt.dt.year().alias(f"{profile.name}_year").cast(pl.Int16),
                dt.dt.month().alias(f"{profile.name}_month").cast(pl.Int8),
                dt.dt.day().alias(f"{profile.name}_day").cast(pl.Int8),
                dt.dt.weekday().alias(f"{profile.name}_dayofweek").cast(pl.Int8), # Polars 1=Mon, 7=Sun or 0=Mon, 6=Sun? 1-7
                ((dt.dt.weekday().fill_null(0) >= 6).cast(pl.Int8)).alias(f"{profile.name}_is_weekend")
            ]
            
            # Cyclical
            new_cols.extend([
                (2 * np.pi * dt.dt.month().fill_null(0) / 12).sin().alias(f"{profile.name}_month_sin").cast(pl.Float32),
                (2 * np.pi * dt.dt.month().fill_null(0) / 12).cos().alias(f"{profile.name}_month_cos").cast(pl.Float32),
                (2 * np.pi * dt.dt.weekday().fill_null(0) / 7).sin().alias(f"{profile.name}_dow_sin").cast(pl.Float32),
                (2 * np.pi * dt.dt.weekday().fill_null(0) / 7).cos().alias(f"{profile.name}_dow_cos").cast(pl.Float32)
            ])
            
            return pl.DataFrame(new_cols)
