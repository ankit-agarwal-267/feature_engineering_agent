import pandas as pd
import polars as pl
import numpy as np
from typing import List, Tuple, Any, Optional, Dict
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig

class InteractionTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def generate_numeric_interactions(self, df: RawDataFrame, pairs: List[Tuple[str, str]]) -> RawDataFrame:
        """
        Section 7.7.1: Pairwise products for top numeric columns.
        """
        if isinstance(df, pd.DataFrame):
            res = pd.DataFrame(index=df.index)
            for col_a, col_b in pairs:
                # Force numeric conversion for both columns before multiplication
                col_a_numeric = pd.to_numeric(df[col_a], errors='coerce').fillna(0)
                col_b_numeric = pd.to_numeric(df[col_b], errors='coerce').fillna(0)
                res[f"{col_a}_x_{col_b}"] = col_a_numeric * col_b_numeric
            return res.astype('float32')
        else:
            # Polars conversion (as before)
            new_cols = []
            for col_a, col_b in pairs:
                col_a_n = df[col_a].cast(pl.Float64, strict=False).fill_null(0)
                col_b_n = df[col_b].cast(pl.Float64, strict=False).fill_null(0)
                new_cols.append((col_a_n * col_b_n).alias(f"{col_a}_x_{col_b}").cast(pl.Float32))
            return pl.DataFrame(new_cols)

    def generate_categorical_interactions(self, df: RawDataFrame, pairs: List[Tuple[str, str]]) -> RawDataFrame:
        """
        Section 7.7.2: Concatenate string values of two columns.
        """
        if isinstance(df, pd.DataFrame):
            res = pd.DataFrame(index=df.index)
            for col_a, col_b in pairs:
                res[f"{col_a}_cat_x_{col_b}"] = df[col_a].astype(str) + "_" + df[col_b].astype(str)
            return res
        else:
            new_cols = []
            for col_a, col_b in pairs:
                new_cols.append((df[col_a].cast(pl.String) + "_" + df[col_b].cast(pl.String)).alias(f"{col_a}_cat_x_{col_b}"))
            return pl.DataFrame(new_cols)

    def generate_group_stats(self, df: RawDataFrame, cat_col: str, num_col: str, aggs: List[str]) -> RawDataFrame:
        """
        Section 7.7.3: Numeric aggregates grouped by categorical categories (mean, std, min, max, median).
        """
        if isinstance(df, pd.DataFrame):
            res = pd.DataFrame(index=df.index)
            grouped = df.groupby(cat_col)[num_col]
            for agg in aggs:
                res[f"{num_col}_grp_{cat_col}_{agg}"] = grouped.transform(agg)
            return res.astype('float32')
        else:
            exprs = []
            for agg in aggs:
                # Map standard names to polars
                if agg == "mean": e = pl.col(num_col).mean()
                elif agg == "std": e = pl.col(num_col).std()
                elif agg == "min": e = pl.col(num_col).min()
                elif agg == "max": e = pl.col(num_col).max()
                elif agg == "median": e = pl.col(num_col).median()
                else: continue
                exprs.append(e.over(cat_col).alias(f"{num_col}_grp_{cat_col}_{agg}").cast(pl.Float32))
            return df.select(exprs)
