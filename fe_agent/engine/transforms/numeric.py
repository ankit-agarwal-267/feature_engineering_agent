import numpy as np
import pandas as pd
import polars as pl
from typing import Any, List, Optional, Union
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig
from fe_agent.decisions.decision_log import DecisionRecord

class NumericTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def _is_numeric(self, col_data: Any) -> bool:
        if isinstance(col_data, pd.Series):
            return pd.api.types.is_numeric_dtype(col_data)
        return col_data.dtype.is_numeric()

    def apply_log1p(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        if not self._is_numeric(col_data): return None
        # ... (rest of method) ...
        if profile.has_negative or profile.skewness is None or abs(profile.skewness) < 0.75:
            return None
        
        if profile.has_zero:
            return None
            
        if isinstance(col_data, pd.Series):
            return np.log1p(col_data)
        else:
            return col_data.log1p()

    def apply_sqrt(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        """
        Condition: is_skewed=True, has_negative=False.
        """
        if profile.has_negative or profile.skewness is None or abs(profile.skewness) < 0.75:
            return None
            
        if isinstance(col_data, pd.Series):
            return np.sqrt(col_data)
        else:
            return col_data.sqrt()

    def apply_polynomial(self, col_data: Any, profile: ColumnProfile, degree: int = 2) -> Optional[Any]:
        """
        Generates polynomial feature of specified degree (1, 2, or 3).
        """
        if degree < 1 or degree > 3:
            return None
            
        # Ensure input is numeric
        if isinstance(col_data, pd.Series):
            col_num = pd.to_numeric(col_data, errors='coerce').fillna(0)
            res = col_num ** degree
            return res.astype('float32')
        else:
            col_num = col_data.cast(pl.Float64, strict=False).fill_null(0)
            return (col_num ** degree).alias(f"{profile.name}_pow_{degree}").cast(pl.Float32)

    def apply_binning(self, col_data: Any, profile: ColumnProfile) -> Optional[RawDataFrame]:
        """
        Condition: NUMERIC_CONTINUOUS, is_skewed=True or cardinality > 50.
        Section 7.1.5.
        """
        if profile.semantic_type != SemanticType.NUMERIC_CONTINUOUS or \
           (not profile.is_skewed and profile.n_unique <= 50) or \
           profile.n_unique < 10:
            return None

        n_bins = self.config.n_bins
        if isinstance(col_data, pd.Series):
            res = pd.DataFrame(index=col_data.index)
            # Strategy A: Quantile binning (default)
            res[f"{profile.name}_qbin"] = pd.qcut(col_data, q=n_bins, labels=False, duplicates='drop').astype('Int8')
            
            # Strategy B: Equal-width binning
            if self.config.equal_width_bins:
                res[f"{profile.name}_ebin"] = pd.cut(col_data, bins=n_bins, labels=False).astype('Int8')
            return res
        else:
            # Polars binning
            qbin = col_data.qcut(q=n_bins, labels=[str(i) for i in range(n_bins)]).cast(pl.Int8).alias(f"{profile.name}_qbin")
            new_cols = [qbin]
            if self.config.equal_width_bins:
                ebin = col_data.cut(bins=n_bins, labels=[str(i) for i in range(n_bins)]).cast(pl.Int8).alias(f"{profile.name}_ebin")
                new_cols.append(ebin)
            return pl.DataFrame(new_cols)

    def apply_boxcox(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        """
        Section 7.1.3: scipy.stats.boxcox.
        """
        if not self.config.apply_boxcox:
            return None
        if profile.has_negative or profile.has_zero or profile.skewness is None or abs(profile.skewness) < 0.75:
            return None
        
        try:
            from scipy import stats
            if isinstance(col_data, pd.Series):
                transformed, _ = stats.boxcox(col_data.dropna())
                res = pd.Series(index=col_data.index, dtype='float32')
                res.loc[col_data.notna()] = transformed.astype('float32')
                return res
            else:
                return None
        except ImportError:
            return None

    def apply_ratio(self, col_a_data: Any, col_b_data: Any, a_name: str, b_name: str) -> Optional[Any]:
        """
        Section 7.1.6: col_a / (col_b + epsilon).
        """
        epsilon = 1e-8
        if isinstance(col_a_data, pd.Series):
            res = col_a_data / (col_b_data + epsilon)
            res.replace([np.inf, -np.inf], np.nan, inplace=True)
            return res.astype('float32')
        else:
            res = col_a_data / (col_b_data + epsilon)
            return res.alias(f"{a_name}_div_{b_name}").cast(pl.Float32)
