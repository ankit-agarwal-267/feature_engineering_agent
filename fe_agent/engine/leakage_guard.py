import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig

class LeakageGuard:
    """
    Section 7.8: Detects potential target leakage in engineered features.
    """
    def __init__(self, config: FEConfig):
        self.config = config

    def check_leakage(self, df: RawDataFrame, target: str) -> List[str]:
        """
        Runs leakage detection checks and returns a list of warnings.
        """
        warnings = []
        target_col = df[target] if isinstance(df, pd.DataFrame) else df[target]
        
        for col_name in df.columns:
            if col_name == target:
                continue

            # 1. Direct correlation check (Section 7.8.1)
            corr = self._calculate_correlation(df[col_name], target_col)
            if corr is not None and abs(corr) > self.config.leakage_correlation_threshold:
                warnings.append(f"Potential leakage: '{col_name}' has high correlation ({corr:.4f}) with target '{target}'.")

            # 2. Name heuristics (Section 7.8.2)
            if target.lower() in col_name.lower() and col_name != target:
                warnings.append(f"Potential leakage: '{col_name}' contains target name heuristics.")

        return warnings

    def _calculate_correlation(self, col: Any, target: Any) -> Optional[float]:
        try:
            if isinstance(col, pd.Series):
                # Using Pearson for numeric; point-biserial would be better for categorical target
                # but pandas .corr defaults to Pearson
                return col.corr(target)
            else:
                # Polars correlation
                return pl.corr(col, target)
        except Exception:
            return None
