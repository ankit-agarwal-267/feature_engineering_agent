import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import mutual_info_classif
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame

class InformationValueScorer:
    """
    Ranks features by their predictive power for the target.
    """
    def score(self, df: RawDataFrame, target: str, profiles: List[ColumnProfile]) -> Dict[str, Dict[str, float]]:
        results = {}
        
        # Ensure target is binary numeric
        if isinstance(df, pd.DataFrame):
            y = (df[target].astype(str).str.lower() == 'yes').astype(int)
            df_numeric = df.copy()
        else:
            y = (df[target].cast(pl.String).str.to_lowercase() == 'yes').cast(pl.Int8)
            df_numeric = df.to_pandas()
            y = y.to_pandas()
        
        for p in profiles:
            if p.name == target: continue
            
            # Prepare feature for numeric analysis (Label Encode if needed)
            col_data = df_numeric[p.name]
            if not pd.api.types.is_numeric_dtype(col_data):
                col_data = col_data.astype('category').cat.codes
            
            # MI & Corr (using encoded data)
            mi = 0.0
            corr = 0.0
            try:
                X = col_data.fillna(col_data.median()).values.reshape(-1, 1)
                mi = float(mutual_info_classif(X, y, discrete_features='auto')[0])
                corr = float(col_data.corr(y))
            except: pass

            # IV
            iv = self.calculate_iv(df_numeric, p.name, target)

            results[p.name] = {"mi": mi, "iv": iv, "corr": corr}
        return results

    def calculate_iv(self, df_pd: pd.DataFrame, feature: str, target: str) -> float:
        # Re-use the target_bin logic
        y = (df_pd[target].astype(str).str.lower() == 'yes').astype(int)
        
        feat = df_pd[feature].copy()
        if not pd.api.types.is_numeric_dtype(feat):
            feat = feat.astype('category').cat.codes
        else:
            if feat.nunique() > 10:
                feat = pd.qcut(feat, q=10, duplicates='drop', labels=False)
        
        df_iv = pd.DataFrame({'feat': feat, 'target': y})
        # Standard IV
        total_pos = float(df_iv['target'].sum())
        total_neg = float(len(df_iv) - total_pos)
        if total_pos == 0 or total_neg == 0: return 0.0
        
        counts = df_iv.groupby('feat')['target'].agg(['count', 'sum'])
        counts.columns = ['total', 'pos']
        counts['neg'] = counts['total'] - counts['pos']
        counts['iv'] = (counts['pos']/total_pos - counts['neg']/total_neg) * np.log((counts['pos']/total_pos + 1e-9) / (counts['neg']/total_neg + 1e-9))
        
        return float(counts['iv'].fillna(0).sum())
