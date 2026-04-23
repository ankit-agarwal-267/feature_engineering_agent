from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.stats import chi2_contingency
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame

class InformationValueScorer:
    """
    Ranks features by their predictive power for generic classification tasks.
    """
    def score(self, df: RawDataFrame, target: str, profiles: List[ColumnProfile]) -> Dict[str, Dict[str, float]]:
        results = {}
        is_pandas = isinstance(df, pd.DataFrame)
        y_raw = df[target].astype(str).values if is_pandas else df[target].cast(pl.String).to_numpy()
        le = LabelEncoder()
        y_multi = le.fit_transform(y_raw)
        
        for p in profiles:
            if p.name == target: continue
            
            # Prepare feature data
            col_raw = df[p.name]
            
            # 1. MI (Generic)
            mi = 0.0
            try:
                # MI requires numeric inputs
                col_numeric = pd.to_numeric(col_raw.astype(str), errors='coerce').fillna(0) if is_pandas \
                              else col_raw.cast(pl.Float64, strict=False).fill_null(0).to_numpy()
                X = col_numeric.values.reshape(-1, 1) if is_pandas else col_numeric.reshape(-1, 1)
                mi = float(mutual_info_classif(X, y_multi, discrete_features='auto')[0])
            except: pass

            # 2. ANOVA (F-Test) for Numeric
            anova = 0.0
            if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE):
                try:
                    col_numeric = pd.to_numeric(col_raw, errors='coerce').fillna(0) if is_pandas \
                                  else col_raw.cast(pl.Float64, strict=False).fill_null(0).to_numpy()
                    anova, _ = f_classif(col_numeric.values.reshape(-1, 1), y_multi)
                    anova = float(anova[0])
                except: pass

            # 3. Cramer's V for Categorical
            cramer = 0.0
            if p.semantic_type in (SemanticType.CATEGORICAL_LOW, SemanticType.CATEGORICAL_HIGH, SemanticType.BOOLEAN):
                try:
                    ct = pd.crosstab(col_raw.astype(str), y_raw)
                    chi2 = chi2_contingency(ct)[0]
                    n = ct.sum().sum()
                    phi2 = chi2 / n
                    r, k = ct.shape
                    cramer = float(np.sqrt(phi2 / min(k-1, r-1))) if min(k-1, r-1) > 0 else 0.0
                except: pass

            results[p.name] = {"mi": mi, "anova": anova, "cramer": cramer}
        return results
