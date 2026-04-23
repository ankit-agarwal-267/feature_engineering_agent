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
        
        # Prepare targets
        y_raw = df[target].astype(str).values if is_pandas else df[target].cast(pl.String).to_numpy()
        unique_labels = np.unique(y_raw)
        is_binary = len(unique_labels) == 2
        
        # Binary target for IV/Corr
        y_bin = None
        if is_binary:
            # Consistent mapping: first unique value is 1, second is 0
            y_bin = (y_raw == unique_labels[0]).astype(int)
        
        # Multi-class target for MI/ANOVA
        le = LabelEncoder()
        y_multi = le.fit_transform(y_raw)
        
        for p in profiles:
            if p.name == target: continue
            
            col_raw = df[p.name]
            
            # 1. MI (Generic)
            mi = 0.0
            try:
                # MI requires numeric inputs; handle categoricals via label encoding
                if is_pandas:
                    col_numeric = pd.to_numeric(col_raw, errors='coerce')
                    if col_numeric.isna().any():
                         col_numeric = col_raw.astype(str).astype('category').cat.codes
                    X = col_numeric.fillna(0).values.reshape(-1, 1)
                else:
                    X = col_raw.cast(pl.Float64, strict=False).fill_null(0).to_numpy().reshape(-1, 1)
                
                mi = float(mutual_info_classif(X, y_multi, discrete_features='auto')[0])
            except: pass

            # 2. ANOVA (F-Test) for Numeric
            anova = 0.0
            if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE):
                try:
                    if is_pandas:
                        col_numeric = pd.to_numeric(col_raw, errors='coerce').fillna(0)
                        X = col_numeric.values.reshape(-1, 1)
                    else:
                        X = col_raw.cast(pl.Float64, strict=False).fill_null(0).to_numpy().reshape(-1, 1)
                    
                    res_anova, _ = f_classif(X, y_multi)
                    anova = float(res_anova[0])
                except: pass

            # 3. Cramer's V for Categorical
            cramer = 0.0
            try:
                # Works for any target type
                ct = pd.crosstab(col_raw.astype(str), y_raw)
                chi2 = chi2_contingency(ct)[0]
                n = ct.sum().sum()
                phi2 = chi2 / n
                r, k = ct.shape
                if min(k-1, r-1) > 0:
                    cramer = float(np.sqrt(phi2 / min(k-1, r-1)))
            except: pass

            # 4. Correlation (Pearson - Binary only)
            corr = 0.0
            if is_binary:
                try:
                    if is_pandas:
                        col_numeric = pd.to_numeric(col_raw, errors='coerce').fillna(0)
                        corr = float(col_numeric.corr(pd.Series(y_bin)))
                    else:
                        col_numeric = col_raw.cast(pl.Float64, strict=False).fill_null(0)
                        corr = float(pl.corr(col_numeric, pl.Series(y_bin)))
                except: pass

            # 5. IV (Information Value - Binary only)
            iv = 0.0
            if is_binary:
                try:
                    iv = self.calculate_iv_binary(col_raw, y_bin)
                except: pass

            results[p.name] = {"mi": mi, "anova": anova, "cramer": cramer, "iv": iv, "corr": corr}
            
        return results

    def calculate_iv_binary(self, col_raw: Any, y_bin: np.ndarray) -> float:
        # Bin numeric features for stable IV
        is_series = isinstance(col_raw, pd.Series)
        feat = col_raw.copy() if is_series else pd.Series(col_raw.to_numpy())
        
        if pd.api.types.is_numeric_dtype(feat) and feat.nunique() > 10:
            feat = pd.qcut(feat, q=10, duplicates='drop', labels=False)
        
        df_iv = pd.DataFrame({'feat': feat.astype(str), 'target': y_bin})
        total_pos = float(df_iv['target'].sum())
        total_neg = float(len(df_iv) - total_pos)
        if total_pos == 0 or total_neg == 0: return 0.0
        
        counts = df_iv.groupby('feat')['target'].agg(['count', 'sum'])
        counts.columns = ['total', 'pos']
        counts['neg'] = counts['total'] - counts['pos']
        
        # IV formula
        p = (counts['pos'] / total_pos) + 1e-9
        n = (counts['neg'] / total_neg) + 1e-9
        counts['iv'] = (p - n) * np.log(p / n)
        return float(counts['iv'].sum())
