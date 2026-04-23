import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import mutual_info_classif
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame

class InformationValueScorer:
    """
    Ranks features by their predictive power for the target (Generic Binary/Multiclass).
    """
    def score(self, df: RawDataFrame, target: str, profiles: List[ColumnProfile]) -> Dict[str, Dict[str, float]]:
        results = {}
        
        # Generic Target Encoding
        if isinstance(df, pd.DataFrame):
            le = LabelEncoder()
            y = le.fit_transform(df[target].astype(str))
            df_numeric = df.copy()
        else:
            y = LabelEncoder().fit_transform(df[target].cast(pl.String).to_pandas())
            df_numeric = df.to_pandas()
        
        for p in profiles:
            if p.name == target: continue
            
            # Prepare feature
            col_data = df_numeric[p.name]
            if not pd.api.types.is_numeric_dtype(col_data):
                col_data = col_data.astype('category').cat.codes
            
            # MI (Multiclass compatible)
            mi = 0.0
            try:
                X = col_data.fillna(col_data.median()).values.reshape(-1, 1)
                mi = float(mutual_info_classif(X, y, discrete_features='auto')[0])
            except: pass

            results[p.name] = {"mi": mi, "iv": 0.0, "corr": 0.0}
        return results

    def calculate_iv(self, df_pd: pd.DataFrame, feature: str, target: str) -> float:
        # Note: IV is inherently binary. For multiclass, we use MI.
        return 0.0
