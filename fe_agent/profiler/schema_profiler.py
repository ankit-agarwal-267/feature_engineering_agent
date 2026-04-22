import pandas as pd
import polars as pl
import numpy as np
from typing import Any, List, Optional, Union
from fe_agent.profiler.semantic_types import SemanticType, ColumnProfile, RawDataFrame
from fe_agent.config.config_schema import FEConfig

class SchemaProfiler:
    def __init__(self, config: FEConfig):
        self.config = config

    def profile(self, df: RawDataFrame) -> List[ColumnProfile]:
        """
        Profiles every column in the dataframe and assigns a SemanticType.
        """
        profiles = []
        n_rows = len(df)
        
        # Section 18.4: Sampling for distribution statistics on large datasets
        if n_rows > 5000:
            if isinstance(df, pd.DataFrame):
                df_sample = df.sample(n=5000, random_state=self.config.random_seed)
            else:
                df_sample = df.sample(n=5000, seed=self.config.random_seed)
        else:
            df_sample = df

        for col_name in df.columns:
            if isinstance(df, pd.DataFrame):
                col_data = df[col_name]
                col_sample = df_sample[col_name]
                raw_dtype = str(col_data.dtype)
                n_unique = col_data.nunique()
                null_count = col_data.isna().sum()
                sample_values = col_sample.dropna().head(10).tolist()
            else:
                col_data = df[col_name]
                col_sample = df_sample[col_name]
                raw_dtype = str(col_data.dtype)
                n_unique = col_data.n_unique()
                null_count = col_data.null_count()
                sample_values = col_sample.drop_nulls().head(10).to_list()

            null_pct = null_count / n_rows if n_rows > 0 else 0
            cardinality_ratio = n_unique / n_rows if n_rows > 0 else 0
            
            # Initial distribution flags
            is_skewed = False
            skewness = None
            has_negative = False
            has_zero = False
            
            # Check for numeric columns
            is_numeric = False
            if isinstance(df, pd.DataFrame):
                is_numeric = pd.api.types.is_numeric_dtype(col_data)
            else:
                is_numeric = col_data.dtype.is_numeric()

            if is_numeric:
                # Distribution stats
                if isinstance(df, pd.DataFrame):
                    skewness = col_sample.skew()
                    has_negative = (col_data < 0).any()
                    has_zero = (col_data == 0).any()
                else:
                    skewness = col_sample.skew()
                    has_negative = (col_data < 0).any()
                    has_zero = (col_data == 0).any()
                
                if skewness is not None and abs(skewness) > 1.0:
                    is_skewed = True

            # Semantic Type Inference
            semantic_type = self._infer_semantic_type(
                col_name, col_sample, raw_dtype, n_unique, cardinality_ratio, is_numeric, n_rows
            )

            profile = ColumnProfile(
                name=col_name,
                raw_dtype=raw_dtype,
                semantic_type=semantic_type,
                inferred_semantic_type=semantic_type,
                n_unique=n_unique,
                null_count=int(null_count),
                null_pct=null_pct,
                sample_values=sample_values,
                is_skewed=is_skewed,
                skewness=skewness,
                has_negative=has_negative,
                has_zero=has_zero,
                cardinality_ratio=cardinality_ratio
            )
            profiles.append(profile)
            
        return profiles

    def _infer_semantic_type(self, name: str, sample: Any, raw_dtype: str, n_unique: int, 
                             cardinality_ratio: float, is_numeric: bool, n_rows: int) -> SemanticType:
        """
        Implements deterministic profiling rules (Section 6.3).
        """
        # RULE 1: Constant check
        if n_unique == 1:
            return SemanticType.CONSTANT

        # RULE 1.5: Quasi-constant check (Section 18.5)
        if n_rows > 0:
            if isinstance(sample, pd.Series):
                max_freq = sample.value_counts(normalize=True).max()
            else:
                max_freq = sample.value_counts(normalize=True)["proportion"].max()
            
            if max_freq > self.config.quasi_constant_threshold:
                return SemanticType.CONSTANT # Treated as constant for dropping

        # RULE 2: ID check
        if cardinality_ratio > self.config.id_cardinality_ratio:
            # Check if dtype is int/string
            if "int" in raw_dtype.lower() or "str" in raw_dtype.lower() or "object" in raw_dtype.lower():
                return SemanticType.ID_COLUMN

        # RULE 3: Boolean check
        if n_unique <= 2:
            # Subset of {0, 1, True, False, "yes", "no", "y", "n", "true", "false"}
            bool_vocab = {0, 1, True, False, "yes", "no", "y", "n", "true", "false", "Active", "Inactive"}
            unique_values = set()
            if isinstance(sample, pd.Series):
                unique_values = set(sample.dropna().unique())
            else:
                unique_values = set(sample.drop_nulls().unique())
            
            if all(str(v).lower() in {str(bv).lower() for bv in bool_vocab} for v in unique_values):
                return SemanticType.BOOLEAN

        # RULE 4: DateTime check (Enhanced)
        if isinstance(sample, pd.Series):
            try:
                # 1. Sample based inspection
                sample_str = sample.dropna().head(30).astype(str).tolist()
                
                # 2. Check for keywords
                date_keywords = {'date', 'time', '_at', '_on', 'timestamp', 'dt'}
                is_keyword_match = any(kw in name.lower() for kw in date_keywords)
                
                # 3. Optional LLM inference if enabled
                llm_verdict = False
                if self.config.llm.enabled and self.config.llm.enabled:
                    # Simple heuristic: only call LLM if keyword match or looks like potential date
                    prompt = f"Is the following column '{name}' a datetime? Values: {sample_str}. Respond 'yes' or 'no'."
                    # (In actual implementation, we'd use the LLMAdvisor instance)
                    # For now, stick to robust heuristics to avoid unnecessary complex dependency chains
                    pass

                # 4. Conversion check (very high threshold)
                if is_keyword_match:
                    parsed = pd.to_datetime(sample.dropna().head(30), errors='coerce')
                    if parsed.notna().sum() > len(parsed) * 0.9:
                        return SemanticType.DATETIME
            except:
                pass

        # RULE 5: Text check
        if "object" in raw_dtype.lower() or "str" in raw_dtype.lower() or "string" in raw_dtype.lower():
            # calculate avg token count
            if isinstance(sample, pd.Series):
                avg_token_count = sample.dropna().astype(str).str.split().str.len().mean()
            else:
                avg_token_count = sample.drop_nulls().cast(pl.String).str.split(by=" ").list.len().mean()
            
            if avg_token_count is not None and avg_token_count > 5:
                return SemanticType.TEXT

        # RULE 6: Categorical vs Ordinal (Simplified ordinal detection)
        if ("object" in raw_dtype.lower() or "str" in raw_dtype.lower() or "category" in raw_dtype.lower()):
            if n_unique <= self.config.high_cardinality_threshold:
                return SemanticType.CATEGORICAL_LOW
            else:
                return SemanticType.CATEGORICAL_HIGH

        # RULE 7: Numeric discrete vs continuous
        if is_numeric:
            if "int" in raw_dtype.lower() and n_unique <= self.config.high_cardinality_threshold:
                return SemanticType.NUMERIC_DISCRETE
            else:
                return SemanticType.NUMERIC_CONTINUOUS

        return SemanticType.UNKNOWN
