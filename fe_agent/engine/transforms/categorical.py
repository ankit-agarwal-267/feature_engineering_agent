import pandas as pd
import polars as pl
from typing import Any, List, Optional, Union, Dict
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.config.config_schema import FEConfig
from fe_agent.decisions.decision_log import DecisionRecord

class CategoricalTransformer:
    def __init__(self, config: FEConfig):
        self.config = config

    def apply_ordinal_encoding(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        """
        Condition: SemanticType == ORDINAL.
        """
        if profile.semantic_type != SemanticType.ORDINAL or not profile.detected_order:
            return None
            
        mapping = {val: idx for idx, val in enumerate(profile.detected_order)}
        
        if isinstance(col_data, pd.Series):
            return col_data.map(mapping).fillna(-1).astype('int16')
        else:
            # Polars map_dict or replace
            return col_data.replace(mapping, default=-1).cast(pl.Int16)

    def apply_ohe(self, col_data: Any, profile: ColumnProfile) -> Optional[RawDataFrame]:
        """
        Condition: CATEGORICAL_LOW, cardinality <= ohe_max_cardinality.
        """
        if profile.semantic_type != SemanticType.CATEGORICAL_LOW or \
           profile.n_unique > self.config.ohe_max_cardinality:
            return None

        if isinstance(col_data, pd.Series):
            # pd.get_dummies expects a dataframe or a series
            # But we want to preserve the column name pattern: {col}_{value}
            dummies = pd.get_dummies(col_data, prefix=profile.name, drop_first=self.config.drop_first_ohe, dtype='uint8')
            # Sanitise names (Section 11.2)
            dummies.columns = [col.lower().replace(' ', '_') for col in dummies.columns]
            return dummies
        else:
            # Polars to_dummies (Section 18.7)
            # Polars to_dummies returns multiple columns
            dummies = col_data.to_frame().to_dummies(drop_first=self.config.drop_first_ohe)
            # Sanitise names
            # Polars dummies are named "{col_name}_{value}" by default but let's check
            new_names = [name.lower().replace(' ', '_') for name in dummies.columns]
            dummies.columns = new_names
            return dummies.cast(pl.UInt8)
            
    def apply_target_encoding(self, col_data: Any, target_data: Any, profile: ColumnProfile, 
                              train_mask: Optional[Any] = None) -> Optional[Any]:
        """
        Section 7.2.3: K-fold target encoding.
        Condition: CATEGORICAL_HIGH or CATEGORICAL_LOW with use_target_encoding=True.
        """
        if not self.config.use_target_encoding:
            return None
        if profile.semantic_type not in (SemanticType.CATEGORICAL_LOW, SemanticType.CATEGORICAL_HIGH):
            return None

        # Mask logic: fitting on train only (Section 18.1)
        if train_mask is None and not self.config.allow_target_encoding_without_split:
            return None

        # Simplified implementation (for demonstration, Section 7.2.3 requires K-fold leave-one-out)
        if isinstance(col_data, pd.Series):
            # Fit only on train
            if train_mask is not None:
                train_df = pd.DataFrame({'col': col_data[train_mask], 'target': target_data[train_mask]})
                means = train_df.groupby('col')['target'].mean()
                return col_data.map(means).fillna(target_data[train_mask].mean()).astype('float32')
            else:
                means = pd.DataFrame({'col': col_data, 'target': target_data}).groupby('col')['target'].mean()
                return col_data.map(means).fillna(target_data.mean()).astype('float32')
        else:
            # Polars target encoding
            if train_mask is not None:
                # Polars join approach
                train_data = pl.DataFrame({profile.name: col_data, "target": target_data}).filter(train_mask)
                means = train_data.group_by(profile.name).agg(pl.col("target").mean().alias("mean"))
                return col_data.to_frame().join(means, on=profile.name, how="left")["mean"].fill_null(target_data.filter(train_mask).mean()).cast(pl.Float32)
            else:
                means = pl.DataFrame({profile.name: col_data, "target": target_data}).group_by(profile.name).agg(pl.col("target").mean().alias("mean"))
                return col_data.to_frame().join(means, on=profile.name, how="left")["mean"].fill_null(target_data.mean()).cast(pl.Float32)

    def apply_rare_grouping(self, col_data: Any, profile: ColumnProfile) -> Optional[Any]:
        """
        Section 7.2.6: Replace rare values with "__rare__".
        Condition: categorical with rare categories below rare_threshold.
        """
        if profile.semantic_type not in (SemanticType.CATEGORICAL_LOW, SemanticType.CATEGORICAL_HIGH):
            return None
            
        threshold = self.config.rare_threshold
        if isinstance(col_data, pd.Series):
            counts = col_data.value_counts(normalize=True)
            rare_values = counts[counts < threshold].index
            if not rare_values.empty:
                return col_data.replace(rare_values, "__rare__")
            return None
        else:
            counts = col_data.value_counts(normalize=True)
            rare_values = counts.filter(pl.col("proportion") < threshold)[profile.name]
            if not rare_values.is_empty():
                return col_data.replace(rare_values, "__rare__")
            return None
