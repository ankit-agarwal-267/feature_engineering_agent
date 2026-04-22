from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Dict
import pandas as pd
import polars as pl
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType, RawDataFrame
from fe_agent.decisions.decision_log import DecisionRecord
from fe_agent.config.config_schema import FEConfig

@dataclass
class TransformResult:
    new_columns: RawDataFrame
    decision_record: DecisionRecord
    replace_source: bool = False

from fe_agent.engine.transforms.numeric import NumericTransformer
from fe_agent.engine.transforms.categorical import CategoricalTransformer
from fe_agent.engine.transforms.datetime import DateTimeTransformer
from fe_agent.engine.transforms.boolean import BooleanTransformer
from fe_agent.engine.transforms.text import TextTransformer
from fe_agent.pipeline.pipeline_artifact import FEPipeline, FEStep

class FEEngine:
    def __init__(self, config: FEConfig):
        self.config = config
        self.numeric_transformer = NumericTransformer(config)
        self.categorical_transformer = CategoricalTransformer(config)
        self.datetime_transformer = DateTimeTransformer(config)
        self.boolean_transformer = BooleanTransformer(config)
        self.text_transformer = TextTransformer(config)

    def transform(self, df: RawDataFrame, profiles: List[ColumnProfile], decisions: List[DecisionRecord], 
                  train_mask: Optional[Any] = None) -> tuple[RawDataFrame, FEPipeline]:
        """
        Orchestrates transforms (Section 7).
        """
        pipeline = FEPipeline()
        transformed_df = df.copy() if isinstance(df, pd.DataFrame) else df.clone()
        
        # 1. Pre-pass (Section 7.6): Duplicate / Quasi-constant
        cols_to_drop = [d.column_name for d in decisions if d.decision == "dropped"]
        
        # Detect duplicates
        if isinstance(transformed_df, pd.DataFrame):
            duplicates = transformed_df.columns[transformed_df.T.duplicated()]
            for dup in duplicates:
                if dup != self.config.target_column and dup not in cols_to_drop:
                    cols_to_drop.append(dup)
        
        # 2. Apply decisions
        target_col_data = transformed_df[self.config.target_column]
        
        for decision in decisions:
            if decision.decision != "accepted":
                continue
                
            col_name = decision.column_name
            profile = next((p for p in profiles if p.name == col_name), None)
            if not profile: continue
            
            col_data = transformed_df[col_name]
            new_cols_data = None
            
            if decision.transform_name == "log1p":
                new_val = self.numeric_transformer.apply_log1p(col_data, profile)
                if new_val is not None:
                    transformed_df[decision.output_columns[0]] = new_val
                    new_cols_data = [decision.output_columns[0]]
            
            elif decision.transform_name == "sqrt":
                new_val = self.numeric_transformer.apply_sqrt(col_data, profile)
                if new_val is not None:
                    transformed_df[decision.output_columns[0]] = new_val
                    new_cols_data = [decision.output_columns[0]]
            
            elif decision.transform_name == "polynomial":
                new_val = self.numeric_transformer.apply_polynomial(col_data, profile)
                if new_val is not None:
                    transformed_df[decision.output_columns[0]] = new_val
                    new_cols_data = [decision.output_columns[0]]

            elif decision.transform_name == "binning":
                bin_df = self.numeric_transformer.apply_binning(col_data, profile)
                if bin_df is not None:
                    if isinstance(transformed_df, pd.DataFrame):
                        transformed_df = pd.concat([transformed_df, bin_df], axis=1)
                        new_cols_data = bin_df.columns.tolist()
                    else:
                        transformed_df = pl.concat([transformed_df, bin_df], how="horizontal")
                        new_cols_data = bin_df.columns

            elif decision.transform_name == "one_hot_encoding":
                ohe_df = self.categorical_transformer.apply_ohe(col_data, profile)
                if ohe_df is not None:
                    if isinstance(transformed_df, pd.DataFrame):
                        transformed_df = pd.concat([transformed_df, ohe_df], axis=1)
                        new_cols_data = ohe_df.columns.tolist()
                    else:
                        transformed_df = pl.concat([transformed_df, ohe_df], how="horizontal")
                        new_cols_data = ohe_df.columns
                    cols_to_drop.append(col_name)

            elif decision.transform_name == "target_encoding":
                new_val = self.categorical_transformer.apply_target_encoding(col_data, target_col_data, profile, train_mask)
                if new_val is not None:
                    transformed_df[decision.output_columns[0]] = new_val
                    new_cols_data = [decision.output_columns[0]]

            elif decision.transform_name == "datetime_extraction":
                dt_df = self.datetime_transformer.extract_features(col_data, profile)
                if dt_df is not None:
                    if isinstance(transformed_df, pd.DataFrame):
                        transformed_df = pd.concat([transformed_df, dt_df], axis=1)
                        new_cols_data = dt_df.columns.tolist()
                    else:
                        transformed_df = pl.concat([transformed_df, dt_df], how="horizontal")
                        new_cols_data = dt_df.columns
                    if self.config.drop_source_datetime:
                        cols_to_drop.append(col_name)

            elif decision.transform_name == "boolean_cast":
                new_val = self.boolean_transformer.apply_bool_cast(col_data, profile)
                if new_val is not None:
                    transformed_df[col_name] = new_val
                    new_cols_data = [col_name]

            elif decision.transform_name == "text_fe":
                text_df = self.text_transformer.extract_features(col_data, profile)
                if text_df is not None:
                    if isinstance(transformed_df, pd.DataFrame):
                        transformed_df = pd.concat([transformed_df, text_df], axis=1)
                        new_cols_data = text_df.columns.tolist()
                    else:
                        transformed_df = pl.concat([transformed_df, text_df], how="horizontal")
                        new_cols_data = text_df.columns
                    if not self.config.keep_source_text:
                        cols_to_drop.append(col_name)

            # Record step in pipeline
            if new_cols_data:
                pipeline.add_step(FEStep(
                    step_id=f"step_{len(pipeline.steps)}",
                    transform_name=decision.transform_name,
                    source_columns=[col_name],
                    output_columns=list(new_cols_data)
                ))

        # 3. Ratio pairs (Section 7.1.6)
        for pair in self.config.ratio_pairs:
            if len(pair) == 2 and pair[0] in transformed_df.columns and pair[1] in transformed_df.columns:
                ratio_val = self.numeric_transformer.apply_ratio(transformed_df[pair[0]], transformed_df[pair[1]], pair[0], pair[1])
                ratio_name = f"{pair[0]}_div_{pair[1]}"
                transformed_df[ratio_name] = ratio_val
                pipeline.add_step(FEStep(step_id=f"ratio_{ratio_name}", transform_name="ratio", source_columns=pair, output_columns=[ratio_name]))

        # Final cleanup: drop columns
        unique_drops = list(set(cols_to_drop))
        if isinstance(transformed_df, pd.DataFrame):
            transformed_df = transformed_df.drop(columns=[c for c in unique_drops if c in transformed_df.columns])
        else:
            transformed_df = transformed_df.drop([c for c in unique_drops if c in transformed_df.columns])

        return transformed_df, pipeline
