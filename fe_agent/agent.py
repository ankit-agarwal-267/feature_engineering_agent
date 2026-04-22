from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Union, Dict
import pandas as pd
import polars as pl
import numpy as np
import time
import uuid
import hashlib
import json
from datetime import datetime

from fe_agent.config.config_schema import FEConfig, LLMConfig
from fe_agent.profiler.semantic_types import ColumnProfile, RawDataFrame, SemanticType
from fe_agent.decisions.decision_log import DecisionLog, DecisionRecord
from fe_agent.ingestion.csv_loader import CSVLoader
from fe_agent.ingestion.parquet_loader import ParquetLoader
from fe_agent.ingestion.json_loader import JSONLoader
from fe_agent.ingestion.sql_loader import SQLLoader
from fe_agent.ingestion.dict_loader import DictLoader
from fe_agent.profiler.schema_profiler import SchemaProfiler
from fe_agent.profiler.override_resolver import OverrideResolver
from fe_agent.engine.fe_engine import FEEngine
from fe_agent.decisions.decision_engine import DecisionEngine
from fe_agent.docs.audit_reporter import AuditReporter
from fe_agent.pipeline.pipeline_artifact import FEPipeline, FEStep
from fe_agent.decisions.ranking import InformationValueScorer
from fe_agent.engine.transforms.interactions import InteractionTransformer
from fe_agent.ask_user import ask_user
from fe_agent.engine.leakage_guard import LeakageGuard
from fe_agent.llm.llm_advisor import LLMAdvisor

@dataclass
class LLMAdvisory:
    validated_decisions: List[str]
    challenged_decisions: List[Dict[str, Any]]
    additional_transforms: List[Dict[str, Any]]
    domain_notes: str
    web_search_performed: bool
    web_search_query: Optional[str] = None
    web_search_summary: Optional[str] = None

@dataclass
class FEResult:
    transformed_df: RawDataFrame
    decision_log: DecisionLog
    column_profiles: List[ColumnProfile]
    pipeline_artifact: FEPipeline
    output_dir: Optional[Path] = None
    run_id: str = ""
    warnings: List[str] = field(default_factory=list)
    llm_advisory: Optional[LLMAdvisory] = None
    # For HITL
    recommended_interactions: Dict[str, List[Any]] = field(default_factory=dict)

class FEJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, SemanticType):
            return str(obj)
        return super().default(obj)

from fe_agent.engine.transforms.numeric import NumericTransformer

class FeatureEngineeringAgent:
    def __init__(self, config: FEConfig, custom_transforms: Optional[List[Any]] = None):
        self.config = config
        self.custom_transforms = custom_transforms or []
        self.loaders = [
            CSVLoader(), ParquetLoader(), JSONLoader(), 
            SQLLoader(), DictLoader()
        ]
        self.numeric_transformer = NumericTransformer(config)
        self.interaction_transformer = InteractionTransformer(config)
        self.scorer = InformationValueScorer()
        self.leakage_guard = LeakageGuard(config)
        self.llm_advisor = LLMAdvisor(config)
        
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hex_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{hex_id}"

    def run(self, source: Union[str, Dict[str, Any], RawDataFrame], train_mask: Optional[Any] = None) -> FEResult:
        """
        Executes the feature engineering pipeline.
        """
        run_id = self.config.run_id or self._generate_run_id()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Ingestion
        df = None
        if isinstance(source, (pd.DataFrame, pl.DataFrame)):
            df = source
        else:
            for loader in self.loaders:
                if loader.supports(source):
                    df = loader.load(source, backend=self.config.dataframe_backend)
                    break
        if df is None: raise ValueError(f"No loader supports source: {source}")

        # 2. Profiling & Overrides
        profiler = SchemaProfiler(self.config)
        profiles = profiler.profile(df)
        resolver = OverrideResolver(self.config)
        profiles = resolver.resolve(profiles)
        
        # 3. Decisions & Base Transforms
        decision_engine = DecisionEngine(self.config)
        all_decisions = [d for p in profiles if p.name != self.config.target_column for d in decision_engine.decide(p)]
        fe_engine = FEEngine(self.config)
        transformed_df, pipeline = fe_engine.transform(df, profiles, all_decisions, train_mask)
        
        # 4. LLM Advisor
        llm_advisory_data = self.llm_advisor.review_decisions(profiles, all_decisions) if self.config.llm.enabled else None
        
        # 5. Ranking & Interaction Logic
        ranking_initial = self.scorer.score(df, self.config.target_column, profiles)
        top_features = list(ranking_initial.keys())[:self.config.interaction_top_n]
        
        # Recommended Numeric Pairs
        recommended_num = []
        if len(top_features) >= 2:
            recommended_num = [(top_features[i], top_features[j]) for i in range(len(top_features)) for j in range(i+1, len(top_features))]

        # Recommended Polynomial Features
        recommended_poly = []
        if top_features:
            recommended_poly = [{"col": feat, "degrees": [2, 3]} for feat in top_features]

        # Apply interactions if provided in config
        if self.config.selected_numeric_interactions:
            # Filter pairs where BOTH columns exist in transformed_df
            valid_num_pairs = [tuple(p) for p in self.config.selected_numeric_interactions 
                               if p[0] in transformed_df.columns and p[1] in transformed_df.columns]
            
            if valid_num_pairs:
                int_df = self.interaction_transformer.generate_numeric_interactions(transformed_df, valid_num_pairs)
                if isinstance(transformed_df, pd.DataFrame):
                    transformed_df = pd.concat([transformed_df, int_df], axis=1)
                else:
                    transformed_df = pl.concat([transformed_df, int_df], how="horizontal")
                
                for p in valid_num_pairs:
                    all_decisions.append(DecisionRecord(
                        column_name=f"{p[0]}, {p[1]}", transform_name="numeric_interaction", 
                        output_columns=[f"{p[0]}_x_{p[1]}"], decision="accepted", 
                        rule_triggered="USER_APPROVED", rationale="User approved interaction."))

        # Apply Explicit Group Stats
        if self.config.selected_group_stats:
            for item in self.config.selected_group_stats:
                num, cat = item['num'], item['cat']
                if num in transformed_df.columns and cat in transformed_df.columns:
                    grp_df = self.interaction_transformer.generate_group_stats(transformed_df, cat, num, self.config.group_stats_agg or ["mean", "std"])
                    transformed_df = pd.concat([transformed_df, grp_df], axis=1) if isinstance(transformed_df, pd.DataFrame) else pl.concat([transformed_df, grp_df], how="horizontal")
                    all_decisions.append(DecisionRecord(column_name=f"{num}, {cat}", transform_name="group_stats", output_columns=grp_df.columns.tolist(), decision="accepted", rule_triggered="USER_APPROVED", rationale="User approved group stats."))

        # Apply Explicit Polynomials
        if self.config.selected_polynomial_features:
            for item in self.config.selected_polynomial_features:
                col_name, degree = item['col'], item['degree']
                if col_name not in transformed_df.columns:
                    continue
                
                profile = next((p for p in profiles if p.name == col_name), None)
                if not profile: continue
                
                poly_val = self.numeric_transformer.apply_polynomial(transformed_df[col_name], profile, degree)
                if poly_val is not None:
                    out_name = f"{col_name}_pow_{degree}"
                    transformed_df[out_name] = poly_val
                    all_decisions.append(DecisionRecord(column_name=col_name, transform_name=f"polynomial_d{degree}", output_columns=[out_name], decision="accepted", rule_triggered="USER_APPROVED", rationale=f"User selected polynomial degree {degree}."))

        # 6. Final Pass (Scoring & Leakage)
        # We need to score ALL features that are present in the final dataset.
        # Ensure that profiles are also updated for all final features.
        final_profiles = SchemaProfiler(self.config).profile(transformed_df)
        ranking_final = self.scorer.score(transformed_df, self.config.target_column, final_profiles)
        leakage_warnings = self.leakage_guard.check_leakage(transformed_df, self.config.target_column)

        # 7. Output Assembly
        config_hash = hashlib.sha256(json.dumps(self.config.model_dump(), sort_keys=True).encode()).hexdigest()
        log = DecisionLog(
            run_id=run_id, timestamp=datetime.now().isoformat(), config_hash=config_hash,
            dataset_shape=list(transformed_df.shape), target_column=self.config.target_column, task_type=self.config.task_type,
            decisions=all_decisions, leakage_warnings=leakage_warnings,
            llm_advisory={"feature_importance_ranking": ranking_final, "llm_review": llm_advisory_data}
        )
        
        if self.config.write_audit_report:
            AuditReporter(run_id, output_dir).generate_report(log, profiles, transformed_df)
        if self.config.write_decision_log:
            with open(output_dir / f"decision_log_{run_id}.json", 'w', encoding='utf-8') as f:
                json.dump(asdict(log), f, indent=2, cls=FEJSONEncoder)

        return FEResult(
            transformed_df=transformed_df, decision_log=log, column_profiles=profiles,
            pipeline_artifact=pipeline, output_dir=output_dir, run_id=run_id,
            recommended_interactions={"numeric": recommended_num, "polynomial": recommended_poly, "ranking": ranking_initial}
        )
