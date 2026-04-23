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
from fe_agent.engine.transforms.numeric import NumericTransformer

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
    recommended_interactions: Dict[str, Any] = field(default_factory=dict)

class FEJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, SemanticType): return str(obj)
        return super().default(obj)

class FeatureEngineeringAgent:
    def __init__(self, config: FEConfig, custom_transforms: Optional[List[Any]] = None):
        self.config = config
        self.custom_transforms = custom_transforms or []
        self.loaders = [CSVLoader(), ParquetLoader(), JSONLoader(), SQLLoader(), DictLoader()]
        self.numeric_transformer = NumericTransformer(config)
        self.interaction_transformer = InteractionTransformer(config)
        self.scorer = InformationValueScorer()
        self.leakage_guard = LeakageGuard(config)
        self.llm_advisor = LLMAdvisor(config)
        
    def _generate_run_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def run(self, source: Union[str, Dict[str, Any], RawDataFrame], train_mask: Optional[Any] = None, skip_io: bool = False) -> FEResult:
        run_id = self.config.run_id or self._generate_run_id()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Ingestion
        df = None
        if isinstance(source, (pd.DataFrame, pl.DataFrame)): df = source
        else:
            for loader in self.loaders:
                if loader.supports(source):
                    df = loader.load(source, backend=self.config.dataframe_backend)
                    break
        if df is None: raise ValueError(f"No loader supports source: {source}")

        # 2. Profiling & Overrides
        profiler = SchemaProfiler(self.config)
        profiles = profiler.profile(df)
        profiles = OverrideResolver(self.config).resolve(profiles)
        
        # 3. Decisions & Initial Metrics
        decision_engine = DecisionEngine(self.config)
        all_decisions = [d for p in profiles if p.name != self.config.target_column for d in decision_engine.decide(p)]
        ranking_initial = self.scorer.score(df, self.config.target_column, profiles)
        
        # 4. Interaction Discovery (Everything except DATETIME)
        valid_feats = [p.name for p in profiles if p.name != self.config.target_column and p.semantic_type != SemanticType.DATETIME]
        recommended_pairs = []
        for i in range(len(valid_feats)):
            for j in range(i+1, len(valid_feats)):
                recommended_pairs.append((valid_feats[i], valid_feats[j]))
        
        recommended_poly = [{"col": p.name} for p in profiles if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)]

        # 5. Apply User Selections (Interactions & Polynomials)
        new_cols_list = []
        
        # Apply Interactions
        for pair in self.config.selected_interactions:
            col_a, col_b = pair[0], pair[1]
            p_a = next(p for p in profiles if p.name == col_a)
            p_b = next(p for p in profiles if p.name == col_b)
            is_num_a = p_a.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)
            is_num_b = p_b.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)
            
            if is_num_a and is_num_b:
                res_df = self.interaction_transformer.generate_numeric_interactions(df, [(col_a, col_b)])
            elif not is_num_a and not is_num_b:
                res_df = self.interaction_transformer.generate_categorical_interactions(df, [(col_a, col_b)])
            else:
                cat, num = (col_a, col_b) if not is_num_a else (col_b, col_a)
                res_df = self.interaction_transformer.generate_group_stats(df, cat, num, ["mean", "std"])
            
            new_cols_list.append(res_df)
            all_decisions.append(DecisionRecord(f"{col_a}, {col_b}", "interaction", list(res_df.columns), "accepted", "USER", "User selected interaction."))

        # Apply Polynomials
        for item in self.config.selected_polynomial_features:
            col, deg = item['col'], item['degree']
            p = next(p for p in profiles if p.name == col)
            res_val = self.numeric_transformer.apply_polynomial(df[col], p, deg)
            if res_val is not None:
                name = f"{col}_pow_{deg}"
                poly_df = pd.DataFrame({name: res_val}) if isinstance(df, pd.DataFrame) else pl.DataFrame({name: res_val})
                new_cols_list.append(poly_df)
                all_decisions.append(DecisionRecord(col, f"polynomial_d{deg}", [name], "accepted", "USER", f"User selected degree {deg}."))

        # Single Concatenation to prevent fragmentation
        if new_cols_list:
            if isinstance(df, pd.DataFrame): df = pd.concat([df] + new_cols_list, axis=1)
            else: df = pl.concat([df] + new_cols_list, how="horizontal")

        # 6. Transformation (Standard Rules)
        profiles = profiler.profile(df)
        transformed_df, pipeline = FEEngine(self.config).transform(df, profiles, all_decisions, train_mask)

        # 7. Final Assembly & Export
        final_profiles = profiler.profile(transformed_df)
        ranking_final = self.scorer.score(transformed_df, self.config.target_column, final_profiles)
        leakage_warnings = self.leakage_guard.check_leakage(transformed_df, self.config.target_column)
        llm_advisory = self.llm_advisor.review_decisions(profiles, all_decisions) if self.config.llm.enabled else None
        llm_pruning = self.llm_advisor.get_pruning_advice(ranking_final) if self.config.llm.enabled else None
        
        from fe_agent.decisions.decision_optimizer import DecisionOptimizer
        selection_rationale = DecisionOptimizer(self.config).get_baseline_selection(ranking_final)
        
        # Merge LLM Pruning Feedback
        if llm_pruning and isinstance(llm_pruning, dict) and "feedback" in llm_pruning:
            for item in llm_pruning["feedback"]:
                feat = item.get('feature')
                if feat in selection_rationale:
                    reason = item.get('reason', 'LLM Insight')
                    action = item.get('action', 'keep').lower()
                    current_rationale = selection_rationale[feat]['rationale']
                    selection_rationale[feat]['rationale'] = f"[LLM] {reason} | {current_rationale}"
                    if action == 'drop': selection_rationale[feat]['status'] = "drop_llm"

        log = DecisionLog(run_id, datetime.now().isoformat(), hashlib.sha256(json.dumps(self.config.model_dump(), sort_keys=True).encode()).hexdigest(),
                          list(transformed_df.shape), self.config.target_column, self.config.task_type, all_decisions, leakage_warnings, 
                          {"feature_importance_ranking": ranking_final, "llm_review": llm_advisory, "llm_pruning": llm_pruning, "selection_rationale": selection_rationale})
        
        if not skip_io:
            if self.config.write_audit_report: AuditReporter(run_id, output_dir).generate_report(log, profiles, transformed_df)
            if self.config.write_decision_log:
                with open(output_dir / f"decision_log_{run_id}.json", 'w', encoding='utf-8') as f: json.dump(asdict(log), f, indent=2, cls=FEJSONEncoder)
            
            ext = Path(source).suffix.lower() if isinstance(source, str) else '.csv'
            out_path = output_dir / f"transformed_data_{run_id}{ext}"
            if isinstance(transformed_df, pd.DataFrame):
                if ext == '.parquet': transformed_df.to_parquet(out_path, index=False)
                else: transformed_df.to_csv(out_path, index=False)
            else:
                if ext == '.parquet': transformed_df.write_parquet(out_path)
                else: transformed_df.write_csv(out_path)

        return FEResult(transformed_df, log, profiles, pipeline, output_dir, run_id, recommended_interactions={"numeric": recommended_pairs, "polynomial": recommended_poly})
