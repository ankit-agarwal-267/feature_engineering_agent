from enum import Enum
from typing import Any, List, Dict, Optional, Literal, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fe_agent.profiler.semantic_types import SemanticType

class ColumnOverride(BaseSettings):
    semantic_type: Optional[SemanticType] = Field(None)
    ordinal_order: Optional[List[Any]] = Field(None)
    boolean_true_values: Optional[List[str]] = Field(None)
    boolean_false_values: Optional[List[str]] = Field(None)
    datetime_format: Optional[str] = Field(None)
    skip_transforms: List[str] = Field(default_factory=list)
    force_transforms: List[str] = Field(default_factory=list)
    drop: bool = Field(False)

class LLMConfig(BaseSettings):
    enabled: bool = Field(False)
    provider: str = Field("ollama")
    model: str = Field("mistral:7b")
    base_url: str = Field("http://localhost:11434")
    timeout_seconds: int = Field(120)
    temperature: float = Field(0.2)

class FEConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FE_AGENT_", env_nested_delimiter="__", extra="ignore")

    target_column: str = Field(...)
    task_type: str = Field("binary_classification")
    dataframe_backend: str = Field("pandas")
    random_seed: int = Field(42)
    output_dir: str = Field("./fe_output")
    run_id: Optional[str] = Field(None)

    # Ingestion
    csv_delimiter: Optional[str] = Field(None)
    parquet_engine: str = Field("pyarrow")

    # Profiler
    high_cardinality_threshold: int = Field(20)
    id_cardinality_ratio: float = Field(0.95)
    quasi_constant_threshold: float = Field(0.98)

    # Transformations
    apply_log1p: bool = Field(True)
    apply_sqrt: bool = Field(True)
    apply_boxcox: bool = Field(False)
    polynomial_degree: int = Field(2)
    apply_binning: bool = Field(True)
    n_bins: int = Field(5)
    equal_width_bins: bool = Field(False)
    ratio_pairs: List[List[str]] = Field(default_factory=list)
    
    # Interactions (Generic)
    selected_interactions: List[List[str]] = Field(default_factory=list)
    selected_polynomial_features: List[Dict[str, Any]] = Field(default_factory=list)
    interaction_top_n: int = Field(10)
    
    # Categorical
    ohe_max_cardinality: int = Field(15)
    drop_first_ohe: bool = Field(False)
    use_target_encoding: bool = Field(False)
    rare_threshold: float = Field(0.01)

    # Flags
    drop_source_datetime: bool = Field(False)
    keep_source_text: bool = Field(False)
    leakage_correlation_threshold: float = Field(0.95)
    write_audit_report: bool = Field(True)
    write_decision_log: bool = Field(True)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    column_overrides: Dict[str, ColumnOverride] = Field(default_factory=dict)
