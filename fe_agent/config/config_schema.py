from enum import Enum
from typing import Any, List, Dict, Optional, Literal, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fe_agent.profiler.semantic_types import SemanticType

class ColumnOverride(BaseSettings):
    """
    Override settings for a specific column.
    """
    semantic_type: Optional[SemanticType] = Field(None, description="Force this semantic type")
    ordinal_order: Optional[List[Any]] = Field(None, description="Explicit category rank order, lowest to highest. Required if semantic_type: ordinal")
    boolean_true_values: Optional[List[str]] = Field(None, description="String values to map to 1")
    boolean_false_values: Optional[List[str]] = Field(None, description="String values to map to 0")
    datetime_format: Optional[str] = Field(None, description="strptime-compatible format string")
    skip_transforms: List[str] = Field(default_factory=list, description="Transform names to suppress (e.g. 'binning', 'log1p', 'polynomial', 'ohe')")
    force_transforms: List[str] = Field(default_factory=list, description="Transform names to force even if rule conditions not met")
    drop: bool = Field(False, description="Unconditionally drop this column before any profiling or FE")

    @field_validator("ordinal_order")
    @classmethod
    def check_ordinal_order(cls, v: Any, info: Any) -> Any:
        if info.data.get("semantic_type") == SemanticType.ORDINAL and v is None:
            raise ValueError("ordinal_order is required when semantic_type is 'ordinal'")
        return v

class LLMConfig(BaseSettings):
    """
    Configuration for the LLM Advisor.
    """
    enabled: bool = Field(False, description="Whether to enable the LLM advisor layer")
    provider: Literal["ollama", "openai_compat", "anthropic"] = Field("ollama", description="LLM provider: 'ollama', 'openai_compat', or 'anthropic'")
    model: str = Field("mistral:7b", description="Model name to use")
    base_url: str = Field("http://localhost:11434", description="Base URL for the LLM provider API")
    timeout_seconds: int = Field(120, description="Timeout for LLM API calls in seconds")
    max_retries: int = Field(2, description="Maximum number of retries for LLM API calls")
    temperature: float = Field(0.2, description="Sampling temperature for the LLM")
    llm_auto_apply: bool = Field(False, description="Whether to automatically apply LLM-suggested transforms")
    web_search: bool = Field(False, description="Whether to enable web search for the LLM advisor")
    web_search_backend: Literal["duckduckgo", "brave", "serper"] = Field("duckduckgo", description="Web search backend to use")
    max_web_searches: int = Field(3, description="Maximum number of web searches per run")

class FEConfig(BaseSettings):
    """
    Global configuration for the Feature Engineering Agent.
    """
    model_config = SettingsConfigDict(env_prefix="FE_AGENT_", env_nested_delimiter="__", extra="ignore")

    # Core
    target_column: str = Field(..., description="Target column name [required]")
    task_type: Literal["binary_classification", "multiclass_classification"] = Field("binary_classification", description="Classification task type")
    dataframe_backend: Literal["pandas", "polars"] = Field("pandas", description="Dataframe backend: 'pandas' or 'polars'")
    random_seed: int = Field(42, description="Seed for reproducibility")
    output_dir: str = Field("./fe_output", description="Output directory for all generated artifacts")
    run_id: Optional[str] = Field(None, description="Optional unique run identifier; auto-generated if null")

    # Data Ingestion
    sql_query: Optional[str] = Field(None, description="SQL query string to fetch data")
    sql_table: Optional[str] = Field(None, description="Table name to fetch data from (if SOURCE is a DSN)")
    csv_delimiter: Optional[str] = Field(None, description="CSV delimiter character; auto-detected if null")
    parquet_engine: Literal["pyarrow", "fastparquet"] = Field("pyarrow", description="Parquet engine for pandas")

    # Profiler
    high_cardinality_threshold: int = Field(20, description="Cardinality > this → CATEGORICAL_HIGH")
    id_cardinality_ratio: float = Field(0.95, description="n_unique / n_rows > this and dtype is int/string → ID_COLUMN")
    quasi_constant_threshold: float = Field(0.98, description="One value covers > this fraction of rows → QUASI_CONSTANT")

    # Numeric Transforms
    apply_log1p: bool = Field(True, description="Apply log1p transform to skewed numeric columns")
    apply_sqrt: bool = Field(True, description="Apply sqrt transform to skewed numeric columns")
    apply_boxcox: bool = Field(False, description="Apply Box-Cox transform (requires scipy)")
    polynomial_degree: int = Field(2, description="Degree for polynomial features (2 or 3)")
    apply_polynomial: bool = Field(True, description="Apply polynomial feature generation")
    apply_binning: bool = Field(True, description="Apply binning/discretisation")
    binning_strategy: Literal["quantile", "equal_width", "both"] = Field("quantile", description="Binning strategy")
    n_bins: int = Field(5, description="Number of bins for discretisation")
    equal_width_bins: bool = Field(False, description="Apply equal-width binning in addition to quantile")
    ratio_pairs: List[List[str]] = Field(default_factory=list, description="Pairs of columns for ratio features: [[col_a, col_b], ...]")
    log_or_sqrt: Literal["log", "sqrt", "both"] = Field("both", description="Selection when both log and sqrt are applicable")

    # Categorical Transforms
    ohe_max_cardinality: int = Field(15, description="Max cardinality for one-hot encoding")
    drop_first_ohe: bool = Field(False, description="Whether to use drop_first=True for OHE")
    use_target_encoding: bool = Field(False, description="Whether to use target encoding")
    allow_target_encoding_without_split: bool = Field(False, description="Allow target encoding without train/test split (not recommended)")
    target_encoding_folds: int = Field(5, description="Number of folds for k-fold target encoding")
    count_encoding: bool = Field(False, description="Whether to use count encoding")
    binary_encoding: bool = Field(False, description="Whether to use binary encoding")
    rare_threshold: float = Field(0.01, description="Threshold fraction for rare category grouping")

    # DateTime Transforms
    drop_source_datetime: bool = Field(False, description="Whether to drop original datetime columns after extraction")
    reference_date: Optional[str] = Field(None, description="ISO 8601 reference date for 'days_since' extraction")
    extract_is_month_start: bool = Field(False, description="Extract is_month_start flag")
    extract_is_month_end: bool = Field(False, description="Extract is_month_end flag")

    # Text Transforms
    keep_source_text: bool = Field(False, description="Whether to keep original text columns after extraction")
    text_tfidf: bool = Field(False, description="Apply TF-IDF to text columns (requires scikit-learn)")
    tfidf_max_features: int = Field(50, description="Maximum number of TF-IDF features")

    # Interaction Features
    numeric_interactions: bool = Field(False, description="Generate numeric interaction features")
    cat_interactions: bool = Field(False, description="Generate categorical interaction features")
    interaction_top_n: int = Field(5, description="Number of top columns to use for interactions")
    selected_numeric_interactions: List[List[str]] = Field(default_factory=list, description="Explicitly selected numeric interaction pairs")
    selected_group_stats: List[Dict[str, Any]] = Field(default_factory=list, description="Explicitly selected group stats: [{'num': 'col', 'cat': 'col'}]")
    interaction_max_cardinality: int = Field(50, description="Max cardinality for categorical interactions")
    selected_polynomial_features: List[Dict[str, Any]] = Field(default_factory=list, description="Explicitly selected polynomial features: [{'col': 'name', 'degree': 2}]")
    group_stats: bool = Field(False, description="Generate group-level numeric aggregates")
    group_stats_agg: List[str] = Field(default_factory=list, description="Aggregate statistics to compute (mean, std, min, max, median)")

    # Leakage Guard
    leakage_correlation_threshold: float = Field(0.95, description="Pearson/point-biserial correlation > this flags potential leakage")

    # Schema Overrides
    strict_overrides: bool = Field(False, description="Raise error if an override key doesn't match any column")
    column_overrides: Dict[str, ColumnOverride] = Field(default_factory=dict, description="Per-column semantic type and transform overrides")

    # Pipeline & Output
    write_pipeline_json: bool = Field(True, description="Write pipeline artifact as JSON")
    write_pipeline_pkl: bool = Field(False, description="Write pipeline artifact as Pickle")
    sklearn_pipeline: bool = Field(False, description="Export pipeline in scikit-learn format")
    write_audit_report: bool = Field(True, description="Write human-readable audit report")
    write_decision_log: bool = Field(True, description="Write machine-readable decision log")
    write_generated_code: bool = Field(True, description="Write standalone Python replay script")

    # LLM Advisor
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM Advisor configuration")
