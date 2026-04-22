from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Union, TypeAlias
import pandas as pd
import polars as pl

RawDataFrame: TypeAlias = Union[pd.DataFrame, pl.DataFrame]

class SemanticType(str, Enum):
    NUMERIC_CONTINUOUS   = "numeric_continuous"
    NUMERIC_DISCRETE     = "numeric_discrete"
    CATEGORICAL_LOW      = "categorical_low"      # cardinality <= threshold (default 20)
    CATEGORICAL_HIGH     = "categorical_high"     # cardinality > threshold
    ORDINAL              = "ordinal"              # numeric but with clear discrete ordering
    BOOLEAN              = "boolean"
    DATETIME             = "datetime"
    TEXT                 = "text"                 # free-form string, avg token count > 5
    ID_COLUMN            = "id_column"            # unique or near-unique, no predictive value
    CONSTANT             = "constant"             # single unique value
    TARGET               = "target"               # reserved; assigned externally
    UNKNOWN              = "unknown"

@dataclass
class ColumnProfile:
    name: str
    raw_dtype: str
    semantic_type: SemanticType
    n_unique: int
    null_count: int
    null_pct: float
    sample_values: list[Any]          # up to 10 representative values
    is_skewed: bool                   # |skewness| > 1.0 for numeric columns
    skewness: float | None = None
    has_negative: bool = False
    has_zero: bool = False
    cardinality_ratio: float = 0.0          # n_unique / n_rows
    detected_order: list[Any] | None = None  # for ordinal columns
    datetime_format: str | None = None
    avg_token_count: float | None = None     # for text columns
    notes: list[str] = field(default_factory=list) # human-readable profiler observations
    
    # Override tracking (Section 6.5.2)
    inferred_semantic_type: SemanticType | None = None
    override_applied: bool = False
    override_source: str | None = None
    skip_transforms: list[str] = field(default_factory=list)
    force_transforms: list[str] = field(default_factory=list)
    drop: bool = False
