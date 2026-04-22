from dataclasses import dataclass, field
from typing import Any, List, Optional, Literal, Dict
from datetime import datetime

@dataclass
class DecisionRecord:
    column_name: str                  # source column
    transform_name: str               # e.g. "log1p", "one_hot_encoding"
    output_columns: List[str]         # new column name(s)
    decision: Literal["accepted", "rejected", "skipped", "warned"]
    rule_triggered: str               # which rule caused the decision
    rationale: str                    # human-readable explanation
    data_evidence: Dict[str, Any] = field(default_factory=dict)     # e.g. {"skewness": 2.3, "cardinality": 15}
    llm_reviewed: bool = False
    llm_suggestion: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class DecisionLog:
    run_id: str
    timestamp: str
    config_hash: str
    dataset_shape: List[int]
    target_column: str
    task_type: str
    decisions: List[DecisionRecord] = field(default_factory=list)
    leakage_warnings: List[str] = field(default_factory=list)
    llm_advisory: Optional[Dict[str, Any]] = None
