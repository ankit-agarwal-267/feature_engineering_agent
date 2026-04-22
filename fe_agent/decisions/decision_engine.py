from typing import List, Literal, Optional, Dict, Any
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType
from fe_agent.decisions.decision_log import DecisionRecord, DecisionLog
from fe_agent.config.config_schema import FEConfig

class DecisionEngine:
    def __init__(self, config: FEConfig):
        self.config = config

    def decide(self, profile: ColumnProfile) -> List[DecisionRecord]:
        """
        Evaluates top-down rules to decide which transforms to apply (Section 8).
        """
        records = []
        
        # RULE 000: USER_OVERRIDE
        if profile.drop:
            records.append(self._create_record(profile, "drop", [], "dropped", "RULE 000", "User-requested drop"))
            return records
            
        # RULE 001: DROP_CONSTANT
        if profile.semantic_type == SemanticType.CONSTANT:
            records.append(self._create_record(profile, "drop", [], "dropped", "RULE 001", "Column has zero variance (constant)"))
            return records
            
        # RULE 002: DROP_ID
        if profile.semantic_type == SemanticType.ID_COLUMN:
            records.append(self._create_record(profile, "drop", [], "dropped", "RULE 002", "High cardinality ID column with no predictive value"))
            return records

        # Per-column transform rules (simplified)
        if profile.semantic_type == SemanticType.ORDINAL:
            records.append(self._create_record(profile, "ordinal_encoding", [f"{profile.name}_ordinal"], "accepted", "RULE 006", "Applying ordinal encoding based on detected/supplied order"))
            
        if profile.semantic_type == SemanticType.CATEGORICAL_LOW:
            if profile.n_unique <= self.config.ohe_max_cardinality:
                records.append(self._create_record(profile, "one_hot_encoding", ["multiple"], "accepted", "RULE 007", f"Applying one-hot encoding (cardinality={profile.n_unique})"))
            else:
                records.append(self._create_record(profile, "frequency_encoding", [f"{profile.name}_freq"], "accepted", "RULE 008", f"Applying frequency encoding (cardinality={profile.n_unique} > threshold)"))

        if profile.semantic_type == SemanticType.NUMERIC_CONTINUOUS:
            if profile.is_skewed and not profile.has_negative:
                if not profile.has_zero:
                    records.append(self._create_record(profile, "log1p", [f"{profile.name}_log1p"], "accepted", "RULE 010", f"Applying log1p transform (skewness={profile.skewness:.2f})"))
                else:
                    records.append(self._create_record(profile, "sqrt", [f"{profile.name}_sqrt"], "accepted", "RULE 011", f"Applying sqrt transform for skewed data with zeros (skewness={profile.skewness:.2f})"))
            
            if profile.null_pct < 0.10 and profile.n_unique > 10:
                records.append(self._create_record(profile, "polynomial", [f"{profile.name}_sq"], "accepted", "RULE 013", f"Adding polynomial feature (degree={self.config.polynomial_degree})"))

        if profile.semantic_type == SemanticType.DATETIME:
            records.append(self._create_record(profile, "datetime_extraction", ["multiple"], "accepted", "RULE 015", "Extracting year, month, day, and cyclical temporal features"))

        if profile.semantic_type == SemanticType.BOOLEAN:
            records.append(self._create_record(profile, "boolean_cast", [profile.name], "accepted", "RULE 016", "Casting to int8 (0/1)"))

        if profile.semantic_type == SemanticType.TEXT:
            records.append(self._create_record(profile, "text_fe", ["multiple"], "accepted", "RULE 017", "Extracting text-based numeric features (char count, word count, etc.)"))

        return records

    def _create_record(self, profile: ColumnProfile, transform: str, output_cols: List[str], 
                       decision: str, rule: str, rationale: str) -> DecisionRecord:
        return DecisionRecord(
            column_name=profile.name,
            transform_name=transform,
            output_columns=output_cols,
            decision=decision, # type: ignore
            rule_triggered=rule,
            rationale=rationale,
            data_evidence={
                "skewness": profile.skewness,
                "cardinality": profile.n_unique,
                "null_pct": profile.null_pct,
                "semantic_type": profile.semantic_type
            }
        )
