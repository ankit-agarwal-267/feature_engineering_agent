import pandas as pd
import polars as pl
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from fe_agent.profiler.semantic_types import ColumnProfile, RawDataFrame, SemanticType
from fe_agent.decisions.decision_log import DecisionLog, DecisionRecord

class AuditReporter:
    def __init__(self, run_id: str, output_dir: Path):
        self.run_id = run_id
        self.output_dir = output_dir

    def _format_metric(self, val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    def generate_report(self, log: DecisionLog, profiles: List[ColumnProfile], transformed_df: RawDataFrame) -> Path:
        """
        Produces the human-readable Markdown audit report.
        """
        report_path = self.output_dir / f"audit_report_{self.run_id}.md"
        
        content = []
        content.append(f"# Feature Engineering Audit Report")
        content.append(f"**Run ID:** {self.run_id}  **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Dataset:** {log.target_column} ({log.dataset_shape[0]} rows, {log.dataset_shape[1]} columns)")
        
        content.append("\n## 1. Dataset Summary")
        content.append(f"- Input shape: {log.dataset_shape[0]} rows x {len(profiles)} cols")
        content.append(f"- Output shape: {log.dataset_shape[0]} rows x {transformed_df.shape[1]} cols")
        content.append(f"- Task type: {log.task_type}")

        # Section 2: Predictive Power Metrics
        if log.llm_advisory and "feature_importance_ranking" in log.llm_advisory:
            content.append("\n## 2. Predictive Power Metrics")
            content.append("| Feature | MI | ANOVA | Cramer's V | IV | Corr |")
            content.append("|---|---|---|---|---|---|")
            ranking = log.llm_advisory["feature_importance_ranking"]
            # Sort by MI
            sorted_feats = sorted(ranking.items(), key=lambda x: x[1].get('mi', 0), reverse=True)
            for feat, metrics in sorted_feats:
                content.append(f"| {feat} | {self._format_metric(metrics.get('mi'))} | {self._format_metric(metrics.get('anova'))} | {self._format_metric(metrics.get('cramer'))} | {self._format_metric(metrics.get('iv'))} | {self._format_metric(metrics.get('corr'))} |")


        # Section 3: LLM Advisor Reasoning
        if log.llm_advisory and "llm_review" in log.llm_advisory:
            review = log.llm_advisory["llm_review"]
            if review and "error" not in str(review):
                content.append("\n## 3. LLM Advisor Recommendations")
                if isinstance(review, dict):
                    improvements = review.get("improvements", []) or review.get("additional_transforms", [])
                    if improvements:
                        for imp in improvements:
                            content.append(f"- **{imp.get('col', 'Global')}**: {imp.get('transform', '')}")
                            content.append(f"  - Rationale: {imp.get('rationale', 'N/A')}")
                    else:
                        content.append(f"```json\n{json.dumps(review, indent=2)}\n```")
                else:
                    content.append(f"> {review}")

        # Section 4: Feature Selection Rationale (New)
        if log.llm_advisory and "selection_rationale" in log.llm_advisory:
            content.append("\n## 4. Feature Selection Rationale")
            content.append("| Feature | Status | Reasoning / Insight |")
            content.append("|---|---|---|")
            rationale_map = log.llm_advisory["selection_rationale"]
            for feat, data in rationale_map.items():
                content.append(f"| {feat} | {data.get('status')} | {data.get('rationale')} |")

        # Section 5: Data Quality & Health
        content.append("\n## 5. Data Quality & Health")
        quasi_constants = [p.name for p in profiles if p.semantic_type == SemanticType.CONSTANT]
        if quasi_constants:
            content.append(f"- **Dropped (Constant/Quasi-constant):** {', '.join(quasi_constants)}")
        else:
            content.append("- No constant or quasi-constant columns detected.")
        
        leakage = log.leakage_warnings
        if leakage:
            content.append("\n### Leakage Warnings")
            for w in leakage:
                content.append(f"- ⚠️ {w}")

        # Section 5: Feature Engineering Decisions
        content.append("\n## 5. Feature Engineering Decisions")
        col_decisions: Dict[str, List[DecisionRecord]] = {}
        for d in log.decisions:
            if d.column_name not in col_decisions:
                col_decisions[d.column_name] = []
            col_decisions[d.column_name].append(d)
            
        for col_name, decisions in col_decisions.items():
            if any("interaction" in d.transform_name or d.rule_triggered == "USER_APPROVED" for d in decisions):
                continue
            content.append(f"\n### {col_name}")
            for d in decisions:
                status = "✅" if d.decision == "accepted" else "❌"
                content.append(f"- {status} `{d.transform_name}` — {d.rationale}")
                if d.output_columns:
                    content.append(f"  - Produced: `{', '.join(d.output_columns)}`")

        # Section 6: Cross-Feature Interactions
        interactions = [d for d in log.decisions if "interaction" in d.transform_name or d.rule_triggered == "USER_APPROVED"]
        if interactions:
            content.append("\n## 6. Cross-Feature Interactions & Polynomials")
            content.append("The following interactions were generated based on predictive ranking and user approval:")
            for i in interactions:
                content.append(f"- **{i.column_name}**: `{i.transform_name}` -> `{', '.join(i.output_columns)}`")

        # Section 7: Data Preview
        content.append("\n## 7. Transformed Data Preview (Top 5)")
        if isinstance(transformed_df, pd.DataFrame):
            preview = transformed_df.head(5).to_markdown()
        else:
            preview = transformed_df.head(5).to_pandas().to_markdown()
        content.append(preview)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
            
        return report_path
