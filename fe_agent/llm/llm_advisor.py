import json
from typing import List, Dict, Any, Optional
from fe_agent.config.config_schema import FEConfig, LLMConfig
from fe_agent.profiler.semantic_types import ColumnProfile
from fe_agent.decisions.decision_log import DecisionRecord
from fe_agent.llm.base import OllamaProvider

class LLMAdvisor:
    def __init__(self, config: FEConfig):
        self.config = config
        self.provider = None
        if config.llm.enabled:
            if config.llm.provider == "ollama":
                self.provider = OllamaProvider(config.llm)
            # Add other providers as needed

    def review_decisions(self, profiles: List[ColumnProfile], decisions: List[DecisionRecord]) -> Optional[Dict[str, Any]]:
        """
        Section 9.4: Review decisions and suggest additions.
        """
        if not self.provider:
            return None

        # Prepare context for LLM
        dataset_summary = {
            "target": self.config.target_column,
            "task": self.config.task_type,
            "columns": [p.name for p in profiles]
        }
        
        decisions_summary = [
            {"col": d.column_name, "transform": d.transform_name, "rationale": d.rationale}
            for d in decisions if d.decision == "accepted"
        ]

        system_prompt = (
            "You are an expert Data Scientist. Review the feature engineering decisions "
            "and suggest improvements or domain-specific transforms. "
            "Respond ONLY in valid JSON format. If you provide a list of improvements, "
            "use the key 'improvements' containing objects with 'col', 'transform', and 'rationale'."
        )
        
        user_message = json.dumps({
            "dataset": dataset_summary,
            "decisions": decisions_summary
        }, indent=2)

        try:
            response = self.provider.chat(system_prompt, user_message)
            content = response.content
            
            # Robust JSON extraction (handles markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            return {
                "error": "LLM response parsing failed",
                "exception": str(e),
                "raw_response": getattr(response, 'content', 'No response') if 'response' in locals() else 'No response'
            }
