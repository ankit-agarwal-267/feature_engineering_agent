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

    def review_decisions(self, profiles: List[ColumnProfile], decisions: List[DecisionRecord]) -> Optional[Dict[str, Any]]:
        if not self.provider: return None
        system_prompt = (
            "You are an expert Data Scientist. Review the feature engineering decisions "
            "and suggest improvements. Respond ONLY in valid JSON with 'improvements' key."
        )
        user_msg = json.dumps({
            "target": self.config.target_column,
            "decisions": [{"col": d.column_name, "transform": d.transform_name} for d in decisions if d.decision == "accepted"]
        })
        return self._call_llm(system_prompt, user_msg)

    def get_pruning_advice(self, ranking: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Asks the LLM to review the final metrics and provide domain-specific insights.
        """
        if not self.provider: return None
        system_prompt = (
            "You are an expert Data Scientist. Review these feature metrics (Mutual Information, ANOVA, etc.). "
            "For the most important features, provide a brief domain-specific insight. "
            "For redundant or low-signal features, suggest dropping them with a reason. "
            "Respond ONLY in valid JSON with a 'feedback' key containing objects with 'feature', 'action' (keep/drop), and 'reason'."
        )
        # Limit to top 50 features to avoid context overflow
        top_metrics = dict(list(ranking.items())[:50])
        user_msg = json.dumps({"task": self.config.task_type, "metrics": top_metrics}, indent=2)
        return self._call_llm(system_prompt, user_msg)

    def _call_llm(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        try:
            response = self.provider.chat(system_prompt, user_message)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}
