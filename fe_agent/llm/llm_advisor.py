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
            "You are an expert Data Scientist. Review these feature engineering decisions. "
            "Suggest additional improvements. Respond ONLY in valid JSON with an 'improvements' key."
        )
        user_msg = json.dumps({
            "target": self.config.target_column,
            "decisions": [{"col": d.column_name, "transform": d.transform_name} for d in decisions if d.decision == "accepted"]
        })
        return self._call_llm(system_prompt, user_msg)

    def get_construction_advice(self, ranking: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        if not self.provider: return None
        system_prompt = (
            "You are an expert Data Scientist. Review these predictive metrics. "
            "Recommend the best 3-5 feature interactions (pairs) and 2-3 polynomial features. "
            "Respond ONLY in valid JSON with 'interactions' (list of pairs) and 'polynomials' (list of strings)."
        )
        # Simplify metrics for LLM
        top_metrics = {k: {"mi": v.get("mi", 0)} for k, v in list(ranking.items())[:15]}
        user_msg = json.dumps({"target": self.config.target_column, "metrics": top_metrics})
        return self._call_llm(system_prompt, user_msg)

    def get_pruning_advice(self, ranking: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        if not self.provider: return None
        system_prompt = (
            "You are an expert Data Scientist. Review these feature metrics. "
            "Identify features to drop due to redundancy or low signal. "
            "Respond ONLY in valid JSON with a 'feedback' key containing objects with 'feature', 'action' (keep/drop), and 'reason'."
        )
        top_metrics = {k: {"mi": v.get("mi", 0)} for k, v in list(ranking.items())[:20]}
        user_msg = json.dumps({"target": self.config.target_column, "metrics": top_metrics})
        return self._call_llm(system_prompt, user_msg)

    def _call_llm(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        try:
            response = self.provider.chat(system_prompt, user_message)
            content = response.content
            # Clean possible markdown noise
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find first { and last } to avoid conversational noise
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
                
            return json.loads(content)
        except Exception as e:
            return {"error": str(e), "raw": content if 'content' in locals() else ""}
