from typing import List, Dict, Any, Optional

class DecisionOptimizer:
    """
    Categorizes features based on predictive metrics and provides a baseline drop list with rationale.
    """
    def __init__(self, config: Any):
        self.config = config

    def get_baseline_selection(self, ranking: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, str]]:
        """
        Categorize features and provide rationale.
        Returns: {feat_name: {"status": "drop_useless", "rationale": "..."}}
        """
        suggestions = {}
        for feat, metrics in ranking.items():
            mi = metrics.get('mi', 0)
            anova = metrics.get('anova', 0)
            cramer = metrics.get('cramer', 0)
            score = max(mi, anova, cramer)

            if score < 0.01:
                suggestions[feat] = {
                    "status": "drop_useless",
                    "rationale": f"[Rule] Max predictive score ({score:.4f}) is below 'useless' threshold."
                }
            elif score < 0.05:
                suggestions[feat] = {
                    "status": "keep_weak",
                    "rationale": f"[Rule] Predictive score ({score:.4f}) is weak. Low signal suspected."
                }
            else:
                suggestions[feat] = {
                    "status": "keep_strong",
                    "rationale": f"[Rule] Strong predictive signal detected ({score:.4f})."
                }
        return suggestions
