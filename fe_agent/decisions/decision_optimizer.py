from typing import List, Dict, Any, Optional
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType

class DecisionOptimizer:
    """
    Categorizes features based on predictive metrics and provides a baseline drop list with rationale.
    """
    def __init__(self, config: Any):
        self.config = config

    def get_baseline_selection(self, ranking: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, str]]:
        # ... (existing baseline logic) ...
        suggestions = {}
        for feat, metrics in ranking.items():
            mi, anova, cramer = metrics.get('mi', 0), metrics.get('anova', 0), metrics.get('cramer', 0)
            score = max(mi, anova, cramer)
            if score < 0.01: status, reason = "drop_useless", f"[Rule] Max predictive score ({score:.4f}) is low."
            elif score < 0.05: status, reason = "keep_weak", f"[Rule] Predictive score ({score:.4f}) is weak."
            else: status, reason = "keep_strong", f"[Rule] Strong predictive signal detected ({score:.4f})."
            suggestions[feat] = {"status": status, "rationale": reason}
        return suggestions

    def get_construction_recommendations(self, ranking: Dict[str, Dict[str, float]], profiles: List[ColumnProfile]) -> Dict[str, Any]:
        """
        Implements Dynamic Priority Logic for recommendations.
        1. Keep all Strong/Medium/Suspicious.
        2. Fallback to top 50% if signal is low.
        """
        scored_feats = []
        for feat, metrics in ranking.items():
            score = max(metrics.get('mi', 0), metrics.get('anova', 0), metrics.get('cramer', 0))
            scored_feats.append((feat, score))
        
        scored_feats.sort(key=lambda x: x[1], reverse=True)
        
        # Priority Set: Strong/Medium (> 0.05)
        priority_set = [f for f, s in scored_feats if s >= 0.05]
        
        # 50% Floor calculation
        min_count = len(scored_feats) // 2
        
        if len(priority_set) >= min_count:
            # We have enough high signal, use only priority
            active_set = priority_set
        else:
            # Low signal overall, top up to 50% floor
            active_set = [f for f, s in scored_feats[:min_count]]

        # Suggested Interactions
        suggested_ints = []
        for i in range(len(active_set)):
            for j in range(i+1, len(active_set)):
                suggested_ints.append((active_set[i], active_set[j]))
        
        # Suggested Polynomials
        suggested_polys = []
        for feat in active_set:
            prof = next((p for p in profiles if p.name == feat), None)
            if prof and prof.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE):
                suggested_polys.append(feat)
                
        return {"interactions": suggested_ints, "polynomials": suggested_polys}
