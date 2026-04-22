import argparse
import json
import sys
import traceback
import pandas as pd
from fe_agent import FeatureEngineeringAgent, FEConfig, LLMConfig

def main():
    parser = argparse.ArgumentParser(description="Dynamic FE Agent Driver")
    parser.add_argument("--source", required=True, help="Path to data file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--interactions", help="JSON string of selected interactions")
    parser.add_argument("--polynomials", help="JSON string of selected polynomials")
    parser.add_argument("--analyze_only", action="store_true", help="Only output recommendations")
    parser.add_argument("--llm", action="store_true", help="Enable LLM reasoning")
    parser.add_argument("--output_dir", default="./fe_output", help="Output directory")
    
    args = parser.parse_args()
    
    selected_interactions = json.loads(args.interactions) if args.interactions else []
    selected_polynomials = json.loads(args.polynomials) if args.polynomials else []

    config = FEConfig(
        target_column=args.target,
        output_dir=args.output_dir,
        selected_numeric_interactions=selected_interactions,
        selected_polynomial_features=selected_polynomials,
        numeric_interactions=True,
        llm=LLMConfig(enabled=args.llm, provider="ollama", model="mistral:7b")
    )
    
    agent = FeatureEngineeringAgent(config=config)
    
    try:
        result = agent.run(source=args.source)
        
        if args.analyze_only:
            # Output ONLY the recommendations for the orchestrator to read
            print("---METRICS_START---")
            print(json.dumps(result.decision_log.llm_advisory.get("feature_importance_ranking", {})))
            print("---METRICS_END---")
            print("---RECOMMENDATIONS_START---")
            print(json.dumps(result.recommended_interactions.get("numeric", [])))
            print("---RECOMMENDATIONS_END---")
        else:
            print(f"Run Complete. ID: {result.run_id}")
            print(f"Final Columns: {len(result.transformed_df.columns)}")
            
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
