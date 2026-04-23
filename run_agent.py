import argparse
import json
import sys
import traceback
import pandas as pd
from fe_agent import FeatureEngineeringAgent, FEConfig, LLMConfig
from fe_agent.ask_user import ask_user

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Agent")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--interactions", action="store_true", help="Prompt for interactions")
    parser.add_argument("--polynomials", action="store_true", help="Prompt for polynomials")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--output_dir", default="./fe_output")
    args = parser.parse_args()

    # 1. Analysis Stage
    print("Running initial analysis...")
    config_analysis = FEConfig(target_column=args.target, interaction_top_n=5)
    agent = FeatureEngineeringAgent(config=config_analysis)
    result = agent.run(args.source, skip_io=True)
    
    # 2. HITL Stage
    selected_ints = []
    selected_polys = []
    
    if args.interactions or args.polynomials:
        questions = []
        if args.interactions:
            pairs = result.recommended_interactions.get("numeric", [])
            questions.append({"header": "Interactions", "type": "choice", "multiSelect": True,
                "options": [{"label": f"{a} x {b}", "description": "Numeric interaction"} for a, b in pairs],
                "question": "Select interaction pairs:"})
        if args.polynomials:
            poly_options = []
            for feat in result.recommended_interactions.get("polynomial", [])[:5]:
                col = feat['col']
                for deg in [1, 2, 3]: poly_options.append({"label": f"{col} deg {deg}", "description": "Polynomial"})
            questions.append({"header": "Polynomials", "type": "choice", "multiSelect": True,
                "options": poly_options, "question": "Select polynomials:"})
        
        user_choices = ask_user(questions=questions)
        
        # Parse choices
        if args.interactions and len(user_choices) > 0:
            selected_ints = [opt.get('label').split(' x ') for opt in user_choices[0]['value']]
        
        poly_idx = 1 if args.interactions else 0
        if args.polynomials and len(user_choices) > poly_idx:
            for opt in user_choices[poly_idx]['value']:
                col, _, deg = opt.get('label').split(' ')
                selected_polys.append({'col': col, 'degree': int(deg)})

    # 3. Final Execution
    print("Running transformation...")
    final_config = FEConfig(
        target_column=args.target,
        output_dir=args.output_dir, # Ensure this is used
        selected_numeric_interactions=selected_ints,
        selected_polynomial_features=selected_polys,
        llm=LLMConfig(enabled=args.llm, provider="ollama", model="mistral:7b")
    )
    final_agent = FeatureEngineeringAgent(config=final_config)
    final_agent.run(args.source)
    print(f"Done! Reports in {final_config.output_dir}")

if __name__ == "__main__":
    main()
