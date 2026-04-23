import argparse
import json
import sys
import traceback
from pathlib import Path
import pandas as pd
from fe_agent import FeatureEngineeringAgent, FEConfig, LLMConfig
from fe_agent.ask_user import ask_user
from fe_agent.decisions.decision_optimizer import DecisionOptimizer
from fe_agent.profiler.semantic_types import SemanticType

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Agent")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--polynomials", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--output_dir", default="./fe_output")
    args = parser.parse_args()

    # PHASE 1: Initial Analysis
    print("PHASE 1: Profiling and Recommendation...")
    agent = FeatureEngineeringAgent(config=FEConfig(target_column=args.target))
    analysis = agent.run(args.source, skip_io=True)
    
    # PHASE 2: Selection
    print("PHASE 2: HITL Feature Construction...")
    selected_ints, selected_polys = [], []
    
    if args.interactions:
        all_pairs = analysis.recommended_interactions.get("numeric", [])
        user_choice = ask_user([{
            "header": "Interactions", "type": "choice", "multiSelect": True,
            "options": [{"label": f"{a} x {b}", "description": "Interaction"} for a, b in all_pairs],
            "question": f"Select from {len(all_pairs)} interactions (A=All, N=None):"
        }])
        if user_choice and user_choice[0].get('value'):
            selected_ints = [opt['label'].split(' x ') for opt in user_choice[0]['value']]

    if args.polynomials:
        all_poly = analysis.recommended_interactions.get("polynomial", [])
        poly_opts = []
        for pf in all_poly:
            for deg in [1, 2, 3]: poly_opts.append({"label": f"{pf['col']} deg {deg}", "description": "Polynomial"})
        
        user_choice = ask_user([{
            "header": "Polynomials", "type": "choice", "multiSelect": True,
            "options": poly_opts,
            "question": f"Select polynomials (A=All, N=None):"
        }])
        if user_choice and user_choice[0].get('value'):
            for opt in user_choice[0]['value']:
                parts = opt['label'].split(' ')
                selected_polys.append({'col': parts[0], 'degree': int(parts[2])})

    # PHASE 3: Transformation and Final Review
    print("PHASE 3: Transformation and Final Review...")
    final_config = FEConfig(
        target_column=args.target, output_dir=args.output_dir,
        selected_interactions=selected_ints,
        selected_polynomial_features=selected_polys,
        llm=LLMConfig(enabled=args.llm)
    )
    final_agent = FeatureEngineeringAgent(config=final_config)
    final_result = final_agent.run(args.source)

    # 4. Final Optimization (Unified Rationale)
    selection_rationale = final_result.decision_log.llm_advisory["selection_rationale"]

    # C. Final HITL Prompt
    print(f"\nFinal Feature Set: {len(final_result.transformed_df.columns)} features.")
    pruning_options = []
    for f, data in selection_rationale.items():
        pruning_options.append({
            "label": f"{f} ({data['status']})",
            "description": data['rationale']
        })

    user_final = ask_user([{
        "header": "Final Pruning", "type": "choice", "multiSelect": True,
        "options": pruning_options,
        "question": "Select features to DROP (U=All Useless, W=Useless+Weak, N=Keep All):"
    }])
    
    if user_final and user_final[0].get('value'):
        to_drop = [opt['label'].split(' (')[0] for opt in user_final[0]['value']]
        final_df = final_result.transformed_df.drop(columns=[c for c in to_drop if c in final_result.transformed_df.columns])
        ext = Path(args.source).suffix.lower() if isinstance(args.source, str) else '.csv'
        final_path = Path(args.output_dir) / f"final_dataset_{final_result.run_id}{ext}"
        if ext == '.parquet': final_df.to_parquet(final_path, index=False)
        else: final_df.to_csv(final_path, index=False)
        print(f"Pruning complete. Final column count: {len(final_df.columns)}")
        print(f"Final file: {final_path}")

    print(f"Process complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()
