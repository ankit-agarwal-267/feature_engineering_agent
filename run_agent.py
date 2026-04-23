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

    # PHASE 0: Initial Analysis
    print("PHASE 0: Initial Analysis & Ranking...")
    base_config = FEConfig(
        target_column=args.target,
        llm=LLMConfig(enabled=args.llm, provider="ollama", model="mistral:7b")
    )
    agent_base = FeatureEngineeringAgent(config=base_config)
    analysis = agent_base.run(args.source, skip_io=True)
    
    # PHASE 1: Smart Construction Selection
    selected_ints, selected_polys = [], []
    
    if args.interactions or args.polynomials:
        print("PHASE 1: Smart Feature Construction Selection...")
        optimizer = DecisionOptimizer(None)
        ranking = analysis.decision_log.llm_advisory["feature_importance_ranking"]
        
        # A. Rule-Based Recommendations
        rule_recs = optimizer.get_construction_recommendations(ranking, analysis.column_profiles)
        
        # B. LLM Recommendations (Optional)
        llm_recs = None
        if args.llm:
            llm_recs = agent_base.llm_advisor.get_construction_advice(ranking)

        # C. Unify Recommendations for Prompt
        questions = []
        if args.interactions:
            # Mark recommendations
            rule_pairs = set([tuple(sorted(p)) for p in rule_recs["interactions"]])
            llm_pairs = set([tuple(sorted(p)) for p in llm_recs["interactions"]]) if llm_recs and "interactions" in llm_recs else set()
            
            # Full pool
            all_feats = [p.name for p in analysis.column_profiles if p.name != args.target and p.semantic_type != SemanticType.DATETIME]
            all_possible = [(all_feats[i], all_feats[j]) for i in range(len(all_feats)) for j in range(i+1, len(all_feats))]
            
            int_options = []
            for a, b in all_possible:
                pair = tuple(sorted((a,b)))
                is_rule = pair in rule_pairs
                is_llm = pair in llm_pairs
                
                prefix = ""
                if is_rule: prefix += "[Rule]"
                if is_llm: prefix += "[LLM]"
                
                label = f"{a} x {b}"
                if prefix: label = f"{prefix} {label}"
                int_options.append({"label": label, "description": "Interaction", "is_rec": (is_rule or is_llm)})
                
            questions.append({
                "header": "Interactions", "type": "choice", "multiSelect": True,
                "options": int_options,
                "question": "Select interactions (R=All Recommended, A=All Possible, N=None):"
            })

        if args.polynomials:
            rule_poly = set(rule_recs["polynomials"])
            llm_poly = set([p['col'] if isinstance(p, dict) else p for p in llm_recs["polynomials"]]) if llm_recs and "polynomials" in llm_recs else set()
            
            all_numeric = [p.name for p in analysis.column_profiles if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)]
            poly_options = []
            for col in all_numeric:
                is_rule = col in rule_poly
                is_llm = col in llm_poly
                prefix = ""
                if is_rule: prefix += "[Rule]"
                if is_llm: prefix += "[LLM]"
                
                for deg in [1, 2, 3]:
                    label = f"{col} deg {deg}"
                    if prefix: label = f"{prefix} {label}"
                    poly_options.append({"label": label, "description": "Polynomial", "is_rec": (is_rule or is_llm)})

            questions.append({
                "header": "Polynomials", "type": "choice", "multiSelect": True,
                "options": poly_options,
                "question": "Select polynomials (R=All Recommended, A=All Possible, N=None):"
            })

        # D. Capture Input with shortcuts
        # Overriding ask_user here for 'R' shortcut
        user_choices = ask_user_construction(questions)
        
        # Parse choices
        if args.interactions and len(user_choices) > 0:
            for opt in user_choices[0]['value']:
                clean_label = opt['label'].split(']')[-1].strip()
                selected_ints.append(clean_label.split(' x '))
        poly_idx = 1 if args.interactions else 0
        if args.polynomials and len(user_choices) > poly_idx:
            for opt in user_choices[poly_idx]['value']:
                # Clean label: "[Rule][LLM] col deg 2" -> "col deg 2"
                clean_label = opt['label'].split(']')[-1].strip()
                parts = clean_label.split(' ')
                selected_polys.append({'col': parts[0], 'degree': int(parts[2])})

    # PHASE 2: Transformation
    print("PHASE 2: Applying Transformations...")
    final_config = FEConfig(
        target_column=args.target, output_dir=args.output_dir,
        selected_interactions=selected_ints,
        selected_polynomial_features=selected_polys,
        llm=LLMConfig(enabled=args.llm)
    )
    agent_final = FeatureEngineeringAgent(config=final_config)
    final_result = agent_final.run(args.source)

    # PHASE 3: Final Pruning
    print(f"\nPHASE 3: Final Feature Pruning ({len(final_result.transformed_df.columns)} features)...")
    selection_rationale = final_result.decision_log.llm_advisory["selection_rationale"]
    pruning_options = [{"label": f"{f} ({d['status']})", "description": d['rationale']} for f, d in selection_rationale.items()]

    user_final = ask_user([{
        "header": "Final Pruning", "type": "choice", "multiSelect": True,
        "options": pruning_options,
        "question": "Select features to DROP (U=Useless, W=Weak, N=Keep All):"
    }])
    
    if user_final and user_final[0].get('value'):
        to_drop = [opt['label'].split(' (')[0] for opt in user_final[0]['value']]
        final_df = final_result.transformed_df.drop(columns=[c for c in to_drop if c in final_result.transformed_df.columns])
        ext = Path(args.source).suffix.lower() if isinstance(args.source, str) else '.csv'
        final_path = Path(args.output_dir) / f"final_dataset_{final_result.run_id}{ext}"
        if ext == '.parquet': final_df.to_parquet(final_path, index=False)
        else: final_df.to_csv(final_path, index=False)
        print(f"Final column count: {len(final_df.columns)}. Saved to: {final_path}")

    print(f"Done! Reports in {args.output_dir}")

def ask_user_construction(questions):
    results = []
    for q in questions:
        print(f"\n--- {q['header']} ---")
        print(f"{q['question']}")
        print(f"R: All Recommended")
        print(f"A: All Possible")
        print(f"N: None of the above")
        for i, opt in enumerate(q['options']):
            print(f"{i}: {opt['label']}")
        
        choice = input("\nChoice (R, A, N, or indices): ").strip().upper()
        if choice == 'R': selected = [o for o in q['options'] if o.get('is_rec')]
        elif choice == 'A': selected = q['options']
        elif choice == 'N': selected = []
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',') if x.strip()]
                selected = [q['options'][i] for i in indices]
            except: selected = []
        results.append({'value': selected})
    return results

if __name__ == "__main__":
    main()
