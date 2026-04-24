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
from fe_agent.pipeline.pipeline_artifact import FEPipeline, FEStep

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

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Agent")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", help="Target column")
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--polynomials", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--output_dir", default="./fe_output")
    parser.add_argument("--replay", help="Path to a saved pipeline JSON")
    args = parser.parse_args()

    # REPLAY MODE
    if args.replay:
        print(f"Replay Mode: Applying pipeline {args.replay} to {args.source}...")
        pipeline = FEPipeline.load(args.replay)
        df = pd.read_csv(args.source)
        transformed_df = pipeline.apply(df)
        out_path = Path(args.output_dir) / f"replayed_{Path(args.source).name}"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        transformed_df.to_csv(out_path, index=False)
        print(f"Replay complete. Saved to {out_path}")
        return

    if not args.target:
        print("Error: --target is required for standard run.")
        sys.exit(1)

    # PHASE 0 & 1: Profiling and Recommendation
    print("PHASE 0: Initial Analysis...")
    agent_base = FeatureEngineeringAgent(config=FEConfig(target_column=args.target))
    analysis = agent_base.run(args.source, skip_io=True)
    
    selected_ints, selected_polys = [], []
    if args.interactions or args.polynomials:
        print("PHASE 1: Smart Feature Construction Selection...")
        optimizer = DecisionOptimizer(None)
        ranking = analysis.decision_log.llm_advisory["feature_importance_ranking"]
        rule_recs = optimizer.get_construction_recommendations(ranking, analysis.column_profiles)
        llm_recs = agent_base.llm_advisor.get_construction_advice(ranking) if args.llm else None

        questions = []
        if args.interactions:
            rule_pairs = set([tuple(sorted(p)) for p in rule_recs["interactions"]])
            llm_pairs = set([tuple(sorted(p)) for p in llm_recs["interactions"]]) if llm_recs and "interactions" in llm_recs else set()
            all_feats = [p.name for p in analysis.column_profiles if p.name != args.target and p.semantic_type != SemanticType.DATETIME]
            all_possible = [(all_feats[i], all_feats[j]) for i in range(len(all_feats)) for j in range(i+1, len(all_feats))]
            int_opts = []
            for a, b in all_possible:
                pair = tuple(sorted((a,b))); prefix = ""
                if pair in rule_pairs: prefix += "[Rule]"
                if pair in llm_pairs: prefix += "[LLM]"
                int_opts.append({"label": f"{prefix} {a} x {b}" if prefix else f"{a} x {b}", "is_rec": (pair in rule_pairs or pair in llm_pairs)})
            questions.append({"header": "Interactions", "type": "choice", "multiSelect": True, "options": int_opts, "question": "Select interactions:"})

        if args.polynomials:
            rec_poly = set(rule_recs["polynomials"])
            llm_poly = set([p['col'] if isinstance(p, dict) else p for p in llm_recs["polynomials"]]) if llm_recs and "polynomials" in llm_recs else set()
            all_num = [p.name for p in analysis.column_profiles if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)]
            poly_opts = []
            for col in all_num:
                prefix = ""
                if col in rec_poly: prefix += "[Rule]"
                if col in llm_poly: prefix += "[LLM]"
                for deg in [1, 2, 3]:
                    poly_opts.append({"label": f"{prefix} {col} deg {deg}" if prefix else f"{col} deg {deg}", "is_rec": (col in rec_poly or col in llm_poly)})
            questions.append({"header": "Polynomials", "type": "choice", "multiSelect": True, "options": poly_opts, "question": "Select polynomials:"})

        user_choices = ask_user_construction(questions)
        if args.interactions and len(user_choices) > 0:
            selected_ints = [opt['label'].split(']')[-1].strip().split(' x ') for opt in user_choices[0]['value']]
        p_idx = 1 if args.interactions else 0
        if args.polynomials and len(user_choices) > p_idx:
            for opt in user_choices[p_idx]['value']:
                parts = opt['label'].split(']')[-1].strip().split(' ')
                selected_polys.append({'col': parts[0], 'degree': int(parts[2])})

    # PHASE 2: Final Execution
    print("PHASE 2: Transformation and Final Execution...")
    final_config = FEConfig(target_column=args.target, output_dir=args.output_dir, selected_interactions=selected_ints, selected_polynomial_features=selected_polys, llm=LLMConfig(enabled=args.llm))
    final_agent = FeatureEngineeringAgent(config=final_config)
    final_result = final_agent.run(args.source)
    
    pipe_path = Path(args.output_dir) / f"pipeline_{final_result.run_id}.json"
    final_result.pipeline_artifact.save(pipe_path)
    print(f"Pipeline saved to: {pipe_path}")

    # PHASE 3: Pruning
    print(f"\nPHASE 3: Final Feature Pruning ({len(final_result.transformed_df.columns)} features)...")
    ranking = final_result.decision_log.llm_advisory["selection_rationale"]
    pruning_opts = [{"label": f"{f} ({d['status']})", "description": d['rationale']} for f, d in ranking.items()]
    user_final = ask_user([{"header": "Final Pruning", "type": "choice", "multiSelect": True, "options": pruning_opts, "question": "Select features to DROP:"}])
    
    if user_final and user_final[0].get('value'):
        to_drop = [opt['label'].split(' (')[0] for opt in user_final[0]['value']]
        final_dataset = final_result.transformed_df.drop(columns=[c for c in to_drop if c in final_result.transformed_df.columns])
        
        # RECORD DROPS & FINAL COLUMNS
        final_result.pipeline_artifact.add_step(FEStep("pruning", "drop", to_drop, []))
        final_result.pipeline_artifact.set_final_columns(list(final_dataset.columns))
        final_result.pipeline_artifact.save(pipe_path)
        
        final_dataset.to_csv(Path(args.output_dir) / f"final_dataset_{final_result.run_id}.csv", index=False)
        print(f"Pruning complete. Final column count: {len(final_dataset.columns)}")

    print(f"Process complete. Pipeline and results in {args.output_dir}")

if __name__ == "__main__":
    main()
