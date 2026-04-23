from typing import List, Dict, Any, Optional

def ask_user(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for q in questions:
        print(f"\n--- {q['header']} ---")
        print(f"{q['question']}")

        options = q['options']
        print(f"A: All of the above")
        print(f"N: None of the above")

        # Only show U/W for Pruning
        is_pruning = "Pruning" in q['header'] or "Selection" in q['header']
        if is_pruning:
            print(f"U: Select all 'drop_useless' features")
            print(f"W: Select all 'drop_useless' and 'keep_weak' features")

        for i, opt in enumerate(options):
            print(f"{i}: {opt['label']} - {opt['description']}")

        prompt_txt = "\nEnter indices (e.g. 0,1), 'A' for All, 'N' for None"
        if is_pruning: prompt_txt += ", 'U' for Useless, or 'W' for Weak"
        choice = input(f"{prompt_txt}: ").strip().upper()

        if choice == 'A':
            selected = options
        elif choice == 'N':
            selected = []
        elif is_pruning and choice == 'U':
            selected = [opt for opt in options if 'drop_useless' in opt['label']]
        elif is_pruning and choice == 'W':
            selected = [opt for opt in options if 'drop_useless' in opt['label'] or 'keep_weak' in opt['label']]
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',') if x.strip()]
                selected = [options[i] for i in indices]
            except:
                selected = []

        results.append({'value': selected})
    return results

