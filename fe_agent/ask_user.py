from typing import List, Dict, Any, Optional

def ask_user(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for q in questions:
        print(f"\n--- {q['header']} ---")
        print(f"{q['question']}")
        
        # Add special options
        options = q['options']
        print(f"A: All of the above")
        print(f"N: None of the above")
        for i, opt in enumerate(options):
            print(f"{i}: {opt['label']} - {opt['description']}")
        
        choice = input("\nEnter indices (e.g., 0,1), 'A' for All, or 'N' for None: ").strip().upper()
        
        if choice == 'A':
            selected = options
        elif choice == 'N':
            selected = []
        else:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected = [options[i] for i in indices]
            
        results.append({'value': selected})
    return results
