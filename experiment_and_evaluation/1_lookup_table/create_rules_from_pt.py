"""
CREATE RULES FROM PT FILES
Uses the same processed_data/*.pt as MLP/XGBoost for fair comparison.
Extracts 5 levels of rules (removed skip-layer rules).
"""
import torch
import json
import os
from collections import defaultdict, Counter

def create_rules_with_vote(data_dict):
    """Create rules using majority vote, also store 2nd best for Top-2"""
    rules = {}
    for key, predictions in data_dict.items():
        counter = Counter(predictions)
        most_common = counter.most_common(2)  # Get top 2
        
        # Top-1 prediction
        rule = most_common[0][0]
        confidence = most_common[0][1] / len(predictions)
        
        # Top-2 prediction (if exists)
        top2 = most_common[1][0] if len(most_common) > 1 else rule
        
        rules[str(key)] = {
            "prediction": rule,
            "prediction2": top2,
            "confidence": confidence,
            "samples": len(predictions)
        }
    return rules

def main():
    # Load training data
    data_path = "../processed_data/train_data.pt"
    print(f"Loading {data_path}...")
    data = torch.load(data_path)
    
    layers = data['layers'].numpy()
    history = data['history'].numpy()
    secondary = data['secondary'].numpy()
    targets = data['targets'].numpy()
    n_samples = len(targets)
    
    print(f"Loaded {n_samples} training samples")
    
    # Level 1: (Layer, Expert) -> Next
    print("\nExtracting Level 1 rules...")
    level1_data = defaultdict(list)
    for i in range(n_samples):
        layer = int(layers[i])
        expert = int(history[i][layer] - 1)  # Unshift +1 offset
        target = int(targets[i])
        level1_data[(layer, expert)].append(target)
    level1_rules = create_rules_with_vote(level1_data)
    print(f"  Level 1: {len(level1_rules)} rules")
    
    # Level 2: (Layer, Prev, Curr) -> Next
    print("Extracting Level 2 rules...")
    level2_data = defaultdict(list)
    for i in range(n_samples):
        layer = int(layers[i])
        if layer < 1:
            continue
        prev_expert = int(history[i][layer-1] - 1)
        curr_expert = int(history[i][layer] - 1)
        target = int(targets[i])
        level2_data[(layer, prev_expert, curr_expert)].append(target)
    level2_rules = create_rules_with_vote(level2_data)
    print(f"  Level 2: {len(level2_rules)} rules")
    
    # Level 3: (Layer, E-2, E-1, E) -> Next
    print("Extracting Level 3 rules...")
    level3_data = defaultdict(list)
    for i in range(n_samples):
        layer = int(layers[i])
        if layer < 2:
            continue
        e2 = int(history[i][layer-2] - 1)
        e1 = int(history[i][layer-1] - 1)
        e0 = int(history[i][layer] - 1)
        target = int(targets[i])
        level3_data[(layer, e2, e1, e0)].append(target)
    level3_rules = create_rules_with_vote(level3_data)
    print(f"  Level 3: {len(level3_rules)} rules")
    
    # Level 4: (Layer, Primary, Secondary) -> Next
    print("Extracting Level 4 rules...")
    level4_data = defaultdict(list)
    for i in range(n_samples):
        layer = int(layers[i])
        prim = int(history[i][layer] - 1)
        sec = int(secondary[i])
        target = int(targets[i])
        level4_data[(layer, prim, sec)].append(target)
    level4_rules = create_rules_with_vote(level4_data)
    print(f"  Level 4: {len(level4_rules)} rules")
    
    # Level 5: (Layer, Full Path) -> Next
    print("Extracting Level 5 rules...")
    level5_data = defaultdict(list)
    for i in range(n_samples):
        layer = int(layers[i])
        # Full path from layer 0 to current
        path = tuple(int(history[i][j] - 1) for j in range(layer + 1))
        target = int(targets[i])
        level5_data[(layer, path)].append(target)
    level5_rules = create_rules_with_vote(level5_data)
    print(f"  Level 5: {len(level5_rules)} rules")
    
    # Save
    output = {
        "metadata": {
            "source": "processed_data/train_data.pt",
            "n_samples": n_samples,
            "description": "Rules extracted from unified PT data"
        },
        "level1": {"description": "(Layer, Expert) -> Next", "rules": level1_rules},
        "level2": {"description": "(Layer, Prev, Curr) -> Next", "rules": level2_rules},
        "level3": {"description": "(Layer, E-2, E-1, E) -> Next", "rules": level3_rules},
        "level4": {"description": "(Layer, Primary, Secondary) -> Next", "rules": level4_rules},
        "level5": {"description": "(Layer, Full Path) -> Next", "rules": level5_rules},
    }
    
    output_path = "rules.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Rules saved to {output_path}")

if __name__ == "__main__":
    main()
