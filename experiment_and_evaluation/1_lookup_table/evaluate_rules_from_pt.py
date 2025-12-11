"""
EVALUATE RULES ON TEST SET FROM PT FILES
Uses the same processed_data/*.pt as MLP/XGBoost for fair comparison.
Now includes Top-2 accuracy!
"""
import torch
import json
import os

def evaluate_level(data, all_rules, level_key, get_key_fn):
    """Generic evaluation function for any level"""
    layers = data['layers'].numpy()
    history = data['history'].numpy()
    secondary = data['secondary'].numpy()
    targets = data['targets'].numpy()
    n_samples = len(targets)
    
    correct_top1, correct_top2, total = 0, 0, 0
    
    for i in range(n_samples):
        layer = int(layers[i])
        target = int(targets[i])
        
        key = get_key_fn(i, layer, history, secondary)
        if key is None:
            continue
            
        key_str = str(key)
        if key_str in all_rules[level_key]['rules']:
            total += 1
            rule = all_rules[level_key]['rules'][key_str]
            
            # Top-1 check
            if rule['prediction'] == target:
                correct_top1 += 1
            
            # Top-2 check
            if rule['prediction'] == target or rule.get('prediction2', rule['prediction']) == target:
                correct_top2 += 1
    
    acc1 = (correct_top1 / total * 100) if total > 0 else 0
    acc2 = (correct_top2 / total * 100) if total > 0 else 0
    return {'correct_top1': correct_top1, 'correct_top2': correct_top2, 'total': total, 'acc_top1': acc1, 'acc_top2': acc2}

def main():
    # Load rules
    print("Loading rules.json...")
    with open("rules.json") as f:
        all_rules = json.load(f)
    
    # Load test data
    print("Loading ../processed_data/test_data.pt...")
    data = torch.load("../processed_data/test_data.pt")
    n_samples = len(data['targets'])
    print(f"Loaded {n_samples} test samples")
    print("=" * 70)
    
    # Define key functions for each level
    def level1_key(i, layer, history, secondary):
        expert = int(history[i][layer] - 1)
        return (layer, expert)
    
    def level2_key(i, layer, history, secondary):
        if layer < 1: return None
        prev = int(history[i][layer-1] - 1)
        curr = int(history[i][layer] - 1)
        return (layer, prev, curr)
    
    def level3_key(i, layer, history, secondary):
        if layer < 2: return None
        e2 = int(history[i][layer-2] - 1)
        e1 = int(history[i][layer-1] - 1)
        e0 = int(history[i][layer] - 1)
        return (layer, e2, e1, e0)
    
    def level4_key(i, layer, history, secondary):
        prim = int(history[i][layer] - 1)
        sec = int(secondary[i])
        return (layer, prim, sec)
    
    def level5_key(i, layer, history, secondary):
        path = tuple(int(history[i][j] - 1) for j in range(layer + 1))
        return (layer, path)
    
    # Evaluate each level
    results = {}
    for name, key_fn in [('level1', level1_key), ('level2', level2_key), 
                          ('level3', level3_key), ('level4', level4_key), 
                          ('level5', level5_key)]:
        print(f"Evaluating {name}...")
        results[name] = evaluate_level(data, all_rules, name, key_fn)
        r = results[name]
        print(f"  Top-1: {r['correct_top1']}/{r['total']} = {r['acc_top1']:.2f}%")
        print(f"  Top-2: {r['correct_top2']}/{r['total']} = {r['acc_top2']:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Level':<35} {'Top-1':<12} {'Top-2':<12}")
    print("-" * 70)
    for level, r in results.items():
        desc = all_rules[level]['description']
        print(f"{desc:<35} {r['acc_top1']:.2f}%       {r['acc_top2']:.2f}%")
    
    # Best level
    best_top1 = max(results.items(), key=lambda x: x[1]['acc_top1'])
    best_top2 = max(results.items(), key=lambda x: x[1]['acc_top2'])
    print(f"\n🏆 Best Top-1: {best_top1[0]} = {best_top1[1]['acc_top1']:.2f}%")
    print(f"🏆 Best Top-2: {best_top2[0]} = {best_top2[1]['acc_top2']:.2f}%")
    print(f"Random baseline: Top-1=12.5%, Top-2=25%")
    
    # Save
    with open("evaluation_results.json", 'w') as f:
        json.dump({"metadata": {"n_samples": n_samples}, "results": results}, f, indent=2)
    print("\n✓ Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
