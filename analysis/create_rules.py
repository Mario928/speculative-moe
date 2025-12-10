#!/usr/bin/env python3
"""
CREATE RULES FROM SPECIFIED TOKEN JOURNEY FILES

Usage:
    python create_rules.py <train_file1> [train_file2] ... --output rules.json

Example:
    python create_rules.py p0_token_0.jsonl --output rules_from_1.json
    python create_rules.py p0_token_0.jsonl p0_token_1.jsonl p0_token_2.jsonl --output rules_from_3.json

Output: JSON file with all rules for 5 levels
"""

import argparse
import json
from collections import defaultdict, Counter

def load_journey(filename):
    """Load a token journey file (32 layers)"""
    with open(filename) as f:
        journey = [json.loads(l) for l in f if l.strip()]
    journey.sort(key=lambda x: x['layer'])
    return journey

def create_rules_with_vote(data_dict):
    """Create rules using majority vote when multiple training files"""
    rules = {}
    for key, predictions in data_dict.items():
        counter = Counter(predictions)
        most_common = counter.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            rule = most_common[0][0]
            confidence = most_common[0][1] / len(predictions)
        else:
            # Tie - pick first (most common)
            rule = most_common[0][0]
            confidence = most_common[0][1] / len(predictions)

        # Convert tuple key to string for JSON
        rules[str(key)] = {
            "prediction": rule,
            "confidence": confidence,
            "votes": dict(counter),
            "samples": len(predictions)
        }
    return rules

def main():
    parser = argparse.ArgumentParser(description='Create expert prediction rules from token journey files')
    parser.add_argument('train_files', nargs='+', help='Token journey files for training (e.g., p0_token_0.jsonl)')
    parser.add_argument('--output', '-o', default='rules.json', help='Output JSON file for rules')
    args = parser.parse_args()

    print("="*70)
    print("CREATE RULES FROM TOKEN JOURNEY FILES")
    print("="*70)

    # Load all training journeys
    print(f"\nTraining files: {len(args.train_files)}")
    journeys = []
    for fname in args.train_files:
        j = load_journey(fname)
        journeys.append(j)
        experts = [d['experts'][0] for d in j]
        print(f"  {fname}: {len(j)} layers")
        print(f"    Experts: {experts}")

    # Create rules for each level
    print("\n" + "-"*70)
    print("Creating rules...")
    print("-"*70)

    # Level 1: (Layer, Expert) -> Next
    level1_data = defaultdict(list)
    for journey in journeys:
        for i in range(len(journey) - 1):
            layer = journey[i]['layer']
            expert = journey[i]['experts'][0]
            next_expert = journey[i + 1]['experts'][0]
            level1_data[(layer, expert)].append(next_expert)
    level1_rules = create_rules_with_vote(level1_data)
    print(f"Level 1 rules: {len(level1_rules)}")

    # Level 2: (Layer, Prev, Curr) -> Next
    level2_data = defaultdict(list)
    for journey in journeys:
        for i in range(1, len(journey) - 1):
            layer = journey[i]['layer']
            prev_e = journey[i-1]['experts'][0]
            curr_e = journey[i]['experts'][0]
            next_e = journey[i+1]['experts'][0]
            level2_data[(layer, prev_e, curr_e)].append(next_e)
    level2_rules = create_rules_with_vote(level2_data)
    print(f"Level 2 rules: {len(level2_rules)}")

    # Level 3: (Layer, E-2, E-1, E) -> Next
    level3_data = defaultdict(list)
    for journey in journeys:
        for i in range(2, len(journey) - 1):
            layer = journey[i]['layer']
            e2 = journey[i-2]['experts'][0]
            e1 = journey[i-1]['experts'][0]
            e0 = journey[i]['experts'][0]
            next_e = journey[i+1]['experts'][0]
            level3_data[(layer, e2, e1, e0)].append(next_e)
    level3_rules = create_rules_with_vote(level3_data)
    print(f"Level 3 rules: {len(level3_rules)}")

    # Level 4: (Layer, Primary, Secondary) -> Next Primary
    level4_data = defaultdict(list)
    for journey in journeys:
        for i in range(len(journey) - 1):
            layer = journey[i]['layer']
            prim = journey[i]['experts'][0]
            sec = journey[i]['experts'][1]
            next_prim = journey[i+1]['experts'][0]
            level4_data[(layer, prim, sec)].append(next_prim)
    level4_rules = create_rules_with_vote(level4_data)
    print(f"Level 4 rules: {len(level4_rules)}")

    # Level 5: Skip layers
    level5_rules = {}
    for skip in [2, 4, 8]:
        level5_data = defaultdict(list)
        for journey in journeys:
            for i in range(len(journey) - skip):
                layer = journey[i]['layer']
                expert = journey[i]['experts'][0]
                future = journey[i + skip]['experts'][0]
                level5_data[(layer, expert)].append(future)
        level5_rules[f"skip_{skip}"] = create_rules_with_vote(level5_data)
        print(f"Level 5 (skip {skip}) rules: {len(level5_rules[f'skip_{skip}'])}")

    # Save to JSON
    output = {
        "metadata": {
            "training_files": args.train_files,
            "num_training_files": len(args.train_files),
            "description": "Expert prediction rules for Mixtral 8x7B"
        },
        "level1": {
            "description": "(Layer, Expert) -> Next Expert",
            "rules": level1_rules
        },
        "level2": {
            "description": "(Layer, Prev, Curr) -> Next Expert",
            "rules": level2_rules
        },
        "level3": {
            "description": "(Layer, E-2, E-1, E) -> Next Expert",
            "rules": level3_rules
        },
        "level4": {
            "description": "(Layer, Primary, Secondary) -> Next Primary",
            "rules": level4_rules
        },
        "level5": {
            "description": "(Layer, Expert) -> Expert at Layer+K",
            "rules": level5_rules
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Rules saved to: {args.output}")

    # Print sample rules
    print("\n" + "-"*70)
    print("SAMPLE RULES (Level 1):")
    print("-"*70)
    for i, (key, val) in enumerate(list(level1_rules.items())[:10]):
        print(f"  {key} -> {val['prediction']} (conf={val['confidence']:.0%}, samples={val['samples']})")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
