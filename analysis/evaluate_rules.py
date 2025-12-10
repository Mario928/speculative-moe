#!/usr/bin/env python3
"""
EVALUATE RULES ON SPECIFIED TOKEN JOURNEY FILES

Usage:
    python evaluate_rules.py --rules rules.json --test <test_file1> [test_file2] ... [--output results.json]

Example:
    python evaluate_rules.py --rules rules_from_1.json --test p0_token_1.jsonl p0_token_2.jsonl
    python evaluate_rules.py --rules rules.json --test p0_token_*.jsonl p1_token_*.jsonl
    python evaluate_rules.py --rules rules.json --test p1_token_*.jsonl --output results.json

Output: Accuracy for all 5 levels (prints to stdout and optionally saves to JSON)
"""

import argparse
import json
import glob
from datetime import datetime

def load_journey(filename):
    """Load a token journey file (32 layers)"""
    with open(filename) as f:
        journey = [json.loads(l) for l in f if l.strip()]
    journey.sort(key=lambda x: x['layer'])
    return journey

def evaluate_level1(journey, rules):
    correct, total = 0, 0
    for i in range(len(journey) - 1):
        key = str((journey[i]['layer'], journey[i]['experts'][0]))
        if key in rules:
            total += 1
            if rules[key]['prediction'] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level2(journey, rules):
    correct, total = 0, 0
    for i in range(1, len(journey) - 1):
        key = str((journey[i]['layer'], journey[i-1]['experts'][0], journey[i]['experts'][0]))
        if key in rules:
            total += 1
            if rules[key]['prediction'] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level3(journey, rules):
    correct, total = 0, 0
    for i in range(2, len(journey) - 1):
        key = str((journey[i]['layer'], journey[i-2]['experts'][0], journey[i-1]['experts'][0], journey[i]['experts'][0]))
        if key in rules:
            total += 1
            if rules[key]['prediction'] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level4(journey, rules):
    correct, total = 0, 0
    for i in range(len(journey) - 1):
        key = str((journey[i]['layer'], journey[i]['experts'][0], journey[i]['experts'][1]))
        if key in rules:
            total += 1
            if rules[key]['prediction'] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level5(journey, rules, skip):
    correct, total = 0, 0
    for i in range(len(journey) - skip):
        key = str((journey[i]['layer'], journey[i]['experts'][0]))
        if key in rules:
            total += 1
            if rules[key]['prediction'] == journey[i+skip]['experts'][0]:
                correct += 1
    return correct, total

def main():
    parser = argparse.ArgumentParser(description='Evaluate expert prediction rules on token journey files')
    parser.add_argument('--rules', '-r', required=True, help='Rules JSON file (from create_rules.py)')
    parser.add_argument('--test', '-t', nargs='+', required=True, help='Token journey files for testing')
    parser.add_argument('--output', '-o', help='Output JSON file to save results (optional)')
    args = parser.parse_args()

    print("="*70)
    print("EVALUATE RULES ON TOKEN JOURNEY FILES")
    print("="*70)

    # Load rules
    print(f"\nLoading rules from: {args.rules}")
    with open(args.rules) as f:
        all_rules = json.load(f)

    print(f"  Training files used: {all_rules['metadata']['training_files']}")
    print(f"  Level 1 rules: {len(all_rules['level1']['rules'])}")
    print(f"  Level 2 rules: {len(all_rules['level2']['rules'])}")
    print(f"  Level 3 rules: {len(all_rules['level3']['rules'])}")
    print(f"  Level 4 rules: {len(all_rules['level4']['rules'])}")

    # Expand glob patterns in test files
    test_files = []
    for pattern in args.test:
        matches = glob.glob(pattern)
        if matches:
            test_files.extend(matches)
        else:
            test_files.append(pattern)
    test_files = sorted(set(test_files))

    # Load test journeys
    print(f"\nTest files: {len(test_files)}")
    test_journeys = {}
    for fname in test_files:
        j = load_journey(fname)
        test_journeys[fname] = j
        print(f"  {fname}: {len(j)} layers")

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    results = {
        'level1': {'correct': 0, 'total': 0},
        'level2': {'correct': 0, 'total': 0},
        'level3': {'correct': 0, 'total': 0},
        'level4': {'correct': 0, 'total': 0},
        'level5_skip2': {'correct': 0, 'total': 0},
        'level5_skip4': {'correct': 0, 'total': 0},
        'level5_skip8': {'correct': 0, 'total': 0},
    }

    print(f"\n{'File':<25} {'L1':<10} {'L2':<10} {'L3':<10} {'L4':<10}")
    print("-"*70)

    for fname, journey in test_journeys.items():
        c1, t1 = evaluate_level1(journey, all_rules['level1']['rules'])
        c2, t2 = evaluate_level2(journey, all_rules['level2']['rules'])
        c3, t3 = evaluate_level3(journey, all_rules['level3']['rules'])
        c4, t4 = evaluate_level4(journey, all_rules['level4']['rules'])

        results['level1']['correct'] += c1
        results['level1']['total'] += t1
        results['level2']['correct'] += c2
        results['level2']['total'] += t2
        results['level3']['correct'] += c3
        results['level3']['total'] += t3
        results['level4']['correct'] += c4
        results['level4']['total'] += t4

        # Level 5
        for skip in [2, 4, 8]:
            c, t = evaluate_level5(journey, all_rules['level5']['rules'][f'skip_{skip}'], skip)
            results[f'level5_skip{skip}']['correct'] += c
            results[f'level5_skip{skip}']['total'] += t

        # Short filename for display
        short_name = fname.split('/')[-1]
        l1 = f"{c1}/{t1}" if t1 > 0 else "-"
        l2 = f"{c2}/{t2}" if t2 > 0 else "-"
        l3 = f"{c3}/{t3}" if t3 > 0 else "-"
        l4 = f"{c4}/{t4}" if t4 > 0 else "-"
        print(f"{short_name:<25} {l1:<10} {l2:<10} {l3:<10} {l4:<10}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nRules from: {all_rules['metadata']['training_files']}")
    print(f"Tested on: {len(test_files)} files")

    print(f"\n{'Level':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'vs Random'}")
    print("-"*70)

    for level_name, data in results.items():
        c = data['correct']
        t = data['total']
        acc = c/t*100 if t > 0 else 0
        vs_random = f"{acc/12.5:.1f}x" if acc > 0 else "-"

        display_name = {
            'level1': 'Level 1 (Layer, E)',
            'level2': 'Level 2 (Layer, Prev, Curr)',
            'level3': 'Level 3 (Layer, E-2, E-1, E)',
            'level4': 'Level 4 (Layer, P, S)',
            'level5_skip2': 'Level 5 (Skip 2)',
            'level5_skip4': 'Level 5 (Skip 4)',
            'level5_skip8': 'Level 5 (Skip 8)',
        }[level_name]

        print(f"{display_name:<30} {c:<10} {t:<10} {acc:.1f}%        {vs_random}")

    print("-"*70)
    print("Random baseline: 12.5%")

    # Best level
    best = max(results.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0)
    best_acc = best[1]['correct']/best[1]['total']*100 if best[1]['total'] > 0 else 0
    print(f"\n🏆 BEST: {best[0]} with {best_acc:.1f}% accuracy")
    print("="*70)

    # Save results to JSON if output file specified
    if args.output:
        # Build results with accuracy
        results_with_accuracy = {}
        for level_name, data in results.items():
            c = data['correct']
            t = data['total']
            acc = c/t*100 if t > 0 else 0
            results_with_accuracy[level_name] = {
                'correct': c,
                'total': t,
                'accuracy_percent': round(acc, 2),
                'vs_random': round(acc/12.5, 2) if acc > 0 else 0
            }

        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'rules_file': args.rules,
                'training_files': all_rules['metadata']['training_files'],
                'test_files': [f.split('/')[-1] for f in test_files],
                'num_test_files': len(test_files)
            },
            'results': results_with_accuracy,
            'best_level': {
                'name': best[0],
                'accuracy_percent': round(best_acc, 2)
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

if __name__ == "__main__":
    main()
