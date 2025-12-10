#!/usr/bin/env python3
"""
EVALUATE ALL 5 LEVELS
- Rules from: p0_token_0 ONLY
- Evaluate on: p0_token_1-9 and p1_token_0-9 (19 tokens)
"""

import json

print("="*70)
print("EXPERT PREDICTION - 5 LEVELS")
print("Rules from: p0_token_0 | Evaluate on: 19 other tokens")
print("="*70)

# Load p0_token_0 (training)
def load_journey(filename):
    with open(filename) as f:
        journey = [json.loads(l) for l in f if l.strip()]
    journey.sort(key=lambda x: x['layer'])
    return journey

print("\nLoading training token (p0_token_0)...")
train = load_journey('p0_token_0.jsonl')
print(f"  Layers: {len(train)}")
print(f"  Experts: {[d['experts'][0] for d in train]}")

# Load all test tokens
print("\nLoading test tokens...")
test_tokens = {}

# Problem 0, tokens 1-9
for t in range(1, 10):
    fname = f'p0_token_{t}.jsonl'
    test_tokens[f'p0_t{t}'] = load_journey(fname)

# Problem 1, tokens 0-9
for t in range(10):
    fname = f'p1_token_{t}.jsonl'
    test_tokens[f'p1_t{t}'] = load_journey(fname)

print(f"  Loaded {len(test_tokens)} test tokens")


# ============================================================================
# CREATE RULES FROM p0_token_0
# ============================================================================
print("\n" + "="*70)
print("CREATING RULES FROM p0_token_0")
print("="*70)

# Level 1: (Layer, Expert) -> Next
level1_rules = {}
for i in range(len(train) - 1):
    layer = train[i]['layer']
    expert = train[i]['experts'][0]
    next_expert = train[i + 1]['experts'][0]
    level1_rules[(layer, expert)] = next_expert
print(f"\nLevel 1 rules: {len(level1_rules)}")

# Level 2: (Layer, Prev, Curr) -> Next
level2_rules = {}
for i in range(1, len(train) - 1):
    layer = train[i]['layer']
    prev_e = train[i-1]['experts'][0]
    curr_e = train[i]['experts'][0]
    next_e = train[i+1]['experts'][0]
    level2_rules[(layer, prev_e, curr_e)] = next_e
print(f"Level 2 rules: {len(level2_rules)}")

# Level 3: (Layer, E-2, E-1, E) -> Next
level3_rules = {}
for i in range(2, len(train) - 1):
    layer = train[i]['layer']
    e2 = train[i-2]['experts'][0]
    e1 = train[i-1]['experts'][0]
    e0 = train[i]['experts'][0]
    next_e = train[i+1]['experts'][0]
    level3_rules[(layer, e2, e1, e0)] = next_e
print(f"Level 3 rules: {len(level3_rules)}")

# Level 4: (Layer, Primary, Secondary) -> Next Primary
level4_rules = {}
for i in range(len(train) - 1):
    layer = train[i]['layer']
    prim = train[i]['experts'][0]
    sec = train[i]['experts'][1]
    next_prim = train[i+1]['experts'][0]
    level4_rules[(layer, prim, sec)] = next_prim
print(f"Level 4 rules: {len(level4_rules)}")

# Level 5: Skip layers (K=2,4,8)
level5_rules = {2: {}, 4: {}, 8: {}}
for skip in [2, 4, 8]:
    for i in range(len(train) - skip):
        layer = train[i]['layer']
        expert = train[i]['experts'][0]
        future = train[i + skip]['experts'][0]
        level5_rules[skip][(layer, expert)] = future
    print(f"Level 5 (skip {skip}) rules: {len(level5_rules[skip])}")


# ============================================================================
# SHOW ALL RULES
# ============================================================================
print("\n" + "="*70)
print("LEVEL 1 RULES: (Layer, Expert) -> Next")
print("="*70)
for (layer, expert), next_e in sorted(level1_rules.items()):
    print(f"  Layer {layer:2d}, Expert {expert} -> Expert {next_e}")

print("\n" + "="*70)
print("LEVEL 2 RULES: (Layer, Prev, Curr) -> Next")
print("="*70)
for (layer, prev, curr), next_e in sorted(level2_rules.items()):
    print(f"  Layer {layer:2d}, ({prev}, {curr}) -> Expert {next_e}")

print("\n" + "="*70)
print("LEVEL 3 RULES: (Layer, E-2, E-1, E) -> Next")
print("="*70)
for (layer, e2, e1, e0), next_e in sorted(level3_rules.items()):
    print(f"  Layer {layer:2d}, ({e2}, {e1}, {e0}) -> Expert {next_e}")

print("\n" + "="*70)
print("LEVEL 4 RULES: (Layer, Primary, Secondary) -> Next Primary")
print("="*70)
for (layer, prim, sec), next_e in sorted(level4_rules.items()):
    print(f"  Layer {layer:2d}, (P={prim}, S={sec}) -> Expert {next_e}")


# ============================================================================
# EVALUATE ON TEST TOKENS
# ============================================================================
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

def evaluate_level1(journey, rules):
    correct, total = 0, 0
    for i in range(len(journey) - 1):
        key = (journey[i]['layer'], journey[i]['experts'][0])
        if key in rules:
            total += 1
            if rules[key] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level2(journey, rules):
    correct, total = 0, 0
    for i in range(1, len(journey) - 1):
        key = (journey[i]['layer'], journey[i-1]['experts'][0], journey[i]['experts'][0])
        if key in rules:
            total += 1
            if rules[key] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level3(journey, rules):
    correct, total = 0, 0
    for i in range(2, len(journey) - 1):
        key = (journey[i]['layer'], journey[i-2]['experts'][0], journey[i-1]['experts'][0], journey[i]['experts'][0])
        if key in rules:
            total += 1
            if rules[key] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level4(journey, rules):
    correct, total = 0, 0
    for i in range(len(journey) - 1):
        key = (journey[i]['layer'], journey[i]['experts'][0], journey[i]['experts'][1])
        if key in rules:
            total += 1
            if rules[key] == journey[i+1]['experts'][0]:
                correct += 1
    return correct, total

def evaluate_level5(journey, rules, skip):
    correct, total = 0, 0
    for i in range(len(journey) - skip):
        key = (journey[i]['layer'], journey[i]['experts'][0])
        if key in rules:
            total += 1
            if rules[key] == journey[i+skip]['experts'][0]:
                correct += 1
    return correct, total

# Evaluate all levels
results = {
    'level1': {'correct': 0, 'total': 0},
    'level2': {'correct': 0, 'total': 0},
    'level3': {'correct': 0, 'total': 0},
    'level4': {'correct': 0, 'total': 0},
    'level5_skip2': {'correct': 0, 'total': 0},
    'level5_skip4': {'correct': 0, 'total': 0},
    'level5_skip8': {'correct': 0, 'total': 0},
}

print("\nPer-token results:")
print("-"*70)
print(f"{'Token':<10} {'L1':<12} {'L2':<12} {'L3':<12} {'L4':<12}")
print("-"*70)

for name, journey in test_tokens.items():
    c1, t1 = evaluate_level1(journey, level1_rules)
    c2, t2 = evaluate_level2(journey, level2_rules)
    c3, t3 = evaluate_level3(journey, level3_rules)
    c4, t4 = evaluate_level4(journey, level4_rules)

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
        c, t = evaluate_level5(journey, level5_rules[skip], skip)
        results[f'level5_skip{skip}']['correct'] += c
        results[f'level5_skip{skip}']['total'] += t

    acc1 = f"{c1}/{t1}" if t1 > 0 else "-"
    acc2 = f"{c2}/{t2}" if t2 > 0 else "-"
    acc3 = f"{c3}/{t3}" if t3 > 0 else "-"
    acc4 = f"{c4}/{t4}" if t4 > 0 else "-"
    print(f"{name:<10} {acc1:<12} {acc2:<12} {acc3:<12} {acc4:<12}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\n{'Level':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'vs Random'}")
print("-"*70)

for level_name, data in results.items():
    c = data['correct']
    t = data['total']
    acc = c/t*100 if t > 0 else 0
    vs_random = f"{acc/12.5:.1f}x" if acc > 0 else "-"

    display_name = {
        'level1': 'Level 1 (Layer, E)',
        'level2': 'Level 2 (Layer, Prev, Curr)',
        'level3': 'Level 3 (Layer, E-2,E-1,E)',
        'level4': 'Level 4 (Layer, P, S)',
        'level5_skip2': 'Level 5 (Skip 2)',
        'level5_skip4': 'Level 5 (Skip 4)',
        'level5_skip8': 'Level 5 (Skip 8)',
    }[level_name]

    print(f"{display_name:<25} {c:<10} {t:<10} {acc:.1f}%      {vs_random}")

print("-"*70)
print("Random baseline: 12.5%")
print("="*70)

# Find best
best = max(results.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0)
best_acc = best[1]['correct']/best[1]['total']*100 if best[1]['total'] > 0 else 0
print(f"\n🏆 BEST: {best[0]} with {best_acc:.1f}% accuracy")
