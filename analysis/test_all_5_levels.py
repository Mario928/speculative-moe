#!/usr/bin/env python3
"""
TEST ALL 5 PREDICTION LEVELS
- Build hypothesis from Token 0 ONLY
- Validate on Tokens 1-9
"""

import json
from collections import defaultdict, Counter

print("=" * 70)
print("TESTING ALL 5 PREDICTION LEVELS")
print("Hypothesis: Token 0 | Validation: Tokens 1-9")
print("=" * 70)

# Load all token journeys
token_files = [
    'token_journey_0_p0_t0.jsonl',  # Token 0 - for hypothesis
    'token_journey_1_p0_t1.jsonl',  # Tokens 1-9 - for validation
    'token_journey_2_p0_t2.jsonl',
    'token_journey_3_p0_t3.jsonl',
    'token_journey_4_p0_t4.jsonl',
    'token_journey_5_p1_t0.jsonl',
    'token_journey_6_p1_t1.jsonl',
    'token_journey_7_p1_t2.jsonl',
    'token_journey_8_p1_t3.jsonl',
    'token_journey_9_p1_t4.jsonl',
]

all_journeys = []
for filename in token_files:
    journey = []
    with open(f'/home/user/temp/{filename}', 'r') as f:
        for line in f:
            if line.strip():
                journey.append(json.loads(line))
    journey.sort(key=lambda x: x['layer'])
    all_journeys.append(journey)

token0 = all_journeys[0]  # For building hypothesis
validation_tokens = all_journeys[1:]  # For testing (9 tokens)

print(f"\nToken 0: {len(token0)} layers (for hypothesis)")
print(f"Validation: {len(validation_tokens)} tokens (for testing)")

# ============================================================
# LEVEL 1: Simple - (Layer, Expert) → Next Expert
# ============================================================
print("\n" + "=" * 70)
print("LEVEL 1: Simple - (Layer, Expert) → Next Expert")
print("=" * 70)

# Build hypothesis from Token 0
level1_rules = {}  # (layer, expert) → predicted_next

print("\n📝 Building rules from Token 0:")
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    expert = token0[i]['experts'][0]
    next_expert = token0[i + 1]['experts'][0]

    key = (layer, expert)
    level1_rules[key] = next_expert
    print(f"  Rule: Layer {layer}, Expert {expert} → Expert {next_expert}")

print(f"\nTotal rules from Token 0: {len(level1_rules)}")

# Validate on Tokens 1-9
print("\n📊 Validating on Tokens 1-9:")
level1_correct = 0
level1_total = 0
level1_no_rule = 0

for t_idx, journey in enumerate(validation_tokens):
    token_correct = 0
    token_total = 0

    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        expert = journey[i]['experts'][0]
        actual_next = journey[i + 1]['experts'][0]

        key = (layer, expert)
        if key in level1_rules:
            predicted = level1_rules[key]
            if predicted == actual_next:
                level1_correct += 1
                token_correct += 1
            level1_total += 1
            token_total += 1
        else:
            level1_no_rule += 1

    accuracy = token_correct / token_total * 100 if token_total > 0 else 0
    print(f"  Token {t_idx + 1}: {token_correct}/{token_total} correct ({accuracy:.1f}%)")

level1_accuracy = level1_correct / level1_total * 100 if level1_total > 0 else 0
print(f"\n✅ LEVEL 1 RESULT: {level1_correct}/{level1_total} = {level1_accuracy:.1f}%")
print(f"   (No rule available for {level1_no_rule} transitions)")

# ============================================================
# LEVEL 2: Two Layers Back - (Layer, Prev Expert, Current Expert) → Next
# ============================================================
print("\n" + "=" * 70)
print("LEVEL 2: Two Layers Back - (Prev Expert, Current Expert) → Next")
print("=" * 70)

# Build hypothesis from Token 0
level2_rules = {}  # (layer, prev_expert, expert) → predicted_next

print("\n📝 Building rules from Token 0:")
for i in range(1, len(token0) - 1):
    layer = token0[i]['layer']
    prev_expert = token0[i - 1]['experts'][0]
    expert = token0[i]['experts'][0]
    next_expert = token0[i + 1]['experts'][0]

    key = (layer, prev_expert, expert)
    level2_rules[key] = next_expert
    print(f"  Rule: Layer {layer}, ({prev_expert}, {expert}) → Expert {next_expert}")

print(f"\nTotal rules from Token 0: {len(level2_rules)}")

# Validate on Tokens 1-9
print("\n📊 Validating on Tokens 1-9:")
level2_correct = 0
level2_total = 0
level2_no_rule = 0

for t_idx, journey in enumerate(validation_tokens):
    token_correct = 0
    token_total = 0

    for i in range(1, len(journey) - 1):
        layer = journey[i]['layer']
        prev_expert = journey[i - 1]['experts'][0]
        expert = journey[i]['experts'][0]
        actual_next = journey[i + 1]['experts'][0]

        key = (layer, prev_expert, expert)
        if key in level2_rules:
            predicted = level2_rules[key]
            if predicted == actual_next:
                level2_correct += 1
                token_correct += 1
            level2_total += 1
            token_total += 1
        else:
            level2_no_rule += 1

    accuracy = token_correct / token_total * 100 if token_total > 0 else 0
    print(f"  Token {t_idx + 1}: {token_correct}/{token_total} correct ({accuracy:.1f}%)")

level2_accuracy = level2_correct / level2_total * 100 if level2_total > 0 else 0
print(f"\n✅ LEVEL 2 RESULT: {level2_correct}/{level2_total} = {level2_accuracy:.1f}%")
print(f"   (No rule available for {level2_no_rule} transitions)")

# ============================================================
# LEVEL 3: Three Layers Back
# ============================================================
print("\n" + "=" * 70)
print("LEVEL 3: Three Layers Back - (E_n-2, E_n-1, E_n) → E_n+1")
print("=" * 70)

# Build hypothesis from Token 0
level3_rules = {}

print("\n📝 Building rules from Token 0:")
for i in range(2, len(token0) - 1):
    layer = token0[i]['layer']
    e_prev2 = token0[i - 2]['experts'][0]
    e_prev1 = token0[i - 1]['experts'][0]
    e_curr = token0[i]['experts'][0]
    e_next = token0[i + 1]['experts'][0]

    key = (layer, e_prev2, e_prev1, e_curr)
    level3_rules[key] = e_next
    print(f"  Rule: Layer {layer}, ({e_prev2}, {e_prev1}, {e_curr}) → Expert {e_next}")

print(f"\nTotal rules from Token 0: {len(level3_rules)}")

# Validate on Tokens 1-9
print("\n📊 Validating on Tokens 1-9:")
level3_correct = 0
level3_total = 0
level3_no_rule = 0

for t_idx, journey in enumerate(validation_tokens):
    token_correct = 0
    token_total = 0

    for i in range(2, len(journey) - 1):
        layer = journey[i]['layer']
        e_prev2 = journey[i - 2]['experts'][0]
        e_prev1 = journey[i - 1]['experts'][0]
        e_curr = journey[i]['experts'][0]
        actual_next = journey[i + 1]['experts'][0]

        key = (layer, e_prev2, e_prev1, e_curr)
        if key in level3_rules:
            predicted = level3_rules[key]
            if predicted == actual_next:
                level3_correct += 1
                token_correct += 1
            level3_total += 1
            token_total += 1
        else:
            level3_no_rule += 1

    accuracy = token_correct / token_total * 100 if token_total > 0 else 0
    print(f"  Token {t_idx + 1}: {token_correct}/{token_total} correct ({accuracy:.1f}%)")

level3_accuracy = level3_correct / level3_total * 100 if level3_total > 0 else 0
print(f"\n✅ LEVEL 3 RESULT: {level3_correct}/{level3_total} = {level3_accuracy:.1f}%")
print(f"   (No rule available for {level3_no_rule} transitions)")

# ============================================================
# LEVEL 4: Pair-Based - (Layer, Primary, Secondary) → Next Primary
# ============================================================
print("\n" + "=" * 70)
print("LEVEL 4: Pair-Based - (Layer, Primary, Secondary) → Next Primary")
print("=" * 70)

# Build hypothesis from Token 0
level4_rules = {}

print("\n📝 Building rules from Token 0:")
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    primary = token0[i]['experts'][0]
    secondary = token0[i]['experts'][1]
    next_primary = token0[i + 1]['experts'][0]

    key = (layer, primary, secondary)
    level4_rules[key] = next_primary
    print(f"  Rule: Layer {layer}, Pair ({primary}, {secondary}) → Expert {next_primary}")

print(f"\nTotal rules from Token 0: {len(level4_rules)}")

# Validate on Tokens 1-9
print("\n📊 Validating on Tokens 1-9:")
level4_correct = 0
level4_total = 0
level4_no_rule = 0

for t_idx, journey in enumerate(validation_tokens):
    token_correct = 0
    token_total = 0

    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        primary = journey[i]['experts'][0]
        secondary = journey[i]['experts'][1]
        actual_next = journey[i + 1]['experts'][0]

        key = (layer, primary, secondary)
        if key in level4_rules:
            predicted = level4_rules[key]
            if predicted == actual_next:
                level4_correct += 1
                token_correct += 1
            level4_total += 1
            token_total += 1
        else:
            level4_no_rule += 1

    accuracy = token_correct / token_total * 100 if token_total > 0 else 0
    print(f"  Token {t_idx + 1}: {token_correct}/{token_total} correct ({accuracy:.1f}%)")

level4_accuracy = level4_correct / level4_total * 100 if level4_total > 0 else 0
print(f"\n✅ LEVEL 4 RESULT: {level4_correct}/{level4_total} = {level4_accuracy:.1f}%")
print(f"   (No rule available for {level4_no_rule} transitions)")

# ============================================================
# LEVEL 5: Skip Layers - Layer N → Layer N+K
# ============================================================
print("\n" + "=" * 70)
print("LEVEL 5: Skip Layers - Layer N → Layer N+K (K=2,4,8)")
print("=" * 70)

for skip in [2, 4, 8]:
    print(f"\n--- Skip = {skip} layers ---")

    # Build hypothesis from Token 0
    level5_rules = {}
    token0_dict = {r['layer']: r['experts'][0] for r in token0}

    for layer in range(32 - skip):
        if layer in token0_dict and (layer + skip) in token0_dict:
            key = (layer, token0_dict[layer])
            level5_rules[key] = token0_dict[layer + skip]

    print(f"  Rules from Token 0: {len(level5_rules)}")

    # Validate
    level5_correct = 0
    level5_total = 0
    level5_no_rule = 0

    for journey in validation_tokens:
        j_dict = {r['layer']: r['experts'][0] for r in journey}

        for layer in range(32 - skip):
            if layer in j_dict and (layer + skip) in j_dict:
                key = (layer, j_dict[layer])
                if key in level5_rules:
                    predicted = level5_rules[key]
                    actual = j_dict[layer + skip]
                    if predicted == actual:
                        level5_correct += 1
                    level5_total += 1
                else:
                    level5_no_rule += 1

    level5_accuracy = level5_correct / level5_total * 100 if level5_total > 0 else 0
    print(f"  ✅ Skip-{skip} RESULT: {level5_correct}/{level5_total} = {level5_accuracy:.1f}%")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("📊 FINAL COMPARISON - ALL LEVELS")
print("=" * 70)

print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│ Level │ Method                          │ Accuracy │ Rules │ Tested │
├──────────────────────────────────────────────────────────────────────┤
│   1   │ (Layer, Expert) → Next          │  {level1_accuracy:>5.1f}%  │  {len(level1_rules):>3}  │  {level1_total:>4}  │
│   2   │ (Layer, Prev, Curr) → Next      │  {level2_accuracy:>5.1f}%  │  {len(level2_rules):>3}  │  {level2_total:>4}  │
│   3   │ (Layer, E-2, E-1, E) → Next     │  {level3_accuracy:>5.1f}%  │  {len(level3_rules):>3}  │  {level3_total:>4}  │
│   4   │ (Layer, Primary, Secondary)     │  {level4_accuracy:>5.1f}%  │  {len(level4_rules):>3}  │  {level4_total:>4}  │
├──────────────────────────────────────────────────────────────────────┤
│       │ Random Baseline (1/8)           │  12.5%   │   -   │   -    │
└──────────────────────────────────────────────────────────────────────┘
""")

# Find best level
accuracies = {
    'Level 1': level1_accuracy,
    'Level 2': level2_accuracy,
    'Level 3': level3_accuracy,
    'Level 4': level4_accuracy,
}

best_level = max(accuracies, key=accuracies.get)
print(f"🏆 BEST: {best_level} with {accuracies[best_level]:.1f}% accuracy")

print(f"""
📌 INTERPRETATION:
- Random guess = 12.5% (1 out of 8 experts)
- If accuracy > 12.5%, the pattern has SOME predictive power
- If accuracy > 25%, pattern is 2x better than random
- If accuracy > 50%, pattern is USEFUL for pre-loading

⚠️ NOTE: Rules from Token 0 may not match Token 1-9
         "No rule available" means the exact pattern wasn't in Token 0
""")
