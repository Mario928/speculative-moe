
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import itertools

# Load Data
def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return pd.DataFrame(data)

t1 = load_data('token_1_routing.jsonl')
t2 = load_data('token_2_routing.jsonl')

print(f"Loaded Token 1 ({len(t1)} layers) and Token 2 ({len(t2)} layers)")

# ---------------------------------------------------------
# Iteration 5: Expert Pair Co-occurrence (Affinity Clusters)
# Hypothesis: Experts X and Y are "Friends". If X is used, Y is likely used in SAME layer.
# ---------------------------------------------------------
print("\n--- Iteration 5: Expert Pair Co-occurrence ---")
pair_counts = Counter()
for _, row in t1.iterrows():
    pair = tuple(sorted(row['experts']))
    pair_counts[pair] += 1

print("Top Expert Pairs (Same Layer) in T1:")
for p, c in pair_counts.most_common(5):
    print(f"  {p}: {c} times")

# Validation on T2
print("\nValidation on T2 (Do these pairs hold?):")
t2_pairs = [tuple(sorted(x)) for x in t2['experts']]
hits = 0
top_pair = pair_counts.most_common(1)[0][0]
for p in t2_pairs:
    if p == top_pair: hits += 1
print(f"  Top pair {top_pair} appears {hits} times in T2 (out of {len(t2)})")


# ---------------------------------------------------------
# Iteration 6: High-Confidence Gating Sub-graph
# Hypothesis: When gating prob > 0.8, the routing is deterministic/predictable?
# ---------------------------------------------------------
print("\n--- Iteration 6: High-Confidence Routing ---")
high_conf_t1 = t1[t1['gating_probs'].apply(lambda x: x[0] > 0.8)]
print(f"High Confidence Layers in T1: {len(high_conf_t1)}/{len(t1)}")

# Check if High Confidence decisions are identifiable by Layer?
# i.e. Are some layers ALWAYS high confidence?
high_conf_layers = set(high_conf_t1['layer'])
print(f"Layers with >0.8 confidence in T1: {sorted(list(high_conf_layers))}")

t2_high_conf = t2[t2['gating_probs'].apply(lambda x: x[0] > 0.8)]
t2_layers = set(t2_high_conf['layer'])
common_confident_layers = high_conf_layers.intersection(t2_layers)
print(f"Layers consistently high confidence in T2: {sorted(list(common_confident_layers))}")
print(f"Overlap Count: {len(common_confident_layers)}")

# ---------------------------------------------------------
# Iteration 7: Skip-Layer Logic (Layer N -> Layer N+2)
# Hypothesis: Maybe connection is ResNet style, skipping a layer.
# ---------------------------------------------------------
print("\n--- Iteration 7: Skip-Layer Logic (N -> N+2) ---")
skip_trans = defaultdict(Counter)
for i in range(len(t1) - 2):
    curr = t1.iloc[i]['experts'][0]
    skip = t1.iloc[i+2]['experts'][0]
    skip_trans[curr][skip] += 1

# Check predictability
correct = 0
total = 0
for i in range(len(t2) - 2):
    curr = t2.iloc[i]['experts'][0]
    actual_skip = t2.iloc[i+2]['experts'][0]
    
    if skip_trans[curr]:
        pred = skip_trans[curr].most_common(1)[0][0]
        if pred == actual_skip: correct += 1
    total += 1
print(f"Skip-Layer Prediction Accuracy: {correct/total:.2%}")

