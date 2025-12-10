
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# 1. Load and Split Data
print("Loading humaneval_1_routing.jsonl...")
data = []
with open('humaneval_1_routing.jsonl', 'r') as f:
    for line in f:
        if line.strip(): data.append(json.loads(line))

df = pd.DataFrame(data)
print(f"Total Rows: {len(df)}")
print(f"Unique Problems: {df['problem_id'].unique()}")

# Split by Problem
problems = {}
for pid in df['problem_id'].unique():
    problems[pid] = df[df['problem_id'] == pid]
    print(f"Problem {pid}: {len(problems[pid])} rows")

# Use Problem 0 as Primary (Train/Test on its tokens)
# We need to separate tokens. `token_idx` resets?
# A "Sequence" is a contiguous run of increasing token_idx.
# Or does token_idx go 0..N for one seq, then 0..M for next?
# Let's count sequences.
p0 = problems[0].copy()
p0['seq_id'] = (p0['token_idx'] < p0['token_idx'].shift(1)).cumsum() # New seq when idx drops
print(f"Problem 0 has {p0['seq_id'].nunique()} distinct sequences (token generations).")

# Select Seq 0 as Train, Seq 1 as Test (if available)
seq_ids = p0['seq_id'].unique()
if len(seq_ids) < 2:
    print("Warning: Only 1 sequence in Problem 0. splitting by halves.")
    train_df = p0.iloc[:len(p0)//2]
    test_df = p0.iloc[len(p0)//2:]
else:
    train_df = p0[p0['seq_id'] == seq_ids[0]]
    test_df = p0[p0['seq_id'] == seq_ids[1]]

print(f"Train Set: {len(train_df)} rows, Test Set: {len(test_df)} rows")

# ---------------------------------------------------------
# Iteration 8: Confident Static Layer
# Hypothesis: If Layer L chooses Expert E with Prob > 0.8 in Train, it ALWAYS chooses E in Test.
# ---------------------------------------------------------
print("\n--- Iteration 8: Confident Static Layer ---")
# Find (Layer, Expert) pairs with avg prob > 0.8 in Train
strong_rules = {} # (Layer) -> Expert
for layer in train_df['layer'].unique():
    layer_data = train_df[train_df['layer'] == layer]
    # Check if any expert is dominant
    # Avg prob for primary expert
    primary_experts = [x[0] for x in layer_data['experts']]
    # freq
    counts = Counter(primary_experts)
    most_common = counts.most_common(1)[0]
    
    # "Strong" if frequency > 90%
    if most_common[1] / len(layer_data) > 0.9:
        strong_rules[layer] = most_common[0]

print(f"Found Strong Rules for {len(strong_rules)} Layers: {strong_rules}")

# Validate on Test
hits = 0
total_strong = 0
for _, row in test_df.iterrows():
    l = row['layer']
    if l in strong_rules:
        total_strong += 1
        if row['experts'][0] == strong_rules[l]:
            hits += 1

acc = hits/total_strong if total_strong else 0
print(f"Validation Accuracy on Strong Layers: {acc:.2%} ({hits}/{total_strong})")


# ---------------------------------------------------------
# Iteration 9: Diagonal Flow (Time-Space)
# Hypothesis: Expert(L, T) correlates with Expert(L+1, T+1)
# ---------------------------------------------------------
print("\n--- Iteration 9: Diagonal Flow ---")
# Need grid: Layer x Token
# Pivot Train DF
try:
    grid = train_df.pivot(index='layer', columns='token_idx', values='experts')
    # grid[l, t] is list [e1, e2]
    
    diag_hits = 0
    diag_total = 0
    
    # Check L, T -> L+1, T+1
    for l in range(31):
        for t in range(len(grid.columns) - 1):
            if t+1 not in grid.columns: continue
            
            curr = grid.iloc[l, t]   # List [e1, e2] at L, T
            next_diag = grid.iloc[l+1, t+1] # at L+1, T+1
            
            if isinstance(curr, list) and isinstance(next_diag, list):
                # Simple check: Does Top-1 match?
                if curr[0] == next_diag[0]:
                    diag_hits += 1
                diag_total += 1
    
    print(f"Diagonal (L->L+1, T->T+1) Consistency: {diag_hits/diag_total:.2%}")

except Exception as e:
    print(f"Skipping Iter 9 due to grid error: {e}")

# ---------------------------------------------------------
# Iteration 10: "Mod 4" Pattern
# Hypothesis: Experts repeat every 4 layers?
# ---------------------------------------------------------
print("\n--- Iteration 10: Modulo-4 Pattern ---")
# Check correaltion between Layer L and Layer L+4
mod_hits = 0
mod_total = 0

for _, row in train_df.iterrows():
    l = row['layer']
    t = row['token_idx']
    
    # Find L+4 for same token
    future = train_df[(train_df['token_idx'] == t) & (train_df['layer'] == l + 4)]
    if not future.empty:
        curr_exp = row['experts'][0]
        next_exp = future.iloc[0]['experts'][0]
        if curr_exp == next_exp:
            mod_hits += 1
        mod_total += 1

print(f"Layer N -> Layer N+4 Consistency: {mod_hits/mod_total:.2%}")
