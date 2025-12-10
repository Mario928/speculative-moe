
import json
import pandas as pd
import numpy as np
from collections import Counter

# 1. Load and Fix Data Sorting
print("Loading and Reshaping Data...")
data = []
with open('humaneval_1_routing.jsonl', 'r') as f:
    for line in f:
        if line.strip(): data.append(json.loads(line))

df = pd.DataFrame(data)
# Sort by Token, then Layer to get "Token Journeys"
df = df.sort_values(by=['token_idx', 'layer'])

print(f"Total Rows: {len(df)}")
unique_tokens = df['token_idx'].unique()
print(f"Unique Tokens: {len(unique_tokens)} (IDs: {unique_tokens[:5]} ... {unique_tokens[-5:]})")

# Split Train/Test by Token ID
# First 50% tokens = Train, Last 50% = Test
split_idx = len(unique_tokens) // 2
train_tokens = unique_tokens[:split_idx]
test_tokens = unique_tokens[split_idx:]

train_df = df[df['token_idx'].isin(train_tokens)]
test_df = df[df['token_idx'].isin(test_tokens)]

print(f"Train Tokens: {len(train_tokens)}, Rows: {len(train_df)}")
print(f"Test Tokens: {len(test_tokens)}, Rows: {len(test_df)}")

# ---------------------------------------------------------
# Iteration 8: Confident Static Layer (Revised)
# Hypothesis: If Layer L chooses E with high prob in Train, does it hold in Test?
# ---------------------------------------------------------
print("\n--- Iteration 8: Confident Static Layer (Revised) ---")
strong_rules = {} 
confidence_threshold = 0.8
rule_hits = 0
rule_candidates = 0

for layer in range(32):
    l_data = train_df[train_df['layer'] == layer]
    if l_data.empty: continue
    
    # Check if any expert is chosen > 80% of time AND has high prob?
    # Or just check if "When Confidence > 0.8, Expert is X"
    
    # Filter for high confidence rows
    high_conf_rows = l_data[l_data['gating_probs'].apply(lambda x: x[0] > confidence_threshold)]
    
    if len(high_conf_rows) > 10: # Minimum support
        # What expert did they pick?
        expert_counts = Counter([x[0] for x in high_conf_rows['experts']])
        top_expert, count = expert_counts.most_common(1)[0]
        
        # Consistency
        consistency = count / len(high_conf_rows)
        if consistency > 0.95:
            strong_rules[layer] = top_expert
            print(f"  Layer {layer}: High Conf (>0.8) -> Exclusively Expert {top_expert} (freq {consistency:.2%})")

print(f"Found Strong High-Conf Rules for {len(strong_rules)} Layers")

# Validate on Test
if strong_rules:
    correct = 0
    total_applicable = 0
    for _, row in test_df.iterrows():
        l = row['layer']
        if l in strong_rules and row['gating_probs'][0] > confidence_threshold:
            total_applicable += 1
            if row['experts'][0] == strong_rules[l]:
                correct += 1
    
    acc = correct/total_applicable if total_applicable else 0
    print(f"Validation of Rules on Test: {acc:.2%} ({correct}/{total_applicable})")
else:
    print("No strong rules found.")

# ---------------------------------------------------------
# Iteration 9: Diagonal Flow (Time-Space)
# Hypothesis: Expert(L, T) == Expert(L+1, T+1)
# ---------------------------------------------------------
print("\n--- Iteration 9: Diagonal Flow ---")
# Pivot: Rows=Layer, Cols=TokenIdx
grid = train_df.pivot(index='layer', columns='token_idx', values='experts')

diag_match = 0
diag_total = 0

# For each diagonal step
for t in train_tokens[:-1]:
    for l in range(31): # 0..30
        try:
            curr = grid.at[l, t]
            next_diag = grid.at[l+1, t+1]
            
            if isinstance(curr, list) and isinstance(next_diag, list):
                if curr[0] == next_diag[0]:
                    diag_match += 1
                diag_total += 1
        except:
            pass

print(f"Diagonal Consistency (L,T -> L+1,T+1): {diag_match/diag_total if diag_total else 0:.2%}")

# ---------------------------------------------------------
# Iteration 10: Modulo-4 Pattern
# Hypothesis: Layer L predicts Layer L+4 (Top-1 Expert Same)
# ---------------------------------------------------------
print("\n--- Iteration 10: Modulo-4 Pattern ---")
# Reset df sort just in case
train_train = train_df.sort_values(['token_idx', 'layer'])

mod_hits = 0
mod_total = 0

for t in train_tokens:
    t_data = train_train[train_train['token_idx'] == t].set_index('layer')
    for l in range(28): # 0..27 (so l+4 <= 31)
        if l in t_data.index and (l+4) in t_data.index:
            e1 = t_data.at[l, 'experts'][0]
            e2 = t_data.at[l+4, 'experts'][0]
            
            if e1 == e2:
                mod_hits += 1
            mod_total += 1

print(f"Mod-4 Consistency (Layer L == Layer L+4): {mod_hits/mod_total if mod_total else 0:.2%}")

