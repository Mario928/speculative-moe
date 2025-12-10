
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load Sorted Data
data = []
with open('humaneval_1_routing.jsonl', 'r') as f:
    for line in f:
        if line.strip(): data.append(json.loads(line))

df = pd.DataFrame(data).sort_values(by=['token_idx', 'layer'])
unique_tokens = df['token_idx'].unique()
train_tokens = unique_tokens[:len(unique_tokens)//2]
train_df = df[df['token_idx'].isin(train_tokens)]

print("\n--- Final Analysis: Expert Distribution by Layer ---")
# Objective: Check if experts are segregated by depth.
# e.g. Experts 0-3 in Layers 0-15, Experts 4-7 in Layers 16-31?

layer_dist = np.zeros((32, 8))
for _, row in train_df.iterrows():
    l = row['layer']
    e = row['experts'][0] # Top 1
    layer_dist[l, e] += 1

# Normalize rows
row_sums = layer_dist.sum(axis=1)
layer_probs = layer_dist / row_sums[:, np.newaxis]

# Print "Dominant Expert" for each layer
print("Layer | Dominant Experts (>20% freq)")
print("---|---")
for l in range(32):
    # Get experts > 0.2
    exps = np.where(layer_probs[l] > 0.2)[0]
    # formatted
    exps_str = ", ".join([f"E{e} ({layer_probs[l, e]:.0%})" for e in exps])
    print(f"{l:02d} | {exps_str}")

# Check for "Role Segregation"
# Calculate Average Layer Depth for each Expert
print("\n--- Expert Depth Roles ---")
expert_depths = defaultdict(list)
for _, row in train_df.iterrows():
    e = row['experts'][0]
    expert_depths[e].append(row['layer'])

for e in range(8):
    depths = expert_depths[e]
    avg_d = np.mean(depths) if depths else 0
    std_d = np.std(depths) if depths else 0
    print(f"Expert {e}: Avg Depth {avg_d:.1f} (std {std_d:.1f})")

