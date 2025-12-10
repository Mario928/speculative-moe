
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
files = ['token_1_routing.jsonl', 'token_2_routing.jsonl', 'token_3_routing.jsonl']
data = []

for fpath in files:
    with open(fpath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    pass

# Build Matrix: Layer x Expert -> Frequency
heatmap = np.zeros((32, 8)) # 32 layers, 8 experts

for entry in data:
    layer = entry['layer']
    # Add count for Top-2 experts
    for exp in entry['experts']:
        heatmap[layer][exp] += 1

# Normalize by number of tokens (3) so it's a probability/frequency
heatmap = heatmap / len(files)

# Plot
plt.figure(figsize=(12, 10))
plt.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')

plt.title('Mixtral 8x7B Expert Activation Frequency per Layer (Averaged over 3 Tokens)')
plt.xlabel('Expert ID')
plt.ylabel('Layer Index')
plt.xticks(range(8))
plt.yticks(range(0, 32, 2))
plt.colorbar(label='Activation Frequency')

# Annotate with values
for i in range(32):
    for j in range(8):
        if heatmap[i, j] > 0.5: # Only show high prob
            plt.text(j, i, f'{heatmap[i, j]:.1f}', ha='center', va='center', color='white', fontsize=8)

plt.tight_layout()
plt.savefig('layer_expert_heatmap.png')
print("Heatmap saved to layer_expert_heatmap.png")
