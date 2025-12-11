# Hyperparameter Optimization Guide

## Quick Answer to Your Questions

### Q1: Is Bayesian optimization just for architecture, not training?

**No! It optimizes BOTH:**

| Category | Parameters Tuned | Fixed (Not Tuned) |
|----------|------------------|-------------------|
| **Training** | • Learning rate (lr)<br>• Dropout rate | • Epochs (fixed at 10)<br>• Batch size (512)<br>• Optimizer (AdamW) |
| **Architecture** | • Layer embedding dims<br>• Expert embedding dims<br>• Hidden layer sizes<br>• Number of layers | • Input features (178)<br>• Output classes (8)<br>• Activation (LeakyReLU) |

**So we tuned 5 hyperparameters total:**
1. `lr` (training) - How fast model learns
2. `layer_emb` (architecture) - Layer ID representation size
3. `expert_emb` (architecture) - Expert ID representation size  
4. `architecture` (architecture) - Hidden layer neuron counts
5. `dropout` (training) - Regularization strength

**NOT tuned (kept constant):**
- Epochs: Fixed at 10 (for speed - normally would tune this too)
- Batch size: 512
- Optimizer: AdamW
- Loss function: CrossEntropyLoss

---

## What Each Hyperparameter Does

### Training Hyperparameters

#### 1. Learning Rate (lr) - Currently: 0.00054

**What it controls:** Speed of weight updates during training

**Training process:**
```python
for epoch in range(10):  # ← Epochs (fixed)
    for batch in data_loader:  # ← Batch size (fixed)
        # Forward pass
        predictions = model(batch)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()  # Calculate gradients
        
        # Update weights
        optimizer.step()  # ← Uses lr to determine step size
```

**Impact of lr:**
- Too high (0.01): Weights jump around, never converge
- Too low (0.0001): Training is too slow, might not finish in 10 epochs
- Optimal (0.00054): Converges in 10 epochs to best accuracy

#### 2. Dropout - Currently: 0.18

**What it controls:** Percentage of neurons randomly disabled during training

**During training:**
```python
# Each forward pass:
hidden = [n1, n2, n3, ..., n512]  # 512 neurons

# Dropout randomly disables 18%:
active = [n1, 0, n3, 0, n5, ..., n512]  # ~92 neurons set to 0

# Forces model to work with any subset of neurons
# Prevents relying too much on specific neurons → Less overfitting
```

**During testing:**
```python
# All neurons active, but scale outputs by (1 - dropout)
# Compensates for having more neurons than during training
```

---

### Architecture Hyperparameters

#### 3. Layer Embedding (layer_emb) - Currently: 5

**What it controls:** How many numbers represent each layer ID

**Your data:** `layer_id` can be 0, 1, 2, ..., 31 (categorical, not numeric)

**How embedding works:**

```python
# Step 1: Create embedding table (learned during training)
layer_embedding = nn.Embedding(num_embeddings=32, embedding_dim=5)

# This creates a lookup table: 32 rows × 5 columns
# Initially random:
#         [dim0   dim1   dim2   dim3   dim4]
# Layer 0: [0.23, -0.45,  0.67, -0.12,  0.89]
# Layer 1: [0.34,  0.78, -0.23,  0.45, -0.67]
# ...
# Layer 31:[0.56, -0.89,  0.12,  0.78, -0.34]

# Step 2: During forward pass
layer_id = 15  # Your input

# Look up row 15 from the table:
layer_vector = layer_embedding(layer_id)
# Returns: [0.12, 0.45, -0.67, 0.34, 0.89]  ← 5 numbers

# Step 3: During training
# These 5 numbers are updated via backpropagation
# Model learns: "Layer 15 should have this specific 5D representation"
```

**Why 5 dimensions specifically?**
- Baseline tried 10 (my guess - maybe overkill?)
- Bayesian found 5 works better (simpler, less overfitting)
- Each dimension captures different "layer properties"
  - Dim 0 might learn: "early vs late layer"
  - Dim 1 might learn: "routing complexity at this layer"
  - Dim 2-4: Other learned patterns

#### 4. Expert Embedding (expert_emb) - Currently: 7

**Same mechanism as layer embedding:**

```python
expert_embedding = nn.Embedding(num_embeddings=9, embedding_dim=7)
# 9 = 8 experts (0-7) + 1 padding (0)

# YOUR DATA has expert IDs in TWO places:
# 1. Secondary expert: Single ID
secondary_expert = 5
sec_vector = expert_embedding(secondary_expert + 1)  # +1 for padding offset
# Returns: [0.23, -0.67, 0.45, -0.89, 0.12, 0.56, -0.34]  ← 7 numbers

# 2. History: 32 expert IDs
history = [2, 5, 5, 1, 7, 3, ..., 0, 0, 0]  # 32 IDs (0 = padding)
hist_vectors = expert_embedding(history)
# Returns: 32 vectors × 7 dims = 32×7 matrix
# Then flattened: [e0_d0, e0_d1, ..., e31_d6] → 224 numbers

# TOTAL from expert embeddings: 7 + 224 = 231 numbers
```

**Why 7 dimensions?**
- More than layer (7 vs 5) because expert relationships are complex
- Model needs to learn: "Expert 2 after Expert 5 is common"
- 7 dimensions can capture these patterns

#### 5. Architecture [512, 512] - Currently: Two layers with 512 neurons each

**Complete model structure:**

```python
# INPUT: Your data sample
layer_id = 15
secondary = 6
history = [2,5,1,7,3,...]  # 32 IDs
gating = [0.77, 0.23]
position = 0.05

# STEP 1: Convert to embeddings
layer_emb = layer_embedding(15)          → 5 numbers
sec_emb = expert_embedding(6+1)          → 7 numbers  
hist_emb = expert_embedding(history).flatten() → 224 numbers
# gating and position stay as-is              → 3 numbers

# STEP 2: Concatenate all features
input_vector = [layer_emb (5) + sec_emb (7) + hist_emb (224) + gating (2) + pos (1)]
# Total: 5+7+224+2+1 = 239 numbers... wait no, history is 32*5 based on baseline

# Actually with optimal config:
# - Layer: 5 dims
# - Secondary: 7 dims
# - History: 32 experts × 7 dims = 224 dims
# - Gating: 2 dims
# - Position: 1 dim
# Total input: 5 + 7 + 224 + 2 + 1 = 239 dimensions

# STEP 3: First hidden layer (512 neurons)
hidden1 = Linear(239 → 512)(input_vector)
hidden1 = BatchNorm(hidden1)
hidden1 = LeakyReLU(hidden1)
hidden1 = Dropout(0.18)(hidden1)  # ← Randomly zero out 18%

# STEP 4: Second hidden layer (512 neurons)  
hidden2 = Linear(512 → 512)(hidden1)
hidden2 = BatchNorm(hidden2)
hidden2 = LeakyReLU(hidden2)
hidden2 = Dropout(0.18)(hidden2)

# STEP 5: Output layer (8 experts)
output = Linear(512 → 8)(hidden2)
# Returns: [p0, p1, p2, p3, p4, p5, p6, p7]  ← Probabilities for each expert
```

**Why [512, 512] instead of [256, 128, 64]?**

```
Baseline (deep pyramid):
239 → [256] → [128] → [64] → 8
Pros: Gradually compresses information
Cons: Information loss at each narrow layer, harder to train

Optimal (wide shallow):
239 → [512] → [512] → 8
Pros: More neurons = more pattern capacity, less information loss
Cons: More parameters (but we have 1.34M samples, so no overfitting)
```

---

## Embedding Mechanics - Complete Example

**Your input data sample:**
```python
{
  'layer': 15,
  'secondary': 6,
  'history': [2, 5, 5, 1, 7, 3, 4, 2, 6, 1, 5, 7, 3, 2, 4, 
              6, 1, 5, 3, 7, 2, 4, 6, 1, 5, 3, 7, 2, 4, 0, 0, 0],
  'gating': [0.77, 0.23],
  'pos': 0.05,
  'target': 3  # Ground truth: should predict expert 3
}
```

**Step-by-step conversion:**

```python
# 1. Layer embedding (5 dims)
layer_id = 15
layer_table = [
    # dim0  dim1   dim2   dim3   dim4  ← Learned values
    [...],  # Layer 0
    [...],  # Layer 1
    ...
    [0.12, 0.45, -0.67, 0.34, 0.89],  # ← Layer 15 (row 15)
    ...
]
layer_vector = layer_table[15] = [0.12, 0.45, -0.67, 0.34, 0.89]

# 2. Secondary expert embedding (7 dims)
secondary = 6
expert_table = [
    # dim0  dim1   dim2   dim3   dim4   dim5   dim6
    [0.00, 0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # Padding (0)
    [0.23, -0.45, 0.67, -0.12,  0.34,  0.56, -0.78],  # Expert 0
    ...
    [0.34,  0.67, -0.23, 0.89, -0.45,  0.12,  0.56],  # ← Expert 6 (row 7, +1 offset)
    ...
]
sec_vector = expert_table[6+1] = [0.34, 0.67, -0.23, 0.89, -0.45, 0.12, 0.56]

# 3. History embedding (32 × 7 = 224 dims)
history = [2, 5, 5, 1, 7, 3, 4, ...]  # 32 values
hist_vectors = [
    expert_table[2+1],   # [e0, e1, e2, e3, e4, e5, e6]  ← Expert 2
    expert_table[5+1],   # [e0, e1, e2, e3, e4, e5, e6]  ← Expert 5
    expert_table[5+1],   # [e0, e1, e2, e3, e4, e5, e6]  ← Expert 5
    ...  # 32 experts total
]
# Flatten: [e2_d0, e2_d1, ..., e2_d6, e5_d0, e5_d1, ..., e0_d6]
# Total: 32 vectors × 7 dims = 224 numbers

# 4. Concatenate everything
final_input = [
    0.12, 0.45, -0.67, 0.34, 0.89,  # layer (5)
    0.34, 0.67, -0.23, 0.89, -0.45, 0.12, 0.56,  # secondary (7)
    ...(224 numbers from history)...,  # history (224)
    0.77, 0.23,  # gating (2)
    0.05  # position (1)
]
# Total: 239 numbers → fed to first hidden layer
```

---

## Bayesian Optimization Results Summary

### What Was Tuned

```python
Search space:
- lr: Range [0.0001, 0.01] (logarithmic scale)
- layer_emb: Range [5, 20] (integer)
- expert_emb: Range [3, 10] (integer)
- architecture: 2-4 layers, each 32-512 neurons
- dropout: Range [0.05, 0.4]
```

### What Was NOT Tuned (Fixed)

```python
Fixed hyperparameters:
- epochs: 10 (for speed - normally would be 30)
- batch_size: 512
- optimizer: AdamW
- activation: LeakyReLU(0.1)
- loss: CrossEntropyLoss
```

### Optimal Configuration Found

```python
Best config (Trial #12):
{
    'lr': 0.000536,         # Training param
    'layer_emb': 5,         # Architecture param
    'expert_emb': 7,        # Architecture param
    'architecture': [512, 512],  # Architecture param
    'dropout': 0.18         # Training param
}

Result: 75.43% Top-2 accuracy
vs Baseline: 70.66% Top-2 accuracy
Improvement: +4.77%
```

### Key Insights

**Pattern discovered:**
- **Wider is better**: 512 neurons > 256 neurons
- **Shallower is better**: 2 layers > 3 layers
- **Small layer embeddings work**: 5 dims is enough for 32 layers
- **Large expert embeddings help**: 7 dims captures expert interactions
- **Lower LR with more data**: 0.00054 works well with 1.34M samples

**Why this configuration wins:**
1. Wide layers (512 neurons) have more capacity to learn complex expert routing patterns
2. Shallow network (2 layers) has less information loss
3. Small layer embedding (5) prevents overfitting on simple layer IDs
4. Large expert embedding (7) captures rich expert relationships
5. Moderate dropout (0.18) balances regularization and learning

---

## Next Steps

**To use the optimal configuration:**

1. **Retrain with full epochs** (30 instead of 10)
2. **Evaluate on test set** to get final accuracy
3. **Compare against baseline** (currently at 71% Top-2 with 95% split)

**Potential further improvements:**
- Top-K loss (optimize Top-2 directly)
- LSTM architecture (capture sequential patterns)
- Feature engineering (add repetition/pattern features)

But first: verify if optimal config with full training beats baseline!
