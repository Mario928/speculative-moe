# Optimal MLP Configuration Details

## Performance Summary
**Validation Top-2 Accuracy**: ~77.9% (at Epoch 30)
**Test Top-2 Accuracy**: 66.95% (Held-out problems)
**Improvement over Baseline**: +4.8% (Top-1)

*Note: The gap between Validation (77.9%) and Test (66.95%) suggests some routing patterns are problem-specific. However, the optimized model consistently outperforms the baseline by ~5% on all metrics.*

---

## 1. Model Architecture

### Input Layer (239 dimensions)
**Source**: Data Pipeline + Feature Engineering
**Detail**: 
The input consists of concatenated features from embeddings and raw values.
- **Layer ID**: Embedded to **5 dimensions** (Found via Bayesian Search)
- **Secondary Expert**: Embedded to **7 dimensions** (Found via Bayesian Search)
- **Expert History**: 32 experts × **7 dimensions** = **224 dimensions** (Derived from expert embedding size)
- **Gating Probabilities**: 2 raw floats (Fixed - data feature)
- **Token Position**: 1 raw float (Fixed - data feature)

**Why 5 and 7 dimensions?**
Bayesian optimization found these specific values were optimal.
- **Layer ID (5 dims)**: Smaller than our initial guess (10). Likely enough to capture simpler layer patterns (early vs late).
- **Expert ID (7 dims)**: Larger than our initial guess (5). Suggests expert relationships are complex and need more capacity to represent interactions.

### Hidden Layers (Wide & Shallow)
**Configuration**: [512, 512]
**Source**: Bayesian Optimization (Trial #12)

**Structure**:
1. **Hidden Layer 1**: Linear(239 → 512) → BatchNorm → LeakyReLU → Dropout
2. **Hidden Layer 2**: Linear(512 → 512) → BatchNorm → LeakyReLU → Dropout

**Why this shape?**
The optimization favored a **wider, shallower** network over the deep pyramid baseline ([256, 128, 64]).
- **Width (512)**: Allows the model to learn more patterns in parallel at each layer.
- **Depth (2 layers)**: Sufficient for the task complexity; deeper networks (3-4 layers) didn't improve performance and were harder to train.

### Output Layer
**Structure**: Linear(512 → 8)
**Source**: Problem Definition (Fixed)
**Detail**: Produces logits for the 8 possible experts.

### Activation Function
**Type**: LeakyReLU (Negative slope = 0.1)
**Source**: Logical Choice / Industry Standard
**Reason**: Generally preferred over ReLU for deeper networks to prevent "dying neurons" (where neurons stop learning because they output 0). Fixed during search.

---

## 2. Training Parameters

### Learning Rate
**Value**: 0.000536
**Source**: Bayesian Optimization
**Context**: Lower than the standard default of 0.001. With a large dataset (1.34M samples), a slower learning rate allows for more precise weight updates and better convergence.

### Dropout Rate
**Value**: 0.18 (18%)
**Source**: Bayesian Optimization
**Context**: A moderate amount of regularization. Prevents the model from memorizing the specific training examples (overfitting) without hindering learning too much.

### Batch Size
**Value**: 512
**Source**: Industry Standard / Hardware Efficiency
**Reason**: Large enough to provide stable gradient estimates, small enough to fit comfortably in GPU memory. Kept fixed to speed up the search.

### Optimizer
**Type**: AdamW
**Source**: Industry Standard
**Reason**: Generally works best for training neural networks. Handles weight decay correctly compared to standard Adam. Kept fixed.

### epochs
**Value**: 30 (for final training)
**Source**: Practical Decision
**Reason**: During search, we used 10 epochs for speed. For the final model, we train longer to ensure full convergence, using early stopping to prevent overfitting if it stabilizes sooner.

---

## 3. How We Got Here

### The Process: Bayesian Optimization
We didn't just guess these values. We used an algorithm called **Tree-structured Parzen Estimator (TPE)** via the Optuna library.

1.  **Defined a Search Space**: complex ranges for learning rate, embeddings, layers, units, and dropout.
2.  **Ran 20 Trials**: The algorithm intelligently tested configurations.
    -   *Trial 0*: Tried high learning rate → Failed.
    -   *Trial 5*: Tried deep network → Mediocre results.
    -   *Trial 12*: Tried wide/shallow network with lower LR → **Success!**
3.  **Result**: The algorithm automatically converged on the settings above as the mathematically most promising region for this specific dataset.

### Why "Fixed" Parameters weren't searched?
Parameters like batch size and optimizer were fixed based on **industry best practices** to save computing time. Searching them usually yields diminishing returns compared to architecture and learning rate tuning.
