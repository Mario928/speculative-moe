# 🧠 Experiment 4: Optimized MLP (Final Model)

## Overview
This is the **final, optimized neural network** for expert routing prediction.
The architecture was determined by Bayesian hyperparameter search (see `3_bayesian_search/`).

## Files
- `train_mlp.py` — Trains the optimized MLP
- `evaluate_detailed.py` — Evaluates on test set with per-layer breakdown
- `mlp_model.py` — Model architecture definition
- `mlp_dataset.py` — PyTorch dataset loader
- `checkpoints/` — Saved model weights
- `Best_Model_Details.md` — Technical details of the optimal configuration

## How to Run
```bash
cd experiment_and_evaluation/4_neural_network_mlp

# Train (takes ~10 minutes)
python train_mlp.py

# Evaluate on test set
python evaluate_detailed.py
```

## Architecture
```
Input Features:
  - Layer embedding (5 dims)
  - Expert history embeddings (7 dims × 32 layers)
  - Secondary expert embedding (7 dims)
  - Gating probabilities (2 dims)
  - Token position (1 dim)

Hidden Layers: [512, 512] with BatchNorm + LeakyReLU + Dropout(0.18)
Output: 8-class softmax (expert IDs)
```

## Results
| Metric | Value |
|--------|-------|
| **Test Top-1** | **55.60%** |
| **Test Top-2** | **73.24%** |
| Improvement over Random | **+345% (4.5×)** |

This model is the **state-of-the-art** for this task.

## Training Curves
![Training Curves](training_curves.png)
