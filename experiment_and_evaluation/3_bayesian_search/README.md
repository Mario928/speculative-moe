# 🔍 Experiment 3: Bayesian Hyperparameter Search

## Overview
This experiment uses **Optuna** to automatically find the optimal MLP configuration.
It searches over learning rate, embedding dimensions, hidden layer sizes, and dropout.

## Files
- `bayesian_search.py` — Runs the Optuna study (20 trials)
- `optuna_study.pkl` — Saved Optuna study for analysis
- `Hyperparameter_Guide.md` — Detailed explanation of hyperparameters

## How to Run
```bash
cd experiment_and_evaluation/3_bayesian_search
python bayesian_search.py
```

## Best Configuration Found
| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 0.000536 |
| Layer Embedding Dim | 5 |
| Expert Embedding Dim | 7 |
| Hidden Layers | [512, 512] |
| Dropout | 0.18 |

**Result**: This configuration achieved **75.4% Validation Top-2** during search.
The final model was trained for 30 epochs in `4_neural_network_mlp/`.

## How to Reload & Inspect Study
```python
import joblib
study = joblib.load('optuna_study.pkl')

# View best trial
print(study.best_trial.value)        # Best accuracy: 75.43%
print(study.best_trial.params)       # Best hyperparameters

# List all trials
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value:.2f}%")
```

---
**➡️ Next Step**: See `../4_neural_network_mlp/` for the final trained model using this configuration.

