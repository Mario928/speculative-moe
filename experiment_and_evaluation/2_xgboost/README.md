# 🌲 Experiment 2: XGBoost Baseline

## Overview
This experiment uses XGBoost (gradient boosting trees) as a classical ML baseline.
It converts the routing prediction problem into tabular features and trains a multi-class classifier.

## Files
- `train_xgboost.py` — Trains and evaluates XGBoost model
- `XGBoost_Results.md` — Detailed analysis of results

## How to Run
```bash
cd experiment_and_evaluation/2_xgboost
python train_xgboost.py
```

## Results
| Metric | Value |
|--------|-------|
| Test Top-1 | 36.2% |
| Test Top-2 | 53.3% |
| Training Time | ~11 seconds |

**Conclusion**: XGBoost underperforms because it cannot learn embedding representations for categorical expert IDs.

---
**➡️ Next Step**: See `../3_bayesian_search/` to find the optimal neural network configuration.
