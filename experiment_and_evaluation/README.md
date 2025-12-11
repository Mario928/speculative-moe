# 🧪 Experiment & Evaluation

This folder contains all experiments for the **Speculative Expert Routing Prediction** project.

## 📁 Folder Structure

```
experiment_and_evaluation/
├── preprocess_data.py          # Generates the processed .pt files
├── processed_data/             # Shared dataset (train/val/test splits)
│
├── 1_lookup_table/             # Step 1: Baseline - Simple frequency rules
├── 2_xgboost/                  # Step 2: Baseline - Classical ML (trees)
├── 3_bayesian_search/          # Step 3: Find optimal neural network config
└── 4_neural_network_mlp/       # Step 4: Train & evaluate final model
```

## 🚀 Quick Start

### 1. Generate Data (if not already done)
```bash
cd experiment_and_evaluation
python preprocess_data.py
```
This creates `processed_data/train_data.pt`, `val_data.pt`, and `test_data.pt`.

### 2. Run Experiments

| Experiment | Command |
|------------|---------|
| **Lookup Table** | `cd 1_lookup_table && python create_rules_from_pt.py && python evaluate_rules_from_pt.py` |
| **XGBoost** | `cd 2_xgboost && python train_xgboost.py` |
| **Bayesian Search** | `cd 3_bayesian_search && python bayesian_search.py` |
| **Neural Network** | `cd 4_neural_network_mlp && python train_mlp.py && python evaluate_detailed.py` |

## 📊 Results Summary

See `Final_Experiment_Evaluation_Summary.md` for detailed analysis.

## 🔗 Related Folders

- `../routing_data_collected/` — Raw routing data from model profiling
- `../profiling_moe_code/` — Code used to collect routing data
