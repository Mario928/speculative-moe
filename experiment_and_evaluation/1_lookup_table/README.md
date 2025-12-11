# 📋 Experiment 1: Lookup Table (Frequency-Based Rules)

## Overview
This experiment extracts simple prediction rules by counting patterns in the training data.
For each pattern (e.g., "Layer 5 + Expert 3"), it predicts the most frequently observed next expert.

## Files
- `create_rules_from_pt.py` — Extracts rules from training data
- `evaluate_rules_from_pt.py` — Evaluates rules on test data
- `rules.json` — Extracted rules (created by create script)
- `evaluation_results.json` — Test set results (created by evaluate script)

## How to Run
```bash
cd experiment_and_evaluation/1_lookup_table
python create_rules_from_pt.py    # Creates rules.json
python evaluate_rules_from_pt.py  # Creates evaluation_results.json
```

## Results
| Level | Description | Top-1 | Top-2 |
|-------|-------------|-------|-------|
| 1 | (Layer, Expert) | 25.8% | 44.5% |
| 2 | (Layer, Prev, Curr) | 33.4% | 52.8% |
| **3** | **(Layer, E-2, E-1, E)** | **39.0%** | **58.2%** |
| 4 | (Layer, Primary, Secondary) | 31.0% | 50.4% |
| 5 | (Layer, Full Path) | 42.2% | 56.2% |

**Best**: Level 3 achieves 58.2% Top-2 accuracy.

---
**➡️ Next Step**: See `../2_xgboost/` for a classical ML baseline comparison.
