# Routing Data Collected

MoE routing decisions collected from **Mixtral-8x7B-Instruct-v0.1-FP8** during inference.

## Data Files

| File | Dataset | Problems | Routing Records |
|------|---------|----------|-----------------|
| `humaneval_full_routing.jsonl` | HumanEval | 164 | 952,217 |
| `gsm8k_full_routing.jsonl` | GSM8K | 80 | 480,096 |
| `sample/` | Test samples | 2 each | - |

**Total: 1.43M routing records** from 244 problems.

## Train/Val/Test Split

Data was split **token-wise** (98/1/1 by sample, not by problem):

| Split | Samples | Purpose |
|-------|---------|---------|
| **Train** | 1,345,317 | Model training |
| **Validation** | 13,727 | Hyperparameter tuning |
| **Test** | 13,729 | Final evaluation |

Preprocessed splits are in `experiment_and_evaluation/processed_data/`.

## Data Format

Each line in the routing JSONL files:

```json
{"dataset": "gsm8k", "problem_id": 0, "layer": 0, "experts": [7, 0], "gating_probs": [0.80, 0.20], "token_idx": 0}
```

| Field | Description |
|-------|-------------|
| `dataset` | Source dataset |
| `problem_id` | Problem number (0-indexed) |
| `layer` | MoE layer (0-31) |
| `token_idx` | Generated token position |
| `experts` | Top-2 selected expert IDs (0-7) |
| `gating_probs` | Gating probabilities |

## Collection

Collected using the vLLM profiler in `../profiling_moe_code/`.
