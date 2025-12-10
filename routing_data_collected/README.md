# Routing Data Collected

This folder contains MoE routing decisions collected from **Mixtral-8x7B-Instruct-v0.1-FP8** during inference on HumanEval and GSM8K datasets.

> [!NOTE]  
> **Large files (`*_full_routing.jsonl`) are not in git** due to GitHub's 100MB limit.  
> Contact the maintainers for access to the full routing data files (~214MB total).

## Data Files

| File | Description | Problems | Records |
|------|-------------|----------|---------|
| `humaneval_full_routing.jsonl` | Full HumanEval routing data | 164 | ~4.5M records |
| `humaneval_full_outputs.jsonl` | Full HumanEval generated code | 164 | - |
| `gsm8k_full_routing.jsonl` | Full GSM8K routing data | 1319 | ~2.2M records |
| `gsm8k_full_outputs.jsonl` | Full GSM8K generated answers | 1319 | - |
| `humaneval_2_*` / `gsm8k_2_*` | Test runs with 2 problems | 2 | - |

## Data Format

Each line in the routing JSONL files contains:

```json
{"dataset": "humaneval", "problem_id": 0, "layer": 15, "token_idx": 42, "experts": [3, 7], "gating_probs": [0.68, 0.32]}
```

| Field | Description |
|-------|-------------|
| `dataset` | Source dataset (`humaneval` or `gsm8k`) |
| `problem_id` | 0-indexed problem number |
| `layer` | MoE layer (0-31 for Mixtral) |
| `token_idx` | 0-indexed generated token position |
| `experts` | Top-2 selected expert IDs (0-7) |
| `gating_probs` | Gating probabilities for each expert |

## Collection Method

Data was collected using the modified vLLM profiler in `../profiling_moe_code/`. See that folder for setup instructions.

## Analysis Results

Analysis scripts and results are in `../analysis/`. Key findings include per-layer expert biases and cross-layer transition patterns useful for speculative pipeline parallelism.
