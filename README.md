# speculative-moe
## Cross-Layer Routing Patterns for MoE Optimization

Research project analyzing cross-layer routing patterns in Mixture-of-Experts (MoE) models for speculative pipeline parallelism.

## Overview

This project collects and analyzes routing decisions from Mixtral-8x7B to discover predictable patterns that can be exploited for speculative pipeline parallelism optimization.

**Key Result**: We achieve **55.6% Top-1** and **73.2% Top-2** accuracy for expert prediction — an accuracy improvement of **+345% (4.5×)** over random guessing.

## Target Model

**Mixtral-8x7B-Instruct-v0.1-FP8** (8-bit quantized)
- 8 experts per layer
- Top-2 routing (selects 2 experts per token)
- 32 MoE transformer layers

## Why Quantization Doesn't Affect Routing Decisions

We use FP8 quantization to fit Mixtral on a single GPU. This is valid for routing analysis because:

> **The gating/router network is NOT quantized** — it always runs at FP16/BF16 precision.

In MoE models:
1. The **router** computes gating logits to decide which experts to use
2. The **experts** (FFN weights) perform the actual computation
3. Quantization (FP8/AWQ/GPTQ) only applies to expert weights, not the router

This means routing patterns remain identical to full-precision models. MoE routers are highly sensitive, and quantization noise could route tokens to wrong experts — so vLLM and other frameworks explicitly keep routers unquantized.

## Datasets

- **HumanEval**: 164 problems (code generation) — [openai_humaneval](https://huggingface.co/datasets/openai_humaneval)
- **GSM8K**: 80 problems profiled (math reasoning) — [gsm8k](https://huggingface.co/datasets/gsm8k)

Combined: **1.43M routing records** across 244 problems, split 98/1/1 for train/val/test.

## Project Structure

```
speculative-moe/
├── README.md                    # This file
├── proposal.pdf                 # Project proposal
│
├── profiling_moe_code/          # Custom profiling code for data collection
│   ├── README.md                # Setup instructions
│   ├── collect_data.py          # Main data collection script
│   ├── custom_profiling/        # vLLM routing profiler
│   └── LAYER_PATCH.txt          # vLLM layer.py patch
│
├── routing_data_collected/      # Collected routing data
│   ├── README.md                # Data format documentation
│   ├── sample/                  # Sample datasets (2 problems each)
│   ├── humaneval_full_*.jsonl   # Full HumanEval data (164 problems)
│   └── gsm8k_full_*.jsonl       # Full GSM8K data (80 problems profiled)
│
├── experiment_and_evaluation/   # All experiments and results
│   ├── README.md                # Experiment navigation guide
│   ├── preprocess_data.py       # Data preprocessing
│   ├── processed_data/          # Train/val/test splits
│   ├── 1_lookup_table/          # Step 1: Frequency-based rules
│   ├── 2_xgboost/               # Step 2: Classical ML baseline
│   ├── 3_bayesian_search/       # Step 3: Hyperparameter optimization
│   ├── 4_neural_network_mlp/    # Step 4: Final optimized model
│   └── Final_Experiment_Evaluation_Summary.md  # Results summary
│
└── others/                      # Legacy scripts and notes
```

## Methodology

### Data Collection

For each token at each MoE layer, we capture:
- `problem_id` — which problem from the dataset
- `layer` — which MoE layer (0-31 for Mixtral)
- `token_idx` — position in the generated sequence
- `experts` — which top-2 experts were selected
- `gating_probs` — gating probabilities for selected experts

### What We Discovered

**1. Layer-Expert Patterns**  
Certain experts are preferred at certain layers. We captured this in our Level 1 lookup rules (Layer → Expert frequency).

**2. Cross-Layer Transitions**  
Expert transitions between layers are not random — they follow predictable patterns. Our Level 2-3 rules capture (Layer, Previous Expert(s), Current Expert) → Next Expert probabilities.

**3. Predictability — The Main Finding**  
Yes, we can predict Layer N+1 routing from Layer N decisions. Our best model achieves:

| Model | Top-1 Accuracy | Top-2 Accuracy | vs Random |
|-------|----------------|----------------|-----------|
| Random Baseline | 12.5% | 25.0% | — |
| Lookup Table (Best) | 42.2% | 56.2% | +237% (3.4x) |
| XGBoost | 36.2% | 53.3% | +189% (2.9x) |
| **Neural Network (MLP)** | **55.6%** | **73.2%** | **+345% (4.5x)** |

This means if we speculatively prepare 2 experts ahead of time, we'll be correct 73% of the time — enabling significant pipeline parallelism gains.

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA GPU with ~45GB VRAM (for FP8 Mixtral)
- vLLM with custom profiling patch

### Setup & Run

See `profiling_moe_code/README.md` for full setup instructions.

```bash
# Install dependencies
pip install vllm datasets

# Apply vLLM patch (see profiling_moe_code/LAYER_PATCH.txt)
# Copy profiling_moe_code/custom_profiling/ to your vLLM installation

# Run data collection
cd profiling_moe_code
python collect_data.py --dataset humaneval --num-problems 5
```

### Output Format

Data is saved as JSONL (one JSON object per line):

```json
{"dataset": "humaneval", "problem_id": 0, "layer": 0, "token_idx": 0, "experts": [3, 7], "gating_probs": [0.68, 0.32]}
```

## References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mistral AI
- [vLLM](https://github.com/vllm-project/vllm) — Fast LLM inference

## License

Research project for academic purposes.
