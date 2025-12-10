# speculative-moe
## Cross-Layer Routing Patterns for MoE Optimization

Research project analyzing cross-layer routing patterns in Mixture-of-Experts (MoE) models for speculative pipeline parallelism.

## Overview

This project collects and analyzes routing decisions from Mixtral-8x7B to discover predictable patterns that can be exploited for speculative pipeline parallelism optimization.

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

This means:
- Routing decisions happen at full FP16 precision
- Only expert FFN computations use quantized weights
- **Routing patterns remain identical** to full-precision models

This is a standard practice — MoE routers are highly sensitive, and quantization noise could route tokens to wrong experts. vLLM and other frameworks explicitly keep routers unquantized.

## Datasets

- **HumanEval** (164 problems): [openai_humaneval](https://huggingface.co/datasets/openai_humaneval) — coding tasks
- **GSM8K** (1319 problems): [gsm8k](https://huggingface.co/datasets/gsm8k) — grade school math

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
│   │   ├── __init__.py
│   │   └── routing_profiler.py
│   └── LAYER_PATCH.txt          # vLLM layer.py patch
│
├── routing_data_collected/      # Collected routing data
│   ├── README.md                # Data format documentation
│   ├── humaneval_full_routing.jsonl  # Full HumanEval routing (164 problems)
│   ├── humaneval_full_outputs.jsonl  # Full HumanEval generations
│   ├── gsm8k_full_routing.jsonl      # Full GSM8K routing (1319 problems)
│   ├── gsm8k_full_outputs.jsonl      # Full GSM8K generations
│   └── ...                       # Additional test runs
│
├── analysis/                    # Analysis scripts and results
│   └── ...                      # Various analysis outputs
│
└── datasets/                    # Dataset configurations
```

## Methodology

### Data Collection

For each token at each MoE layer, we capture:
- `problem_id` — which problem from the dataset
- `layer` — which MoE layer (0-31 for Mixtral)
- `token_idx` — position in the generated sequence
- `experts` — which top-2 experts were selected
- `gating_probs` — gating probabilities for selected experts

### Analysis

1. **Layer bias analysis**: Which experts dominate at each layer
2. **Cross-layer patterns**: Expert transition probabilities between layers
3. **Predictability**: Can we predict layer N+1 routing from layer N decisions

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
