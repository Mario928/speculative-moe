# speculative-moe
## Cross-Layer Routing Patterns for MoE Optimization

Research project analyzing cross-layer routing patterns in Mixture-of-Experts (MoE) models.

## Overview

This project analyzes routing patterns in Mixture-of-Experts (MoE) models to understand cross-layer expert selection behavior.

## Methodology

### Data Collection

For each token at each MoE layer, we capture:
- `problem_id` - which problem from the dataset
- `layer` - which MoE layer (0-31 for Mixtral)
- `token_id` - position in the sequence
- `expert_ids` - which top-k experts were selected
- `weights` - gating probabilities for selected experts

### Analysis

1. **Single-hop prediction**: P(same expert at layer N+1 | expert at layer N)
2. **Multi-hop prediction**: P(same expert at layer N+k | expert at layer N) for k=1,2,...
3. **Cross-domain comparison**: Compare patterns between coding (HumanEval) and math (GSM8K)

## Target Model

**Mixtral-8x7B-v0.1**
- 8 experts per layer
- Top-2 routing (selects 2 experts per token)
- 32 transformer layers

## Datasets

- **HumanEval** (164 problems): [openai_humaneval](https://huggingface.co/datasets/openai_humaneval) - coding tasks
- **GSM8K** (1319 problems): [gsm8k](https://huggingface.co/datasets/gsm8k) - grade school math

## GPU Requirements & Quantization

### Memory Requirements

Mixtral-8x7B has different VRAM requirements depending on precision:

| Precision | VRAM Required | Single A100-80GB? |
|-----------|---------------|-------------------|
| FP16/BF16 | ~90-100 GB | ❌ No |
| INT8 | ~45 GB | ✅ Yes |
| INT4 (AWQ) | ~22 GB | ✅ Yes |

### Why Quantization is Valid for This Research

In vLLM's Mixtral implementation, **the router/gate is NOT quantized** even when using AWQ or GPTQ:

> "The `self.gate` component has `quant_config=None` and always runs at half / full precision"
> — [vLLM source code](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py)

This means:
- Router decisions happen at FP16 precision (not quantized)
- Only expert FFN weights are quantized
- **Routing patterns remain authentic** with quantized models

### Model Choice

We use AWQ-quantized Mixtral for this research:
```python
llm = LLM(model="TheBloke/Mixtral-8x7B-v0.1-AWQ", quantization="awq")
```

This allows running on a single A100-80GB while preserving authentic routing behavior.

## Project Structure

```
speculative-moe/
├── README.md                    # This file
├── proposal.pdf                 # Project proposal
├── vllm_modified/               # Modified vLLM with routing profiler
│   └── vllm/
│       └── custom_profiling/    # Our addition
│           ├── __init__.py
│           └── routing_profiler.py
├── scripts/                     # Modal scripts for data collection
└── analysis/                    # Analysis notebooks (coming soon)
```

## vLLM Modifications

We modified vLLM to capture routing decisions during inference.

**Base version:** vLLM dev (commit `af0444bf4`)

**Files added:**
- `vllm_modified/vllm/custom_profiling/__init__.py`
- `vllm_modified/vllm/custom_profiling/routing_profiler.py`

**Files modified:**
- `vllm_modified/vllm/model_executor/layers/fused_moe/layer.py`
  - Added profiler call at end of `select_experts()` function

## How to Run

### Prerequisites

1. [Modal](https://modal.com) account (has $30/month free credits)
2. Python 3.11+

### Setup

```bash
# Install modal
pip install modal

# Authenticate
modal setup
```

### Run Data Collection

```bash
# Run on 5 HumanEval problems (quick test)
modal run scripts/collect_routing.py --num-problems 5 --dataset humaneval

# Run full collection
modal run scripts/collect_routing.py --num-problems 164 --dataset humaneval
modal run scripts/collect_routing.py --num-problems 500 --dataset gsm8k
```

### Output Format

Data is saved as JSONL (one JSON object per line):

```json
{"problem_id": 0, "layer": 0, "token_id": 0, "expert_ids": [3, 7], "weights": [0.6, 0.4]}
{"problem_id": 0, "layer": 1, "token_id": 0, "expert_ids": [3, 5], "weights": [0.7, 0.3]}
...
```

## References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - Mistral AI
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference

## License

Research project for academic purposes.
