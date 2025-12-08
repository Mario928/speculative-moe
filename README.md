# speculative-moe
## Cross-Layer Routing Patterns for MoE Optimization

Research project analyzing cross-layer routing predictability in Mixture-of-Experts (MoE) models for potential speculative execution optimizations.

## Goal

Measure: **"If a token goes to Expert X at Layer N, how likely is it to go to Expert X at Layer N+1?"**

This research explores whether MoE routing decisions are predictable enough to enable speculative parallel execution.

## Project Structure

```
speculative-moe/
├── README.md           # This file
├── proposal.pdf        # Project proposal
├── vllm_modified/      # Modified vLLM with routing profiler
├── scripts/            # Modal scripts for data collection
└── others/             # Notes and research (not pushed)
```

## vLLM Modifications

We modified vLLM to capture routing decisions during inference.

**Base version:** vLLM dev (commit `af0444bf4`)

**Files added:**
- `vllm_modified/vllm/custom_profiling/__init__.py`
- `vllm_modified/vllm/custom_profiling/routing_profiler.py`

**Files modified:**
- `vllm_modified/vllm/model_executor/layers/fused_moe/layer.py`
  - Added 4 lines at end of `select_experts()` function (around line 1655) to call our profiler

**What the profiler captures:**
- `problem_id` - which problem from dataset
- `layer` - which MoE layer (0-31 for Mixtral)
- `token_id` - token position in sequence
- `expert_ids` - which experts were selected
- `weights` - gating probabilities

## Usage with Modal

```python
# Modal installs from this repo
image = modal.Image.pip_install(
    "git+https://github.com/Mario928/speculative-moe.git#subdirectory=vllm_modified"
)
```

Enable profiling with:
```bash
export ROUTING_PROFILER_ENABLED=1
```

## Target Model

**Mixtral-8x7B**
- 8 experts per layer
- Top-2 routing (k=2)
- 32 transformer layers

## Datasets

- HumanEval (164 coding problems)
- GSM8K (500 math problems)
