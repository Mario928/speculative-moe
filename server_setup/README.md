# MoE Routing Data Collection

Collects routing decisions from Mixtral-8x7B for speculative decoding research.

## Folder Structure (on server)

After setup, your vLLM folder should look like this:

```
vllm/                               ← Your vLLM clone
├── vllm/
│   ├── custom_profiling/           ← COPY this here (Step 1)
│   │   ├── __init__.py
│   │   └── routing_profiler.py
│   └── model_executor/layers/fused_moe/
│       └── layer.py                ← EDIT this (Step 2)
├── test_vllm.py                    ← Your test file
└── collect_data.py                 ← PUT this here (Step 3)
```

## Setup Steps

```bash
cd vllm  # Go to your vLLM folder

# Step 1: Copy the profiling module
cp -r /path/to/server_setup/custom_profiling vllm/

# Step 2: Edit layer.py (add 3 lines from LAYER_PATCH.txt)
nano vllm/model_executor/layers/fused_moe/layer.py

# Step 3: Copy the collection script
cp /path/to/server_setup/collect_data.py .

# Step 4: Run it!
python collect_data.py --dataset humaneval --num-problems 1
```

## Output

```json
{"dataset": "humaneval", "problem_id": 0, "layer": 0, "token_idx": 0, "experts": [6, 5], "gating_probs": [0.68, 0.32]}
```

| Field | Description |
|-------|-------------|
| `dataset` | humaneval or gsm8k |
| `problem_id` | 0-indexed problem number |
| `layer` | MoE layer (0-31) |
| `token_idx` | 0-indexed generated token |
| `experts` | Top-2 expert IDs |
| `gating_probs` | Gating probabilities |
