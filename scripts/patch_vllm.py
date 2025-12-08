"""
Patch vLLM's layer.py to add routing profiler hook.
Called during Modal image build.
"""
import pathlib
import sys

# Find layer.py in the vLLM installation at /vllm
layer = pathlib.Path("/vllm/vllm/model_executor/layers/fused_moe/layer.py")

if not layer.exists():
    print(f"Error: {layer} not found")
    sys.exit(1)

code = layer.read_text()

# The hook: write directly to file (bypasses multiprocessing issue)
hook = """
        # Routing profiler hook - writes directly to file
        import os as _os
        if _os.getenv("ROUTING_PROFILER_ENABLED", "0") == "1":
            try:
                import json as _json
                _layer_idx = int(self.layer_name.split("layers.")[1].split(".")[0])
                _ids = topk_ids.detach().cpu().tolist()
                _weights = topk_weights.detach().cpu().tolist()
                _problem_id = int(_os.getenv("CURRENT_PROBLEM_ID", "0"))
                with open("/data/routing_raw.jsonl", "a") as _f:
                    for _tok_idx, (_eid, _wt) in enumerate(zip(_ids, _weights)):
                        _f.write(_json.dumps({"p": _problem_id, "l": _layer_idx, "t": _tok_idx, "e": _eid, "w": _wt}) + "\\n")
            except Exception as _e:
                print(f"[HOOK ERR] {_e}")
"""

# Insert before return statement
target = "return topk_weights, topk_ids, zero_expert_result"

if "Routing profiler hook" in code:
    print("Already patched - skipping")
elif target not in code:
    print(f"Warning: Target not found")
    sys.exit(1)
else:
    code = code.replace(target, hook + "        " + target)
    layer.write_text(code)
    print("Patched layer.py successfully!")


