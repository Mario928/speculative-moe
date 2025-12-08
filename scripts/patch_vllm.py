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

# The hook to insert - with debug printing
hook = """
        # Routing profiler hook - DEBUG VERSION
        import os as _os
        if _os.getenv("ROUTING_PROFILER_ENABLED", "0") == "1":
            try:
                from vllm.custom_profiling.routing_profiler import get_profiler
                _profiler = get_profiler()
                _profiler.record(self.layer_name, topk_ids, topk_weights)
            except Exception as _e:
                print(f"[PROFILER ERROR] {type(_e).__name__}: {_e}")
"""

# Insert before return statement
target = "return topk_weights, topk_ids, zero_expert_result"

if "routing_profiler" in code:
    print("Already patched - skipping")
elif target not in code:
    print(f"Warning: Target '{target}' not found in layer.py")
    # Try alternate target
    alt_target = "return topk_weights, topk_ids"
    if alt_target in code:
        code = code.replace(alt_target, hook + "        " + alt_target)
        layer.write_text(code)
        print("Patched with alternate target!")
    else:
        print("No suitable target found")
        sys.exit(1)
else:
    code = code.replace(target, hook + "        " + target)
    layer.write_text(code)
    print("Patched layer.py successfully!")

