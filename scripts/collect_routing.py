"""
Routing Data Collection Script

Runs Mixtral-8x7B on HumanEval/GSM8K problems and collects routing decisions.

Usage:
    modal run scripts/collect_routing.py --num-problems 5 --dataset humaneval
"""
import modal
import os

# Image: Official vLLM + our profiling code patched in
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        # Step 1: Clone & install official vLLM
        "git clone https://github.com/vllm-project/vllm.git /vllm",
        "cd /vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .",
        # Step 2: Clone our repo and copy profiling code
        "git clone https://github.com/Mario928/speculative-moe.git /app",
        "cp -r /app/vllm_modified/vllm/custom_profiling /vllm/vllm/",
        # Step 3: Patch layer.py - insert profiler hook before return statement
        "sed -i '/return topk_weights, topk_ids, zero_expert_result/i\\        # Profiler hook\\n        try:\\n            from vllm.custom_profiling.routing_profiler import get_profiler\\n            get_profiler().record(self.layer_name, topk_ids, topk_weights)\\n        except: pass' /vllm/vllm/model_executor/layers/fused_moe/layer.py",
    )
    .uv_pip_install("datasets", "huggingface-hub==0.36.0", "flashinfer-python==0.5.2")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("routing-collection", image=image)

# Volumes
volume = modal.Volume.from_name("routing-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=7200,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
)
def collect_routing_data(
    num_problems: int = 5,
    dataset_name: str = "humaneval",
):
    """
    Run inference and collect routing decisions.
    
    Args:
        num_problems: Number of problems to process
        dataset_name: 'humaneval' or 'gsm8k'
    """
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    
    # Enable routing profiler
    os.environ["ROUTING_PROFILER_ENABLED"] = "1"
    
    # Load dataset
    if dataset_name == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [item["prompt"] for item in ds][:num_problems]
    else:
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [item["question"] for item in ds][:num_problems]
    
    print(f"Loaded {len(prompts)} prompts from {dataset_name}")
    
    # Initialize Mixtral
    print("Loading Mixtral-8x7B...")
    llm = LLM(
        model="TheBloke/Mixtral-8x7B-v0.1-AWQ",
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,  # Disable CUDA graphs to avoid capture error
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    
    # Get profiler
    from vllm.custom_profiling.routing_profiler import get_profiler
    profiler = get_profiler()
    
    # Process each problem one at a time
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing problem {i+1}/{len(prompts)}...")
        
        # Set problem ID
        profiler.set_problem_id(i)
        
        # Run inference
        outputs = llm.generate([prompt], sampling_params)
        
        results.append({
            "problem_id": i,
            "prompt_preview": prompt[:100] + "...",
            "output_preview": outputs[0].outputs[0].text[:100] + "...",
        })
    
    # Save routing data
    output_path = f"/data/{dataset_name}_routing.jsonl"
    profiler.save(output_path)
    
    # Commit volume
    volume.commit()
    
    print(f"\nDone! Processed {len(prompts)} problems.")
    print(f"Routing data saved to {output_path}")
    
    return {
        "dataset": dataset_name,
        "num_problems": len(prompts),
        "output_file": output_path,
        "samples": results[:3],
    }


@app.function(volumes={"/data": volume})
def list_collected_data():
    """List all collected routing data files."""
    import os
    files = []
    for f in os.listdir("/data"):
        path = f"/data/{f}"
        size = os.path.getsize(path)
        files.append({"file": f, "size_mb": round(size / 1024 / 1024, 2)})
    return files


@app.function(volumes={"/data": volume})
def download_data(filename: str):
    """Download a specific routing data file."""
    volume.reload()
    with open(f"/data/{filename}", "r") as f:
        return f.read()


@app.local_entrypoint()
def main(
    num_problems: int = 5,
    dataset: str = "humaneval",
):
    """
    Entry point - run from terminal:
        modal run scripts/collect_routing.py --num-problems 5 --dataset humaneval
    """
    print(f"Starting routing collection...")
    print(f"  Dataset: {dataset}")
    print(f"  Problems: {num_problems}")
    print()
    
    result = collect_routing_data.remote(num_problems, dataset)
    
    print(f"\nCollection complete!")
    print(f"  Output file: {result['output_file']}")
    print(f"  Problems processed: {result['num_problems']}")
    
    # Download the data
    print("\nDownloading data...")
    filename = f"{dataset}_routing.jsonl"
    data = download_data.remote(filename)
    
    with open(filename, "w") as f:
        f.write(data)
    
    print(f"Saved to local file: {filename}")
