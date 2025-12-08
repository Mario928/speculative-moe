"""
Routing Data Collection Script

Runs Mixtral-8x7B on HumanEval/GSM8K problems and collects routing decisions.

Usage:
    modal run scripts/collect_routing.py --num-problems 5 --dataset humaneval
"""
import modal
import os

# Create Modal image with our modified vLLM
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "datasets",
    "huggingface_hub",
    "git+https://github.com/Mario928/speculative-moe.git#subdirectory=vllm_modified",
)

app = modal.App("speculative-moe-collection", image=image)

# Volume to persist collected data
volume = modal.Volume.from_name("routing-data", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
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
        model="mistralai/Mixtral-8x7B-v0.1",
        tensor_parallel_size=1,
        max_model_len=2048,
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
