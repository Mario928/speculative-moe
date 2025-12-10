"""
MoE Routing Data Collection Script

Collects routing decisions from Mixtral-8x7B for speculative decoding research.

Usage:
    # HumanEval - 5 problems
    python collect_data.py --dataset humaneval --num-problems 5
    
    # HumanEval - all 164 problems
    python collect_data.py --dataset humaneval
    
    # GSM8K - 10 problems
    python collect_data.py --dataset gsm8k --num-problems 10
    
    # GSM8K - all 1319 problems
    python collect_data.py --dataset gsm8k

Output files saved to output/ folder:
    - {dataset}_{count}_routing.jsonl  (routing decisions)
    - {dataset}_{count}_outputs.jsonl  (prompts + generated answers)
"""
import os
import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
from vllm.custom_profiling import get_profiler


def process_raw_data(raw_path: str, output_path: str):
    """
    Post-process raw routing data to extract ONLY generated (decode) tokens.
    
    Why needed:
    - Raw data has prefill tokens (prompt processing) + decode tokens (generation)
    - Prefill: token_pos goes 0,1,2,3... (sequential)
    - Decode: token_pos is always 0 for each generated token
    - We detect decode start when token_pos resets to 0 after sequential prefill
    
    Output: Clean data with only decode tokens, indexed 0,1,2,3...
    """
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found")
        return 0
    
    raw_data = [json.loads(line) for line in open(raw_path)]
    print(f"Raw records: {len(raw_data)}")
    
    # Get unique problem_ids from the data
    problem_ids = sorted(set(d['problem_id'] for d in raw_data))
    print(f"Found {len(problem_ids)} problems: {problem_ids}")
    
    decode_records = []
    
    # Process each problem separately, then each layer within that problem
    for problem_id in problem_ids:
        problem_records = [d for d in raw_data if d['problem_id'] == problem_id]
        
        for layer in range(32):  # Mixtral has 32 layers
            layer_records = [d for d in problem_records if d['layer'] == layer]
            
            if not layer_records:
                continue
            
            # Find where decode starts (first token_pos=0 after sequential prefill)
            decode_start = None
            for i, d in enumerate(layer_records):
                if i > 0 and layer_records[i-1]['token_pos'] > 0 and d['token_pos'] == 0:
                    decode_start = i
                    break
            
            if decode_start is not None:
                # Find where first decode cycle ends
                decode_end = len(layer_records)
                for i in range(decode_start + 1, len(layer_records)):
                    if layer_records[i]['token_pos'] > 0:
                        decode_end = i
                        break
                
                # Extract decode tokens with 0-indexed token_idx
                for idx, d in enumerate(layer_records[decode_start:decode_end]):
                    d['token_idx'] = idx
                    del d['token_pos']  # Remove batch position, not needed
                    decode_records.append(d)
    
    # Save processed data
    with open(output_path, 'w') as f:
        for d in decode_records:
            f.write(json.dumps(d) + '\n')
    
    # Keep raw file for debugging (uncomment to auto-delete):
    # os.remove(raw_path)
    
    return len(decode_records)


def main():
    parser = argparse.ArgumentParser(description="Collect MoE routing data")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "gsm8k"])
    parser.add_argument("--num-problems", type=int, default=None, help="Number of problems (default: all)")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup profiler
    os.environ["ROUTING_PROFILER_ENABLED"] = "1"
    os.environ["CURRENT_DATASET"] = args.dataset
    raw_path = os.path.join(args.output_dir, "routing_raw.jsonl")
    os.environ["ROUTING_OUTPUT_PATH"] = raw_path
    
    # File-based problem_id communication (env vars don't update after worker spawn)
    problem_id_file = os.path.join(args.output_dir, ".current_problem_id")
    os.environ["PROBLEM_ID_FILE"] = problem_id_file
    
    profiler = get_profiler()
    profiler.clear_output()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [(item["task_id"], item["prompt"]) for item in ds]
    else:  # gsm8k
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [(f"gsm8k_{i}", item["question"]) for i, item in enumerate(ds)]
    
    # Limit problems if specified
    if args.num_problems:
        prompts = prompts[:args.num_problems]
    num_problems = len(prompts)
    
    print(f"Processing {num_problems} problems...")
    
    # File naming: dataset_count_routing.jsonl or dataset_full_routing.jsonl
    count_str = str(num_problems) if args.num_problems else "full"
    
    # Initialize vLLM - Single GPU with FP8 quantization (8-bit, best quality)
    llm = LLM(
        model="neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    
    # Run inference and store outputs
    outputs_path = os.path.join(args.output_dir, f"{args.dataset}_{count_str}_outputs.jsonl")
    with open(outputs_path, "w") as f_out:
        for i, (task_id, prompt) in enumerate(prompts):
            print(f"  Problem {i+1}/{num_problems}...")
            # Write problem_id to file (worker process reads this)
            with open(problem_id_file, 'w') as f:
                f.write(str(i))
            profiler.set_context(problem_id=i, dataset=args.dataset)
            result = llm.generate([prompt], sampling_params)
            generated = result[0].outputs[0].text
            
            # Save prompt and generated output
            f_out.write(json.dumps({
                "problem_id": i,
                "task_id": task_id,
                "prompt": prompt,
                "generated": generated
            }) + "\n")
    
    print(f"Saved outputs to {outputs_path}")
    
    # Post-process routing data
    print("Post-processing routing data...")
    routing_path = os.path.join(args.output_dir, f"{args.dataset}_{count_str}_routing.jsonl")
    num_records = process_raw_data(raw_path, routing_path)
    
    print(f"\nDone!")
    print(f"  Routing: {routing_path} ({num_records} records)")
    print(f"  Outputs: {outputs_path} ({num_problems} problems)")


if __name__ == "__main__":
    main()

