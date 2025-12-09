"""
MoE Routing Data Collection Script

Collects routing decisions from Mixtral-8x7B for speculative decoding research.

Usage:
    python collect_data.py --dataset humaneval --num-problems 5
"""
import os
import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
from vllm.custom_profiling import get_profiler


def process_raw_data(raw_path: str, output_path: str):
    """
    Post-process raw routing data:
    - Extract only decode tokens (token_pos=0 after prefill)
    - Add 0-indexed token_idx
    - Remove duplicate processing cycles
    """
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found")
        return 0
    
    raw_data = [json.loads(line) for line in open(raw_path)]
    print(f"Raw records: {len(raw_data)}")
    
    decode_records = []
    
    for layer in range(32):  # Mixtral has 32 layers
        layer_records = [d for d in raw_data if d['layer'] == layer]
        
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
    
    return len(decode_records)


def main():
    parser = argparse.ArgumentParser(description="Collect MoE routing data")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "gsm8k"])
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    
    # Setup profiler
    os.environ["ROUTING_PROFILER_ENABLED"] = "1"
    raw_path = os.path.join(args.output_dir, "routing_raw.jsonl")
    os.environ["ROUTING_OUTPUT_PATH"] = raw_path
    
    profiler = get_profiler()
    profiler.clear_output()  # Clear any previous data
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [item["prompt"] for item in ds]
    else:  # gsm8k
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [item["question"] for item in ds]
    
    prompts = prompts[:args.num_problems]
    print(f"Processing {len(prompts)} problems...")
    
    # Initialize vLLM
    # Using FP8 quantization (8-bit) on single GPU for complete routing data
    # - Single GPU = no expert parallelism = complete data for all 32 layers
    # - FP8 preserves routing patterns while fitting in 1x A100 80GB
    # - enforce_eager required for profiler's .cpu() calls
    llm = LLM(
        model="neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for deterministic routing
        max_tokens=256,
    )
    
    # Run inference
    for i, prompt in enumerate(prompts):
        print(f"  Problem {i+1}/{len(prompts)}...")
        profiler.set_context(problem_id=i, dataset=args.dataset)
        llm.generate([prompt], sampling_params)
    
    # Post-process
    print("Post-processing data...")
    output_path = os.path.join(args.output_dir, f"{args.dataset}_routing.jsonl")
    num_records = process_raw_data(raw_path, output_path)
    
    print(f"\nDone! Saved {num_records} records to {output_path}")
    print(f"Output tokens per layer: {num_records // 32}")


if __name__ == "__main__":
    main()
