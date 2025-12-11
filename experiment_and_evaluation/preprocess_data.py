"""
TOKENWISE SPLIT PREPROCESSING
Same as original but splits by SAMPLE (token) instead of by problem.
98% train, 1% val, 1% test
"""
import os
import json
import glob
import torch
import random
import numpy as np
from tqdm import tqdm

def process_all_data(data_dir):
    """Collect ALL samples from a dataset (no split yet)"""
    files = glob.glob(os.path.join(data_dir, "p*_token_*.jsonl"))
    print(f"  Found {len(files)} files")
    
    all_samples = []
    
    for filepath in tqdm(files, desc=f"Processing {data_dir}"):
        try:
            with open(filepath, 'r') as f:
                journey = [json.loads(line) for line in f if line.strip()]
            
            if len(journey) < 2: continue
            
            experts = [step['experts'][0] for step in journey]
            secondaries = [step['experts'][1] if len(step['experts']) > 1 else step['experts'][0] for step in journey]
            layers = [step['layer'] for step in journey]
            token_idx = journey[0]['token_idx']
            norm_pos = min(token_idx / 2048.0, 1.0)
            
            gating_probs = []
            for step in journey:
                probs = step.get('gating_probs', [0.0, 0.0])
                if len(probs) < 2: probs = probs + [0.0]*(2-len(probs))
                gating_probs.append(probs[:2])

            # Create samples for each layer transition
            for i in range(len(journey) - 1):
                hist_vec = [0] * 32
                for k in range(i + 1):
                    hist_vec[k] = experts[k] + 1
                
                sample = {
                    "layer": layers[i],
                    "history": hist_vec,
                    "secondary": secondaries[i],
                    "gating": gating_probs[i],
                    "pos": norm_pos,
                    "target": experts[i+1]
                }
                all_samples.append(sample)
                
        except Exception as e:
            pass
            
    return all_samples

def main():
    out_dir = "processed_data"
    os.makedirs(out_dir, exist_ok=True)
    
    # Paths to token data
    humaneval_dir = "../../routing_data_collected/humaneval_tokens"
    gsm8k_dir = "../../routing_data_collected/gsm8k_tokens"
    
    # 1. Collect ALL samples (no split yet)
    print("Collecting HumanEval samples...")
    h_samples = process_all_data(humaneval_dir)
    
    print("Collecting GSM8K samples...")
    g_samples = process_all_data(gsm8k_dir)
    
    # 2. Combine all samples
    all_samples = h_samples + g_samples
    print(f"\nTotal samples collected: {len(all_samples):,}")
    
    # 3. Shuffle ALL samples
    random.shuffle(all_samples)
    
    # 4. Split 98/1/1 by SAMPLE (not by problem)
    n_total = len(all_samples)
    n_train = int(n_total * 0.98)
    n_val = int(n_total * 0.01)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]
    
    print(f"\nTOKEN-WISE SPLIT (98/1/1):")
    print(f"  Train Samples: {len(train_samples):,} ({len(train_samples)/n_total*100:.1f}%)")
    print(f"  Val Samples:   {len(val_samples):,} ({len(val_samples)/n_total*100:.1f}%)")
    print(f"  Test Samples:  {len(test_samples):,} ({len(test_samples)/n_total*100:.1f}%)")
    
    # 5. Save as tensors
    def save_split(samples, name):
        if not samples: return
        layers = torch.tensor([s['layer'] for s in samples], dtype=torch.long)
        history = torch.tensor([s['history'] for s in samples], dtype=torch.long)
        secondary = torch.tensor([s['secondary'] for s in samples], dtype=torch.long)
        gating = torch.tensor([s['gating'] for s in samples], dtype=torch.float32)
        pos = torch.tensor([s['pos'] for s in samples], dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor([s['target'] for s in samples], dtype=torch.long)
        
        data = {
            "layers": layers,
            "history": history,
            "secondary": secondary,
            "gating": gating,
            "pos": pos,
            "targets": targets
        }
        torch.save(data, os.path.join(out_dir, f"{name}.pt"))
        print(f"Saved {name}.pt")

    print("\nSaving tensors...")
    save_split(train_samples, "train_data")
    save_split(val_samples, "val_data")
    save_split(test_samples, "test_data")
    print("Done!")

if __name__ == "__main__":
    random.seed(42)
    main()
