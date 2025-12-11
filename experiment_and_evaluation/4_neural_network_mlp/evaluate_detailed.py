import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from mlp_dataset import TokenJourneyDataset
from mlp_model import ExpertPredictorMLP
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_detailed(data_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Load Test Data
    test_file = os.path.join(data_dir, 'test_data.pt')
    if not os.path.exists(test_file):
        print("Test data not found!")
        return

    print("Loading Test Data (HELD OUT)...")
    test_dataset = TokenJourneyDataset(test_file)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False) # Large batch for eval
    
    # Load Model
    print(f"Loading Model from {model_path}...")
    # Use Optimal Configuration
    model = ExpertPredictorMLP(
        embed_dim_layer=5, 
        embed_dim_expert=7, 
        hidden_dims=[512, 512], 
        dropout=0.18
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Storage for detailed stats
    all_preds = []
    all_targets = []
    all_layers = []
    
    # Metrics
    total_correct_top1 = 0
    total_correct_top2 = 0
    total_samples = 0
    
    # Per-layer stats
    layer_correct = {} # layer_idx -> correct_count
    layer_total = {}   # layer_idx -> total_count
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in test_loader:
            layer = batch['layer'].to(device)
            history = batch['history'].to(device)
            secondary = batch['secondary'].to(device)
            gating = batch['gating'].to(device)
            pos = batch['pos'].to(device)
            target = batch['target'].to(device)
            
            logits = model(layer, history, secondary, gating, pos)
            
            # Global Top-1
            _, preds = torch.max(logits, 1)
            total_correct_top1 += (preds == target).sum().item()
            
            # Global Top-2
            _, top2_preds = logits.topk(2, dim=1)
            total_correct_top2 += torch.any(top2_preds == target.unsqueeze(1), dim=1).sum().item()
            
            total_samples += target.size(0)
            
            # Store for detailed analysis (Top-1 only usually sufficient for per-layer but let's keep logic)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_layers.extend(layer.cpu().numpy())
            
            # Layer-wise Top-1 metrics
            correct_mask = (preds == target)
            for l, c in zip(layer.cpu().numpy(), correct_mask.cpu().numpy()):
                layer_correct[l] = layer_correct.get(l, 0) + int(c)
                layer_total[l] = layer_total.get(l, 0) + 1

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # --- ANALYSIS ---
    
    print("\n" + "="*40)
    print("       DEEP EVALUATION REPORT       ")
    print("="*40 + "\n")
    
    # 1. Global Metrics
    top1_acc = (total_correct_top1 / total_samples) * 100
    top2_acc = (total_correct_top2 / total_samples) * 100
    print(f"Overall Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Overall Top-2 Accuracy: {top2_acc:.2f}% (CRITICAL METRIC)")
    
    # 2. Baseline Comparison
    # What if we always predicted the most common expert in the test set?
    vals, counts = np.unique(all_targets, return_counts=True)
    majority_class = vals[np.argmax(counts)]
    majority_acc = (np.sum(all_targets == majority_class) / len(all_targets)) * 100
    print(f"Baseline (Majority Expert {majority_class}): {majority_acc:.2f}%")
    print(f" Improvement over Baseline: +{top1_acc - majority_acc:.2f}%")
    
    # 3. Random Baseline
    print(f"Random Guessing Baseline: {100.0/8:.2f}%")
    
    # 4. Per-Layer Performance
    print("\n--- Accuracy per Layer ---")
    print(f"{'Layer':<6} | {'Samples':<8} | {'Accuracy':<8}")
    print("-" * 30)
    sorted_layers = sorted(layer_total.keys())
    for l in sorted_layers:
        l_acc = (layer_correct[l] / layer_total[l]) * 100
        print(f"{l:<6} | {layer_total[l]:<8} | {l_acc:.2f}%")
        
    # 5. Class Distribution Check
    print("\n--- Prediction Distribution ---")
    print("Is the model just predicting mostly one thing?")
    pred_vals, pred_counts = np.unique(all_preds, return_counts=True)
    targ_vals, targ_counts = np.unique(all_targets, return_counts=True)
    
    print(f"{'Expert':<6} | {'Actual %':<10} | {'Predicted %':<10}")
    print("-" * 35)
    total_samples = len(all_preds)
    for i in range(8):
        act_pct = (np.sum(all_targets == i) / total_samples) * 100
        pred_pct = (np.sum(all_preds == i) / total_samples) * 100
        print(f"{i:<6} | {act_pct:.1f}%      | {pred_pct:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../processed_data')
    parser.add_argument('--model-path', default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    evaluate_detailed(args.data_dir, args.model_path)
