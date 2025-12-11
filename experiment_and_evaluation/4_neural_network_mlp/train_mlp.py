import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import json
from mlp_dataset import TokenJourneyDataset
from mlp_model import ExpertPredictorMLP

def train_model(train_file, val_file, output_dir='checkpoints', 
                batch_size=512, epochs=30, lr=0.000536):
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Initializing datasets...")
    train_dataset = TokenJourneyDataset(train_file)
    val_dataset = TokenJourneyDataset(val_file)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 2. Init Model (Optimal Configuration)
    # lr=0.000536, layer_emb=5, expert_emb=7, architecture=[512, 512], dropout=0.18
    print("Using Optimal Configuration: [512, 512], Emb: 5/7, Drop: 0.18, LR: 0.000536")
    model = ExpertPredictorMLP(
        embed_dim_layer=5, 
        embed_dim_expert=7, 
        hidden_dims=[512, 512], 
        dropout=0.18
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # 3. Training Loop
    best_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 5
    
    # Training history for plotting
    training_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct_top1 = 0
        total = 0
        
        for batch in train_loader:
            layer = batch['layer'].to(device)
            history = batch['history'].to(device)
            secondary = batch['secondary'].to(device)
            gating = batch['gating'].to(device)
            pos = batch['pos'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            logits = model(layer, history, secondary, gating, pos)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Train Stats (Top-1)
            _, preds = torch.max(logits, 1)
            correct_top1 += (preds == target).sum().item()
            total += target.size(0)
            
        train_acc = 100 * correct_top1 / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct_top1 = 0
        val_correct_top2 = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                layer = batch['layer'].to(device)
                history = batch['history'].to(device)
                secondary = batch['secondary'].to(device)
                gating = batch['gating'].to(device)
                pos = batch['pos'].to(device)
                target = batch['target'].to(device)
                
                logits = model(layer, history, secondary, gating, pos)
                
                # Top-1
                _, preds = torch.max(logits, 1)
                val_correct_top1 += (preds == target).sum().item()
                
                # Top-2
                _, top2_preds = logits.topk(2, dim=1)
                val_correct_top2 += torch.any(top2_preds == target.unsqueeze(1), dim=1).sum().item()
                
                val_total += target.size(0)
        
        val_acc1 = 100 * val_correct_top1 / val_total
        val_acc2 = 100 * val_correct_top2 / val_total
        
        scheduler.step(val_acc1)
        
        epoch_time = time.time() - start_time
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Top-1: {val_acc1:.2f}% | Val Top-2: {val_acc2:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save to history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_top1": val_acc1,
            "val_top2": val_acc2,
            "time": epoch_time
        })
        
        # Checkpoint
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("  --> New Best Model Saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining Complete. Best Validation Top-1 Accuracy: {best_acc:.2f}%")
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {output_dir}/training_history.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='processed_data', help='Directory containing .pt files')
    args = parser.parse_args()
    
    train_file = os.path.join(args.data_dir, 'train_data.pt')
    val_file = os.path.join(args.data_dir, 'val_data.pt')
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run preprocess_data.py first.")
    else:
        train_model(train_file, val_file)
