"""
Bayesian Optimization for Hyperparameter Search using Optuna
Smarter than grid search - picks configs based on previous results
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from mlp_dataset import TokenJourneyDataset

def create_model(layer_emb, expert_emb, hidden_dims, dropout):
    """Build MLP with specified config"""
    input_size = layer_emb + expert_emb + (32 * expert_emb) + 2 + 1
    
    layers = []
    prev_size = input_size
    for hidden_size in hidden_dims:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        ])
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, 8))
    
    layer_embedding = nn.Embedding(32, layer_emb)
    expert_embedding = nn.Embedding(9, expert_emb)
    
    class ConfigurableMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_embedding = layer_embedding
            self.expert_embedding = expert_embedding
            self.net = nn.Sequential(*layers)
        
        def forward(self, layer, history, secondary, gating, pos):
            layer_emb = self.layer_embedding(layer)
            sec_emb = self.expert_embedding(secondary + 1)
            hist_emb = self.expert_embedding(history).view(history.size(0), -1)
            x = torch.cat([layer_emb, sec_emb, hist_emb, gating, pos], dim=1)
            return self.net(x)
    
    return ConfigurableMLP()

def objective(trial, train_loader, val_loader, device):
    """Optuna objective function - returns metric to MAXIMIZE"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    layer_emb = trial.suggest_int('layer_emb', 5, 20)
    expert_emb = trial.suggest_int('expert_emb', 3, 10)
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    
    # Architecture
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_dims = []
    for i in range(n_layers):
        if i == 0:
            dim = trial.suggest_int(f'hidden_0', 128, 512, step=64)
        else:
            # Each layer should be smaller than previous
            dim = trial.suggest_int(f'hidden_{i}', 32, hidden_dims[-1], step=32)
        hidden_dims.append(dim)
    
    print(f"\nTrial {trial.number}: lr={lr:.4f}, emb={layer_emb}/{expert_emb}, "
          f"arch={hidden_dims}, dropout={dropout:.2f}")
    
    # Build and train model
    model = create_model(layer_emb, expert_emb, hidden_dims, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    best_val_top2 = 0
    epochs = 10  # Fewer epochs to test more configs
    
    for epoch in range(epochs):
        # Train
        model.train()
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
        
        # Validate every 2 epochs (faster)
        if (epoch + 1) % 2 == 0:
            model.eval()
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
                    _, top2_preds = torch.topk(logits, 2, dim=1)
                    val_correct_top2 += torch.any(top2_preds == target.unsqueeze(1), dim=1).sum().item()
                    val_total += target.size(0)
            
            val_top2 = 100 * val_correct_top2 / val_total
            if val_top2 > best_val_top2:
                best_val_top2 = val_top2
            
            print(f"  Epoch {epoch+1}: Val Top-2 = {val_top2:.2f}% (best: {best_val_top2:.2f}%)")
        
        # Pruning: stop if clearly bad
        if epoch == 4 and best_val_top2 < 60:
            raise optuna.TrialPruned()
    
    return best_val_top2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    print(f"Using device: {device}\n")
    
    # Load data
    train_dataset = TokenJourneyDataset("../processed_data/train_data.pt")
    val_dataset = TokenJourneyDataset("../processed_data/val_data.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    
    # Run optimization (20 trials)
    print("="*60)
    print("BAYESIAN OPTIMIZATION - 20 Trials")
    print("="*60)
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=20,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n🏆 Best Trial: #{study.best_trial.number}")
    print(f"   Best Val Top-2: {study.best_value:.2f}%")
    print(f"\n   Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")
    
    # Save study
    import joblib
    joblib.dump(study, 'optuna_study.pkl')
    print(f"\n💾 Study saved to optuna_study.pkl")
    
    # Plot importance (optional)
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html('param_importance.html')
        print("📊 Importance plot saved to param_importance.html")
    except:
        pass

if __name__ == "__main__":
    main()
