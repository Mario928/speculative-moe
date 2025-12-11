"""
XGBoost Baseline - Compare with MLP
"""
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import xgboost as xgb
import time

print("Loading preprocessed data...")
train_data = torch.load("../processed_data/train_data.pt")
val_data = torch.load("../processed_data/val_data.pt")
test_data = torch.load("../processed_data/test_data.pt")

n_train = len(train_data['targets'])
n_val = len(val_data['targets'])
n_test = len(test_data['targets'])

print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

# Convert to DataFrame format for XGBoost
def to_dataframe(data):
    n = len(data['targets'])
    rows = []
    
    for i in range(n):
        row = {
            'layer': int(data['layers'][i]),
            'secondary': int(data['secondary'][i]),
            'gating_1': float(data['gating'][i][0]),
            'gating_2': float(data['gating'][i][1]),
            'position': float(data['pos'][i][0]),
        }
        
        # Add 32 history features
        for j in range(32):
            row[f'h{j}'] = int(data['history'][i][j])
        
        rows.append(row)
    
    X = pd.DataFrame(rows)
    y = data['targets'].numpy()
    
    return X, y

print("\nConverting to tabular format...")
X_train, y_train = to_dataframe(train_data)
X_val, y_val = to_dataframe(val_data)
X_test, y_test = to_dataframe(test_data)

print(f"Features: {len(X_train.columns)}")

# Train XGBoost
print("\n" + "="*60)
print("TRAINING XGBOOST")
print("="*60)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=8,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    tree_method='hist'
)

start = time.time()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
elapsed = time.time() - start

print(f"Training time: {elapsed:.1f}s")

# Evaluate
print("\n" + "="*60)
print("RESULTS")
print("="*60)

def eval_model(X, y, name):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    top1 = accuracy_score(y, preds) * 100
    
    # Top-2
    top2_preds = np.argsort(probs, axis=1)[:, -2:]
    top2_correct = np.any(top2_preds == y[:, None], axis=1).sum()
    top2 = (top2_correct / len(y)) * 100
    
    print(f"{name:10s} Top-1: {top1:.2f}%  Top-2: {top2:.2f}%")
    return top1, top2

val_t1, val_t2 = eval_model(X_val, y_val, "Validation")
test_t1, test_t2 = eval_model(X_test, y_test, "Test")

print("\n" +  "="*60)
print("XGBOOST vs MLP COMPARISON")
print("="*60)
print(f"                      XGBoost    MLP (Baseline)")
print(f"Test Top-1:           {test_t1:5.2f}%     44.51%")
print(f"Test Top-2:           {test_t2:5.2f}%     64.48%")
print(f"Training time:        {elapsed:5.1f}s     ~300s")
print(f"Model complexity:     37 feats   178 dims")

# Feature importance
print("\n" + "="*60)
print("TOP 10 FEATURES")
print("="*60)

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10).to_string(index=False))

print("\n✅ Done!")
