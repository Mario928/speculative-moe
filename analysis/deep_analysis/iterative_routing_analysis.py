
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def load_routing_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(data)

def analyze_transition_probabilities(df):
    """
    Analyzes Layer N -> Layer N+1 expert transitions.
    Returns a transition matrix.
    """
    # Mixtral has 8 experts
    n_experts = 8
    # We have 32 layers (0 to 31)
    
    # Transition count matrix: [Layer][Source_Expert] -> {Target_Expert: Count}
    # Actually, let's just make a global transition matrix first to see if there's a general property
    # And also layer-specific matrices
    
    layer_transitions = defaultdict(lambda: np.zeros((n_experts, n_experts)))
    global_transition = np.zeros((n_experts, n_experts))
    
    # Store 'Top-1' expert transitions and 'Top-2' set transitions
    
    for i in range(len(df) - 1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        # Check if they are consecutive layers for the same token
        if curr_row['layer'] + 1 != next_row['layer']:
            continue
            
        # Top-1 transition
        e1_curr = curr_row['experts'][0] # Primary expert
        e1_next = next_row['experts'][0] # Primary expert next layer
        
        layer_transitions[curr_row['layer']][e1_curr][e1_next] += 1
        global_transition[e1_curr][e1_next] += 1
        
    return layer_transitions, global_transition

def calculate_expert_affinity(df):
    """
    How often do experts appear together in the top-2 for a single layer?
    """
    affinity = np.zeros((8, 8))
    for _, row in df.iterrows():
        e1, e2 = row['experts']
        affinity[e1][e2] += 1
        affinity[e2][e1] += 1
    return affinity

def detect_periodicity(df):
    """
    Checks if expert selection repeats every K layers.
    """
    experts_seq = [row['experts'][0] for _, row in df.iterrows()]
    n = len(experts_seq)
    
    matches = {}
    for lag in range(1, n // 2):
        count = 0
        for i in range(n - lag):
            if experts_seq[i] == experts_seq[i+lag]:
                count += 1
        matches[lag] = count / (n - lag)
        
    return matches

def train_lookup_predictor(df):
    """
    Simulates the 'Option A' Lookup Table approach and evaluates accuracy.
    Predictor: Given Expert E at Layer N, predict Expert E' at Layer N+1.
    """
    # Build Table
    counts = defaultdict(Counter)
    
    train_data = df.iloc[:-1] # simple split implies we need more tokens for real train/test, 
                              # but here we test "fit" to see if pattern exists
    
    for i in range(len(df) - 1):
        curr_layer = df.iloc[i]['layer']
        next_layer = df.iloc[i+1]['layer']
        if curr_layer + 1 != next_layer: continue
        
        e_curr = df.iloc[i]['experts'][0]
        e_next_primary = df.iloc[i+1]['experts'][0] # Can we predict the primary?
        # Or predict the SET? Let's try predicting primary first.
        
        counts[(curr_layer, e_curr)][e_next_primary] += 1
        
    # Evaluate "Training Accuracy" (Self-consistency)
    correct = 0
    total = 0
    
    for i in range(len(df) - 1):
        curr_layer = df.iloc[i]['layer']
        next_layer = df.iloc[i+1]['layer']
        if curr_layer + 1 != next_layer: continue

        e_curr = df.iloc[i]['experts'][0]
        actual_next = df.iloc[i+1]['experts'][0]
        
        # Prediction: Most frequent next expert for this (layer, current_expert) pair
        if counts[(curr_layer, e_curr)]:
            prediction = counts[(curr_layer, e_curr)].most_common(1)[0][0]
            if prediction == actual_next:
                correct += 1
        total += 1
        
    return correct / total if total > 0 else 0.0

def analyze_token_file(filepath):
    print(f"ANALYZING: {filepath}")
    df = load_routing_data(filepath)
    if df.empty:
        print("Empty Data")
        return

    # 1. Transition Predictability
    acc = train_lookup_predictor(df)
    print(f"  Lookup Table (Layer,Expert->NextExpert) Consistency: {acc:.2%}")
    
    # 2. Periodicity
    periodicity = detect_periodicity(df)
    # Get top 3 lags
    top_lags = sorted(periodicity.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top Periodicity Lags: {top_lags}")
    
    # 3. Global Expert Usage
    primary_experts = [x[0] for x in df['experts']]
    counts = Counter(primary_experts)
    print(f"  Expert Usage Distribution: {dict(counts)}")
    
    return df

if __name__ == "__main__":
    t1 = analyze_token_file('token_1_routing.jsonl')
    t2 = analyze_token_file('token_2_routing.jsonl')
    t3 = analyze_token_file('token_3_routing.jsonl')
    
    # Cross-Token Validation
    # Use probability map from Token 1 to predict Token 2
    if t1 is not None and t2 is not None:
        print("\nCROSS-VALIDATION (Train on T1, Test on T2)")
        
        # Build Map from T1
        model = defaultdict(Counter)
        for i in range(len(t1) - 1):
            if t1.iloc[i]['layer'] + 1 != t1.iloc[i+1]['layer']: continue
            state = (t1.iloc[i]['layer'], t1.iloc[i]['experts'][0])
            next_outcome = t1.iloc[i+1]['experts'][0]
            model[state][next_outcome] += 1
        
        # Test on T2
        correct = 0
        total = 0
        for i in range(len(t2) - 1):
            if t2.iloc[i]['layer'] + 1 != t2.iloc[i+1]['layer']: continue
            state = (t2.iloc[i]['layer'], t2.iloc[i]['experts'][0])
            actual = t2.iloc[i+1]['experts'][0]
            
            if model[state]:
                pred = model[state].most_common(1)[0][0]
                if pred == actual:
                    correct += 1
            # else: strictly speaking, random guess is 1/8. 
            total += 1
            
    
    # Cross-Token Analysis
    print("\nITERATION 2: CROSS-TOKEN PATTERNS")
    all_dfs = [t1, t2, t3]
    all_dfs = [d for d in all_dfs if d is not None and not d.empty]
    
    if len(all_dfs) > 1:
        # 1. Layer Consistency
        # For each layer, what are the experts chosen across tokens?
        print("  Checking Layer Consistency (Do all tokens use same expert at Layer X?)")
        layer_consistency_count = 0
        total_layers = 32
        
        for layer in range(total_layers):
            experts_at_layer = []
            for d in all_dfs:
                rows = d[d['layer'] == layer]
                if not rows.empty:
                    experts_at_layer.append(rows.iloc[0]['experts'][0])
            
            # Check if all tokens used the same expert
            if len(set(experts_at_layer)) == 1:
                print(f"    Layer {layer}: Consistent Expert {experts_at_layer[0]}")
                layer_consistency_count += 1
            else:
                # precise breakdown
                # print(f"    Layer {layer}: {experts_at_layer}")
                pass
                
        print(f"  Total Consistent Layers: {layer_consistency_count}/{total_layers}")

        # 2. Global Expert Transition (Independent of Layer)
        # P(Next=B | Curr=A) across ALL layers and ALL tokens
        print("\n  Checking Global Expert Transition (Expert A -> Expert B potential)")
        global_trans = defaultdict(Counter)
        
        for d in all_dfs:
            for i in range(len(d) - 1):
                if d.iloc[i]['layer'] + 1 != d.iloc[i+1]['layer']: continue
                e_curr = d.iloc[i]['experts'][0]
                e_next = d.iloc[i+1]['experts'][0]
                global_trans[e_curr][e_next] += 1
        
        # Print top transitions
        print("  Top Global Transitions (Prob > 0.3):")
        for e_curr in range(8):
            total_out = sum(global_trans[e_curr].values())
            if total_out == 0: continue
            
            for e_next, count in global_trans[e_curr].most_common(3):
                prob = count / total_out
                if prob > 0.3:
                    print(f"    Expert {e_curr} -> {e_next}: {prob:.2%} ({count}/{total_out})")

        # 3. Top-2 Set Analysis (Iteration 3)
        print("\nITERATION 3: TOP-2 SET DYNAMICS")
        
        # Expert Stickiness: Probability that an expert in Layer N is also in Layer N+1
        # P(E in Top2_{N+1} | E in Top2_N)
        
        retained_count = 0
        total_slots = 0
        
        for d in all_dfs:
            for i in range(len(d) - 1):
                curr_set = set(d.iloc[i]['experts'])
                next_set = set(d.iloc[i+1]['experts'])
                
                # How many of curr_set are in next_set?
                intersection = curr_set.intersection(next_set)
                retained_count += len(intersection)
                total_slots += 2 # We consider 2 source experts per layer
                
        stickiness = retained_count / total_slots
        print(f"  Expert Stickiness (Prob. expert stays active in next layer): {stickiness:.2%}")
        
        # 4. Modulo / Periodicity by Tokens
        # Does expert choice depend on (Layer % 8)?
        # Let's visualize the "Active Experts" map
        print("\n  Expert Activity Heatmap (aggregated):")
        # We can simulate a heatmap by printing which experts are active > 50% of time at each layer
        for layer in range(32):
            active_counts = Counter()
            for d in all_dfs:
                rows = d[d['layer'] == layer]
                if not rows.empty:
                    active_counts.update(rows.iloc[0]['experts'])
            
            # Experts active in at least 2 out of 3 tokens
            common_experts = [e for e, c in active_counts.items() if c >= 2]
            if common_experts:
                print(f"    Layer {layer}: Typically uses {sorted(common_experts)}")

    # 5. Static Layer-wise Schedule Evaluation (Iteration 4)
    print("\nITERATION 4: STATIC LAYER-WISE SCHEDULE ACCURACY")
    # Train on T1+T2, Test on T3
    # Or Train on T1, Test on T2+T3
    
    # Let's do Leave-One-Out Cross Validation
    token_files = [t1, t2, t3]
    token_names = ['T1', 'T2', 'T3']
    
    avg_top1_hit = 0
    avg_top2_hit = 0
    
    for i in range(3):
        test_df = token_files[i]
        train_dfs = token_files[:i] + token_files[i+1:]
        
        if test_df is None or not train_dfs: continue
        
        # Build Profile from Train
        layer_profile = defaultdict(Counter)
        for d in train_dfs:
            for l in range(32):
                row = d[d['layer'] == l]
                if not rows.empty: # bug: rows is undefined here, use row instead
                    # Fixed:
                    pass
                if not row.empty:
                    layer_profile[l].update(row.iloc[0]['experts'])
        
        # Predict on Test
        hits_top1 = 0
        hits_top2 = 0
        total_layers = 32
        
        for l in range(32): # Iterate layers
            # My prediction: The most common experts in training
            if not layer_profile[l]:
                # No data, random guess?
                preds = [0, 1] 
            else:
                preds = [e for e, c in layer_profile[l].most_common(2)]
                # Ensure we have 2
                while len(preds) < 2:
                    preds.append(0) # fallback
            
            # Ground Truth
            actual_row = test_df[test_df['layer'] == l]
            if actual_row.empty: continue
            
            actual_experts = actual_row.iloc[0]['experts'] # The Top-2 list
            
            # Check overlap
            # "Speculative Success": If we pre-load our predicted Top-1, is it used?
            # Or if we pre-load Top-2, are they used?
            
            # Metric 1: Does our Prediction[0] appear in Actual[0] or Actual[1]?
            if preds[0] in actual_experts:
                hits_top1 += 1
                
            # Metric 2: Intersection size (0, 1, or 2)
            intersection = len(set(preds).intersection(set(actual_experts)))
            hits_top2 += intersection
            
        acc_top1 = hits_top1 / 32
        # For Top2, max possible "hits" is 64 (2 per layer). Or relative to 2?
        # Let's say "Recall": How many of the needed experts did we predict?
        recall_top2 = hits_top2 / (32 * 2)
        
        print(f"  Test on {token_names[i]}: Top-1 Recall: {acc_top1:.2%},  Top-2 Recall: {recall_top2:.2%}")
        avg_top1_hit += acc_top1
        avg_top2_hit += recall_top2
        
    print(f"  Average Top-1 Recall: {avg_top1_hit/3:.2%}")
    print(f"  Average Top-2 Recall: {avg_top2_hit/3:.2%}")





