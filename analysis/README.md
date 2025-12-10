# Expert Prediction for Mixtral 8x7B - Speculative Pipeline Parallelism

## Goal
Predict which expert will be selected at Layer N+1 based on expert decisions at Layer N (or earlier), enabling speculative pre-loading of expert weights.

## Model Info
- **Model**: Mixtral 8x7B (Mixture of Experts)
- **Layers**: 32 MoE layers
- **Experts per layer**: 8
- **Top-K routing**: 2 experts selected per token per layer

---

## Files

### Data Files (20 token journeys)
| File | Description |
|------|-------------|
| `p0_token_0.jsonl` - `p0_token_9.jsonl` | 10 tokens from HumanEval problem 0 |
| `p1_token_0.jsonl` - `p1_token_9.jsonl` | 10 tokens from HumanEval problem 1 |

Each file contains 32 entries (one per layer) with expert routing decisions.

### Reusable Scripts
| Script | Description |
|--------|-------------|
| `create_rules.py` | Create rules from specified token journey files |
| `evaluate_rules.py` | Evaluate rules on specified test files |
| `extract_tokens.py` | Extract token journeys from raw routing data |

### Rule Files
| File | Description |
|------|-------------|
| `rules_from_1token.json` | Rules from p0_token_0 only (31 rules) |
| `rules_from_5tokens.json` | Rules from 5 tokens (majority vote) |

---

## 5 Prediction Levels

### Level 1: Simple (Layer, Expert) → Next Expert
```
Input:  Layer N, Expert E
Output: Predicted Expert at Layer N+1

Example: Layer 5, Expert 6 → Expert 3
```
**Accuracy: 44.9%** (31/69 correct)

### Level 2: Two Layers Back
```
Input:  Layer N, Previous Expert, Current Expert
Output: Predicted Expert at Layer N+1

Example: Layer 5, (Prev=0, Curr=6) → Expert 3
```
**Accuracy: 83.3%** (25/30 correct)

### Level 3: Three Layers Back (BEST)
```
Input:  Layer N, Expert at N-2, Expert at N-1, Expert at N
Output: Predicted Expert at Layer N+1

Example: Layer 5, (E-2=3, E-1=0, E=6) → Expert 3
```
**Accuracy: 87.5%** (21/24 correct) 🏆

### Level 4: Pair-Based (Primary + Secondary)
```
Input:  Layer N, Primary Expert, Secondary Expert
Output: Predicted Primary Expert at Layer N+1

Example: Layer 5, (Primary=6, Secondary=0) → Expert 3
```
**Accuracy: 65.7%** (23/35 correct)

### Level 5: Skip Layers
```
Input:  Layer N, Expert E
Output: Predicted Expert at Layer N+K (K=2,4,8)

Example: Layer 5, Expert 6 → Expert at Layer 7
```
**Accuracy: 48-55%**

---

## Results Summary

| Level | Method | Accuracy | vs Random (12.5%) |
|-------|--------|----------|-------------------|
| 1 | (Layer, Expert) | 44.9% | 3.6x better |
| 2 | (Layer, Prev, Curr) | 83.3% | 6.7x better |
| **3** | **(Layer, E-2, E-1, E)** | **87.5%** | **7x better** |
| 4 | (Layer, Prim, Sec) | 65.7% | 5.3x better |
| 5 | Skip layers | 48-55% | 4x better |

---

## How to Run (Reusable Scripts)

### Script 1: `extract_tokens.py` - Extract Token Journeys from Raw Data

```bash
# Usage
python extract_tokens.py <input_file> --problem <id> --tokens <start> <end> [--output-dir <dir>]

# Arguments:
#   input_file          Raw routing JSONL file (required)
#   --problem, -p       Problem ID to extract (required)
#   --tokens, -t        Token range [start, end) (required)
#   --output-dir, -o    Output directory (default: current dir)
```

**Examples:**
```bash
# Single problem, tokens 0-9
python extract_tokens.py ../humaneval_2_routing.jsonl --problem 0 --tokens 0 10

# Different problem
python extract_tokens.py ../humaneval_2_routing.jsonl --problem 1 --tokens 0 10

# Custom output directory
python extract_tokens.py ../humaneval_2_routing.jsonl -p 0 -t 0 10 --output-dir ./data/

# Extract tokens 5-14 from problem 2
python extract_tokens.py routing.jsonl -p 2 -t 5 15 -o ./problem2_data/
```

---

### Script 2: `create_rules.py` - Create Prediction Rules

```bash
# Usage
python create_rules.py <file1> [file2] [file3] ... --output <rules.json>

# Arguments:
#   train_files         One or more token journey files (required)
#   --output, -o        Output JSON file (default: rules.json)
```

**Examples:**
```bash
# Single file (1 token)
python create_rules.py p0_token_0.jsonl --output rules_1token.json

# Multiple specific files
python create_rules.py p0_token_0.jsonl p0_token_1.jsonl p0_token_2.jsonl --output rules_3tokens.json

# Wildcard - all problem 0 tokens
python create_rules.py p0_token_*.jsonl --output rules_p0_all.json

# Wildcard - all tokens from both problems
python create_rules.py p0_token_*.jsonl p1_token_*.jsonl -o rules_all.json

# Mix of specific and wildcard
python create_rules.py p0_token_0.jsonl p1_token_*.jsonl --output rules_mixed.json
```

---

### Script 3: `evaluate_rules.py` - Evaluate Rules on Test Data

```bash
# Usage
python evaluate_rules.py --rules <rules.json> --test <file1> [file2] ... [--output results.json]

# Arguments:
#   --rules, -r         Rules JSON file from create_rules.py (required)
#   --test, -t          One or more test files (required)
#   --output, -o        Output JSON file to save results (optional)
```

**Examples:**
```bash
# Single test file
python evaluate_rules.py --rules rules.json --test p0_token_1.jsonl

# Multiple specific test files
python evaluate_rules.py --rules rules.json --test p0_token_1.jsonl p0_token_2.jsonl p0_token_3.jsonl

# Wildcard - all problem 1 tokens
python evaluate_rules.py --rules rules.json --test p1_token_*.jsonl

# Wildcard - all tokens
python evaluate_rules.py -r rules.json -t p0_token_*.jsonl p1_token_*.jsonl

# Save results to JSON file
python evaluate_rules.py -r rules.json -t p1_token_*.jsonl --output results.json
```

**Output JSON format (when using --output):**
```json
{
  "metadata": {
    "timestamp": "2025-12-10T04:46:48.826806",
    "rules_file": "test_rules.json",
    "training_files": ["p0_token_0.jsonl"],
    "test_files": ["p1_token_0.jsonl", "p1_token_1.jsonl", ...],
    "num_test_files": 11
  },
  "results": {
    "level1": {"correct": 32, "total": 75, "accuracy_percent": 42.67, "vs_random": 3.41},
    "level2": {"correct": 22, "total": 31, "accuracy_percent": 70.97, "vs_random": 5.68},
    "level3": {"correct": 18, "total": 21, "accuracy_percent": 85.71, "vs_random": 6.86},
    ...
  },
  "best_level": {"name": "level3", "accuracy_percent": 85.71}
}
```

---

### Complete Workflow Example

```bash
# Step 1: Extract tokens from raw data
python extract_tokens.py ../humaneval_2_routing.jsonl --problem 0 --tokens 0 10
python extract_tokens.py ../humaneval_2_routing.jsonl --problem 1 --tokens 0 10

# Step 2: Create rules from training token(s)
python create_rules.py p0_token_0.jsonl --output rules.json

# Step 3: Evaluate on test tokens
python evaluate_rules.py --rules rules.json --test p0_token_1.jsonl p0_token_2.jsonl p1_token_*.jsonl
```

### Example Output
```
EVALUATION RESULTS
Level 1 (Layer, E)             8          22         36.4%        2.9x
Level 2 (Layer, Prev, Curr)    20         29         69.0%        5.5x
Level 3 (Layer, E-2, E-1, E)   21         24         87.5%        7.0x
Level 4 (Layer, P, S)          10         22         45.5%        3.6x

🏆 BEST: level3 with 87.5% accuracy
```

---

## Methodology

### Step 1: Create Rules from Token 0
```
Token 0 journey: [6, 6, 7, 3, 0, 6, 3, 0, 7, 6, 0, 7, ...]

Level 1 Rule Creation:
  Layer 0: Expert 6 → Layer 1: Expert 6
  RULE: (Layer=0, Expert=6) → Next=6

  Layer 1: Expert 6 → Layer 2: Expert 7
  RULE: (Layer=1, Expert=6) → Next=7

  ... (31 rules total)
```

### Step 2: Validate on Tokens 1-9
```
Token 1 at Layer 5: Expert 1
  → Check rule for (Layer=5, Expert=1)
  → No rule exists (Token 0 had Expert 6 at Layer 5)
  → Skip

Token 1 at Layer X: Expert 6
  → Check rule for (Layer=X, Expert=6)
  → Rule exists! Predicts Expert Y
  → Compare with actual → Count correct/wrong
```

### Step 3: Calculate Accuracy
```
Accuracy = Correct Predictions / Total Tested
         = 31 / 69 = 44.9% (Level 1)
```

---

## Key Insights

1. **More context = Higher accuracy**
   - Level 1 (1 expert): 44.9%
   - Level 2 (2 experts): 83.3%
   - Level 3 (3 experts): 87.5%

2. **More context = Fewer matches**
   - Level 1: 69 transitions tested
   - Level 2: 30 transitions tested
   - Level 3: 24 transitions tested

3. **Recommended: Hybrid Approach**
   ```
   1. Try Level 3 rule first (87.5% accurate)
   2. If no rule, try Level 2 (83.3% accurate)
   3. If no rule, try Level 1 (44.9% accurate)
   4. If no rule, use most common expert for that layer
   ```

---

## Practical Application

For Speculative Pipeline Parallelism:
1. At Layer N, observe which expert was selected
2. Look up prediction rule for Layer N+1
3. Pre-load predicted expert weights BEFORE router computes
4. If prediction correct (87.5% with Level 3): Zero wait time!
5. If prediction wrong: Fall back to normal loading (no worse than baseline)

---

## Data Format

Each `.jsonl` file contains 32 entries (one per layer):
```json
{
  "layer": 5,
  "experts": [6, 0],      // [primary, secondary]
  "gating_probs": [0.85, 0.15],
  "token_idx": 0
}
```
