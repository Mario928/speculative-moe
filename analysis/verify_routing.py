"""
Verify MoE Routing Data Quality

Usage:
    python verify_routing.py <routing_file.jsonl>
"""
import json
import sys
from collections import Counter

def verify(filepath):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*60}")
    
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        print("ERROR: No data found!")
        return
    
    # Basic counts
    total = len(data)
    datasets = set(d.get('dataset', 'MISSING') for d in data)
    problems = sorted(set(d.get('problem_id', -1) for d in data))
    layers = sorted(set(d.get('layer', -1) for d in data))
    
    print(f"\nTotal records: {total}")
    print(f"Dataset values: {datasets}")
    print(f"Problem IDs: {min(problems)} - {max(problems)} ({len(problems)} unique)")
    print(f"Layers: {min(layers)} - {max(layers)} ({len(layers)} unique)")
    
    # Check if all 32 layers present
    expected_layers = set(range(32))
    missing_layers = expected_layers - set(layers)
    if missing_layers:
        print(f"❌ MISSING LAYERS: {sorted(missing_layers)}")
    else:
        print(f"✅ All 32 layers present")
    
    # Records per layer
    layer_counts = Counter(d['layer'] for d in data)
    min_count = min(layer_counts.values())
    max_count = max(layer_counts.values())
    avg_count = sum(layer_counts.values()) / len(layer_counts)
    print(f"\nRecords per layer: {min_count} - {max_count} (avg: {avg_count:.1f})")
    
    # Calculate expected
    num_problems = len(problems)
    # Get max token per problem per layer
    max_tokens = {}
    for d in data:
        key = (d['problem_id'], d['layer'])
        max_tokens[key] = max(max_tokens.get(key, 0), d.get('token_idx', 0) + 1)
    
    # Sum expected
    total_expected = sum(max_tokens.values())
    print(f"Expected records (based on max token_idx): ~{total_expected}")
    
    capture_rate = 100 * total / total_expected if total_expected > 0 else 0
    if capture_rate >= 99:
        print(f"✅ Capture rate: {capture_rate:.1f}%")
    elif capture_rate >= 95:
        print(f"⚠️ Capture rate: {capture_rate:.1f}% (some tokens filtered)")
    else:
        print(f"❌ Capture rate: {capture_rate:.1f}% (significant data loss!)")
    
    # Data quality checks
    print(f"\n{'='*60}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*60}")
    
    # Check required fields
    required = ['dataset', 'problem_id', 'layer', 'experts', 'gating_probs', 'token_idx']
    sample = data[0]
    missing = [f for f in required if f not in sample]
    if missing:
        print(f"❌ Missing fields: {missing}")
    else:
        print(f"✅ All required fields present")
    
    # Check experts format
    bad_experts = sum(1 for d in data if not isinstance(d.get('experts'), list) or len(d.get('experts', [])) != 2)
    if bad_experts:
        print(f"❌ Bad experts format: {bad_experts} records")
    else:
        print(f"✅ Experts format correct (all have 2 experts)")
    
    # Check gating probs sum
    bad_probs = 0
    for d in data:
        probs = d.get('gating_probs', [])
        if len(probs) == 2 and not (0.99 < sum(probs) < 1.01):
            bad_probs += 1
    if bad_probs:
        print(f"❌ Gating probs don't sum to 1.0: {bad_probs} records")
    else:
        print(f"✅ Gating probs sum to ~1.0")
    
    # Check for padding tokens (exact 0.5, 0.5)
    padding = sum(1 for d in data if d.get('gating_probs') == [0.5, 0.5])
    if padding:
        print(f"⚠️ Padding tokens found: {padding} (should be filtered)")
    else:
        print(f"✅ No padding tokens [0.5, 0.5]")
    
    # Routing analysis
    print(f"\n{'='*60}")
    print("ROUTING PATTERNS")
    print(f"{'='*60}")
    
    # Expert usage
    expert_freq = Counter()
    for d in data:
        for e in d.get('experts', []):
            expert_freq[e] += 1
    
    print("\nExpert usage distribution:")
    total_expert_refs = sum(expert_freq.values())
    for e in range(8):
        pct = 100 * expert_freq.get(e, 0) / total_expert_refs if total_expert_refs > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  Expert {e}: {pct:5.1f}% {bar}")
    
    # Top-1 confidence
    avg_top1 = sum(d['gating_probs'][0] for d in data if 'gating_probs' in d) / total
    print(f"\nAverage top-1 expert probability: {avg_top1:.1%}")
    
    # Top pairs
    pair_freq = Counter(tuple(sorted(d['experts'])) for d in data if 'experts' in d)
    print("\nTop 5 expert pairs:")
    for pair, count in pair_freq.most_common(5):
        pct = 100 * count / total
        print(f"  {pair}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    
    issues = []
    if missing_layers:
        issues.append("missing layers")
    if 'unknown' in datasets or 'MISSING' in datasets:
        issues.append("unknown dataset")
    if capture_rate < 95:
        issues.append("low capture rate")
    if bad_experts or bad_probs:
        issues.append("format errors")
    
    if not issues:
        print("✅ DATA LOOKS CORRECT AND COMPLETE!")
    else:
        print(f"❌ ISSUES FOUND: {', '.join(issues)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_routing.py <routing_file.jsonl>")
        sys.exit(1)
    
    verify(sys.argv[1])
