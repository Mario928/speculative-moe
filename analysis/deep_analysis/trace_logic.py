#!/usr/bin/env python3
"""Trace through post-processing logic to understand missing records."""
import json

# Load raw data
raw = []
with open('routing_raw.jsonl') as f:
    for line in f:
        raw.append(json.loads(line))

print('='*60)
print('TRACING POST-PROCESSING LOGIC')
print('='*60)
print(f'Total raw records: {len(raw)}')
print()

results = []
for layer in range(32):
    layer_records = [d for d in raw if d['layer'] == layer]
    
    # Find decode_start: first token_pos=0 AFTER a record with token_pos > 0
    decode_start = None
    for i, d in enumerate(layer_records):
        if i > 0 and layer_records[i-1]['token_pos'] > 0 and d['token_pos'] == 0:
            decode_start = i
            break
    
    if decode_start is None:
        print(f'Layer {layer}: decode_start = None - ERROR!')
        continue
    
    # Find decode_end: next token_pos > 0 after decode_start
    decode_end = len(layer_records)
    for i in range(decode_start + 1, len(layer_records)):
        if layer_records[i]['token_pos'] > 0:
            decode_end = i
            break
    
    decode_tokens = layer_records[decode_start:decode_end]
    results.append({
        'layer': layer,
        'total': len(layer_records),
        'decode_start': decode_start,
        'decode_end': decode_end,
        'extracted': len(decode_tokens),
        'prefill_last': layer_records[decode_start-1]['token_pos'],
    })

print('Layer  Total  Start  End  Extracted  Prefill_last  Expected  Missing')
print('-'*75)
for r in results:
    expected = 83  # We know 83 tokens were generated
    missing = expected - r['extracted']
    status = '' if missing == 0 else f'  <- MISSING {missing}'
    print(f"{r['layer']:5d}  {r['total']:5d}  {r['decode_start']:5d}  {r['decode_end']:3d}  {r['extracted']:9d}  {r['prefill_last']:12d}  {expected:8d}  {missing:7d}{status}")

print()
print('='*60)
print('KEY INSIGHT')
print('='*60)

# Check what decode_end looks like
print()
print('For layers with missing, check if decode_end hit end of file:')
for r in results:
    if r['extracted'] < 83:
        layer = r['layer']
        layer_records = [d for d in raw if d['layer'] == layer]
        if r['decode_end'] < len(layer_records):
            print(f"  Layer {layer}: decode_end={r['decode_end']}, layer_records={len(layer_records)}")
            print(f"    Record at decode_end: token_pos={layer_records[r['decode_end']]['token_pos']}")
        else:
            print(f"  Layer {layer}: decode_end={r['decode_end']} (END OF LAYER DATA)")
