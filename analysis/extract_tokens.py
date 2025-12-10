#!/usr/bin/env python3
"""
EXTRACT TOKEN JOURNEYS FROM RAW ROUTING DATA

Usage:
    python extract_tokens.py <input_file> --problem <id> --tokens <start> <end> [--output-dir <dir>]

Examples:
    # Extract tokens 0-9 from problem 0
    python extract_tokens.py ../humaneval_2_routing.jsonl --problem 0 --tokens 0 10

    # Extract tokens 0-9 from problem 1, output to custom dir
    python extract_tokens.py routing_data.jsonl --problem 1 --tokens 0 10 --output-dir ./data/

    # Extract tokens 5-15 from problem 2
    python extract_tokens.py data.jsonl --problem 2 --tokens 5 15

Output: p{problem}_token_{idx}.jsonl files (32 layers each)
"""

import argparse
import json
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Extract token journeys from raw routing data')
    parser.add_argument('input_file', help='Raw routing JSONL file (e.g., humaneval_2_routing.jsonl)')
    parser.add_argument('--problem', '-p', type=int, required=True, help='Problem ID to extract (e.g., 0, 1, 2)')
    parser.add_argument('--tokens', '-t', type=int, nargs=2, required=True, metavar=('START', 'END'),
                        help='Token range [start, end) - e.g., "0 10" for tokens 0-9')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory (default: current dir)')
    args = parser.parse_args()

    print("="*60)
    print("EXTRACTING TOKEN JOURNEYS")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Problem ID: {args.problem}")
    print(f"Token range: {args.tokens[0]} to {args.tokens[1]-1}")
    print(f"Output dir: {args.output_dir}")

    # Create output directory if needed
    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    # Load raw data
    print(f"\nLoading {args.input_file}...")
    with open(args.input_file) as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Total raw entries: {len(data)}")

    # De-duplicate: keep first entry for each (problem_id, token_idx, layer)
    seen = set()
    clean = []
    for d in data:
        key = (d['problem_id'], d['token_idx'], d['layer'])
        if key not in seen:
            seen.add(key)
            clean.append(d)
    print(f"After de-duplication: {len(clean)} entries")

    # Extract journeys for specified problem and token range
    print(f"\n" + "-"*60)
    print(f"PROBLEM {args.problem}: Extracting tokens {args.tokens[0]}-{args.tokens[1]-1}")
    print("-"*60)

    extracted = 0
    skipped = 0

    for token_idx in range(args.tokens[0], args.tokens[1]):
        # Get journey for this token
        journey = [d for d in clean if d['problem_id'] == args.problem and d['token_idx'] == token_idx]
        journey.sort(key=lambda x: x['layer'])

        if len(journey) == 32:
            experts = [j['experts'][0] for j in journey]

            # Save to file
            fname = os.path.join(args.output_dir, f"p{args.problem}_token_{token_idx}.jsonl")
            with open(fname, 'w') as f:
                for entry in journey:
                    f.write(json.dumps(entry) + '\n')

            print(f"  Token {token_idx}: 32 layers -> {fname}")
            print(f"    Experts: {experts}")
            extracted += 1
        else:
            print(f"  Token {token_idx}: {len(journey)} layers (SKIPPED - need 32)")
            skipped += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Extracted: {extracted} token journeys")
    print(f"Skipped: {skipped} (incomplete)")
    print(f"\nOutput files: p{args.problem}_token_*.jsonl in {args.output_dir}")

if __name__ == "__main__":
    main()
