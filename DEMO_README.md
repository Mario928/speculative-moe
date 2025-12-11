# MoE Routing Analysis Demo

Interactive notebook for exploring expert routing patterns in Mixtral-8x7B.

## What This Is

We collected routing decisions from Mixtral-8x7B while it solved 244 coding and math problems. Each token the model generates passes through 32 layers, and at each layer, a "gating network" picks 2 out of 8 experts to process that token.

This notebook lets you explore those patterns:
- Which experts get picked at each layer?
- Can we predict the next layer's expert from the current one?
- Are there common "paths" tokens take through the 32 layers?
- Do code problems route differently than math problems?

## Quick Start

```bash
jupyter notebook moe_routing_analysis_demo.ipynb
```

Run cells from top to bottom. First run takes ~3 minutes to load data and compute statistics.

## What's Inside

### Section 1: Routing Dashboard
Watch individual tokens travel through all 32 layers. Pick any token from HumanEval or GSM8K and see which experts it picked and how confident the gating was.

### Section 2: Statistical Analysis
- **32x32 Heatmap**: Shows how predictable each layer transition is
- **Hop Accuracy Curve**: Can we predict 1 layer ahead? 5 layers? 31 layers?
- **Path Frequency**: The most common 32-layer expert sequences

### Section 3: Model Comparison
How well do different methods predict the next expert?
- Random guessing: 12.5%
- Lookup tables: 42%
- Our trained MLP: **55.6% Top-1**, **73.2% Top-2** — accuracy improvement of +345% (4.5×)

### Section 4: HumanEval vs GSM8K
Code generation and math reasoning use experts differently. Side-by-side heatmaps show which experts prefer which task type.

## Data

All visualizations use data from `routing_data_collected/`:
- 29,463 token journeys from HumanEval (coding)
- 14,820 token journeys from GSM8K (math)
- Each journey = 32 layers × 2 experts per layer

## Requirements

```
numpy
pandas  
matplotlib
seaborn
ipywidgets (optional, for interactive features)
```

## Notes

- The notebook samples 5,000 tokens by default for speed. Edit `SAMPLE_SIZE` to use more.
- Some cells take 30-60 seconds (computing 32x32 matrices over thousands of tokens).
- Interactive widgets work in Jupyter Notebook/Lab but not in VS Code's notebook viewer.
