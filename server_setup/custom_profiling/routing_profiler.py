"""
Routing Profiler for MoE Models

Records routing decisions (which experts are selected) at each layer
for cross-layer predictability analysis in speculative decoding research.

Updated with learnings:
- Filters padding tokens (uniform weights ~0.5, 0.5)
- Uses 0-indexed token_idx
- Renames weights to gating_probs for clarity
"""
import json
import os
from typing import Optional
import torch


class RoutingProfiler:
    """
    Records MoE routing decisions for analysis.
    
    Captures: dataset, problem_id, layer, token_idx, experts, gating_probs
    """
    
    _instance: Optional['RoutingProfiler'] = None
    
    def __init__(self):
        self.enabled = os.getenv("ROUTING_PROFILER_ENABLED", "0") == "1"
        self.output_path = os.getenv("ROUTING_OUTPUT_PATH", "routing_raw.jsonl")
        self.current_problem_id = 0
        self.current_dataset = "unknown"
    
    @classmethod
    def get_instance(cls) -> 'RoutingProfiler':
        """Returns single shared instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = RoutingProfiler()
        return cls._instance
    
    def set_context(self, problem_id: int, dataset: str = "unknown"):
        """Call this before running inference on each problem."""
        self.current_problem_id = problem_id
        self.current_dataset = dataset
    
    def record(
        self,
        layer_name: str,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """
        Record routing decisions for all tokens at this layer.
        
        Args:
            layer_name: string like "model.layers.5.block_sparse_moe"
            topk_ids: shape [num_tokens, k] - selected expert indices
            topk_weights: shape [num_tokens, k] - gating probabilities
        """
        if not self.enabled:
            return
        
        # Parse layer number from layer_name
        # "model.layers.5.block_sparse_moe" -> 5
        try:
            layer_idx = int(layer_name.split("layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return  # Skip if can't parse layer
        
        # Move tensors to CPU
        ids = topk_ids.detach().cpu().tolist()
        weights = topk_weights.detach().cpu().tolist()
        
        # Write directly to file (avoids memory buildup)
        with open(self.output_path, "a") as f:
            for token_idx, (expert_ids, gating_probs) in enumerate(zip(ids, weights)):
                # Filter padding: skip if weights are uniform (~0.5, 0.5)
                if len(gating_probs) == 2:
                    if abs(gating_probs[0] - 0.5) < 0.01 and abs(gating_probs[1] - 0.5) < 0.01:
                        continue
                
                entry = {
                    "dataset": self.current_dataset,
                    "problem_id": self.current_problem_id,
                    "layer": layer_idx,
                    "token_pos": token_idx,  # Position in batch
                    "experts": expert_ids,
                    "gating_probs": gating_probs,
                }
                f.write(json.dumps(entry) + "\n")
    
    def clear_output(self):
        """Clear the output file (call before starting new collection)."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)


def get_profiler() -> RoutingProfiler:
    """Get the routing profiler instance."""
    return RoutingProfiler.get_instance()
