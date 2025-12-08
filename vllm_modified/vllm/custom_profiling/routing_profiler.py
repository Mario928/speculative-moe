"""
Routing Profiler for MoE Models

Records routing decisions (which experts are selected) at each layer
for cross-layer predictability analysis.
"""
import json
import os
from typing import Optional

import torch


class RoutingProfiler:
    """
    Records MoE routing decisions for analysis.
    
    Captures: layer, token_id, expert_ids, weights, problem_id
    """
    
    _instance: Optional['RoutingProfiler'] = None
    
    def __init__(self):
        # controlled by environment variable
        self.enabled = os.getenv("ROUTING_PROFILER_ENABLED", "0") == "1"
        self.data = []
        self.current_problem_id = 0
        self._record_count = 0
        # Debug: print status on init
        print(f"[RoutingProfiler] Initialized. enabled={self.enabled}, env={os.getenv('ROUTING_PROFILER_ENABLED', 'NOT SET')}")
    
    @classmethod
    def get_instance(cls) -> 'RoutingProfiler':
        """Returns single shared instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = RoutingProfiler()
        return cls._instance
    
    def set_problem_id(self, problem_id: int):
        """Call this before running inference on each problem."""
        self.current_problem_id = problem_id
    
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
        self._record_count += 1
        
        # Debug: print first few calls
        if self._record_count <= 5:
            print(f"[RoutingProfiler] record() called #{self._record_count}: layer={layer_name}, enabled={self.enabled}")
        
        if not self.enabled:
            return
        
        # parse layer number from layer_name
        # "model.layers.5.block_sparse_moe" -> 5
        try:
            layer_idx = int(layer_name.split("layers.")[1].split(".")[0])
        except:
            layer_idx = -1
        
        # move tensors to CPU for logging
        ids = topk_ids.detach().cpu().tolist()
        weights = topk_weights.detach().cpu().tolist()
        
        # record each token
        for token_idx in range(len(ids)):
            entry = {
                "problem_id": self.current_problem_id,
                "layer": layer_idx,
                "token_id": token_idx,
                "expert_ids": ids[token_idx],
                "weights": weights[token_idx],
            }
            self.data.append(entry)
    
    def save(self, filepath: str):
        """Save recorded data to JSON file."""
        print(f"[RoutingProfiler] save() called. enabled={self.enabled}, record_count={self._record_count}, data_len={len(self.data)}")
        
        if not self.data:
            print("No routing data to save.")
            return
        
        with open(filepath, 'w') as f:
            for entry in self.data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved {len(self.data)} routing entries to {filepath}")


def get_profiler() -> RoutingProfiler:
    """Get the routing profiler instance."""
    return RoutingProfiler.get_instance()
