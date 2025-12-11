import torch
import torch.nn as nn

class ExpertPredictorMLP(nn.Module):
    def __init__(self, num_layers=32, num_experts=8, embed_dim_layer=10, embed_dim_expert=5, 
                 hidden_dims=[256, 128, 64], dropout=0.2):
        super(ExpertPredictorMLP, self).__init__()
        
        # 1. Embeddings
        self.layer_embedding = nn.Embedding(num_layers, embed_dim_layer)
        
        # 9 embeddings: 0=Padding, 1-8=Experts (mapped from 0-7)
        self.expert_embedding = nn.Embedding(num_experts + 1, embed_dim_expert)
        
        # 2. Input Calc
        # Layer + Secondary + History (32 * expert_dim) + Gating (2) + Pos (1)
        input_dim = embed_dim_layer + embed_dim_expert + (32 * embed_dim_expert) + 2 + 1
        
        # 3. Dynamic Network Architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_experts))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, layer, history, secondary, gating, pos):
        """
        Args:
            layer: (B)
            history: (B, 32) - Full path history IDs (0=pad, 1-8=experts)
            secondary: (B)
            gating: (B, 2)
            pos: (B, 1)
        """
        # Embeddings
        layer_emb = self.layer_embedding(layer)           # (B, 10)
        sec_emb = self.expert_embedding(secondary + 1)    # (B, 5) Note: +1 because 0 is pad
        
        # History embedding flattened
        hist_emb = self.expert_embedding(history).view(history.size(0), -1) # (B, 32*5 = 160)
        
        # Concatenate ALL features
        # [10 + 5 + 160 + 2 + 1] = 178 features
        x = torch.cat([layer_emb, sec_emb, hist_emb, gating, pos], dim=1)
        
        logits = self.net(x)
        return logits
