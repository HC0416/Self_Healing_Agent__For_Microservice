"""
model/encoder.py
================
GNN Encoder

Supports 3 encoder types selectable via MODEL_TYPE:
  "mlp"  — M1: No graph structure (MLP baseline)
  "gcn"  — M2: Simple GCNConv (graph but no gating)
  "ggnn" — M3/M4: GatedGraphConv (full model, best)

Input:  x           [N, 38]   node features
        edge_index  [2, E]    edges
Output: node_emb    [N, 64]   one embedding per node
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, GCNConv


class MLPEncoder(nn.Module):
    """
    M1 Baseline — no graph structure.
    Treats each span independently, ignores edges entirely.
    Shows that graph structure is important.
    """
    def __init__(self, in_channels=38, hidden=64, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        # Ignores edge_index entirely
        return self.norm(self.net(x))


class GCNEncoder(nn.Module):
    """
    M2 Baseline — simple GCNConv (no gating).
    Uses graph structure but without gating mechanism.
    Shows that GatedGraphConv > simple GCN.
    """
    def __init__(self, in_channels=38, hidden=64,
                 num_layers=3, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList([
            GCNConv(hidden, hidden) for _ in range(num_layers)
        ])
        self.acts = nn.ModuleList([
            nn.ReLU() for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        for conv, act in zip(self.convs, self.acts):
            if edge_index.shape[1] > 0:
                h = conv(h, edge_index)
            h = act(h)
        return self.norm(h)


class GNNEncoder(nn.Module):
    """
    M3/M4 Full model — GatedGraphConv (best).
    Uses gating mechanism to control information flow.
    M3 = trained with mixed data (normal + anomalous)
    M4 = trained with normal-only data (our best model)
    """
    def __init__(self, in_channels=38, hidden=64,
                 num_layers=3, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.ggnn       = GatedGraphConv(out_channels=hidden,
                                         num_layers=num_layers)
        self.norm       = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        if edge_index.shape[1] > 0:
            h = self.ggnn(h, edge_index)
        return self.norm(h)


def build_encoder(model_type="ggnn", in_channels=38,
                  hidden=64, num_layers=3):
    """
    Factory function — returns the right encoder.

    Args:
        model_type: "mlp" | "gcn" | "ggnn"
        in_channels: node feature dimension
        hidden: hidden dimension
        num_layers: number of GNN layers (ignored for MLP)
    """
    encoders = {
        "mlp":  MLPEncoder,
        "gcn":  GCNEncoder,
        "ggnn": GNNEncoder,
    }
    if model_type not in encoders:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: {list(encoders.keys())}")
    cls = encoders[model_type]
    return cls(in_channels=in_channels, hidden=hidden,
               num_layers=num_layers)
