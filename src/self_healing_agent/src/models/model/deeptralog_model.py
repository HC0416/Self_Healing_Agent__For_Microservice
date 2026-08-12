"""
model/deeptralog_model.py
=========================
Full DeepTraLog model — assembles all components.

Supports 4 model configurations via MODEL_TYPE:
  M1: "mlp"  — MLP encoder (no graph)
  M2: "gcn"  — GCN encoder (graph, no gating)
  M3: "ggnn" + normal_only_train=False (mixed training)
  M4: "ggnn" + normal_only_train=True  (our best model)

Stage 1: Encoder (MLP / GCN / GGNN)
Stage 2:   Localizer
Stage 3:   SVDDHead
"""

import torch
import torch.nn as nn
from .encoder import build_encoder
from .localizer import Localizer
from .svdd_head import SVDDHead


class DeepTraLogModel(nn.Module):

    def __init__(self, in_channels=38, hidden=64,
                 latent=32, num_layers=3,
                 model_type="ggnn"):
        super().__init__()
        self.encoder   = build_encoder(
            model_type=model_type,
            in_channels=in_channels,
            hidden=hidden,
            num_layers=num_layers,
        )
        self.localizer = Localizer(hidden)
        self.svdd      = SVDDHead(hidden, latent)
        self.model_type = model_type

    def forward(self, data):
        """
        Returns:
            scores      [B]         anomaly score per graph
            z           [B, latent] latent embeddings
            node_scores [N, 1]      root cause scores
        """
        node_emb    = self.encoder(data.x, data.edge_index)
        node_scores = self.localizer(node_emb, data.batch)
        scores, z   = self.svdd(node_emb, data.batch)
        return scores, z, node_scores

    def init_centre(self, loader, device, eps=0.1):
        self.svdd.init_centre(self.encoder, loader, device, eps)
