"""
model/svdd_head.py
==================
Deep SVDD Head

Input:  node_emb  [N, 64]
        batch     [N]
Output: scores    [B]      anomaly score per graph
        z         [B, 32]  latent embedding
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class SVDDHead(nn.Module):

    def __init__(self, in_channels=64, latent=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, latent),
        )
        self.register_buffer("centre", torch.zeros(latent))
        self.latent = latent

    def forward(self, node_emb, batch):
        graph_emb = global_mean_pool(node_emb, batch)   # [B, 64]
        z         = self.mlp(graph_emb)                  # [B, 32]
        scores    = torch.sum((z - self.centre) ** 2, dim=1)  # [B]
        return scores, z

    def init_centre(self, encoder, loader, device, eps=0.1):
        self.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                batch     = batch.to(device)
                node_emb  = encoder(batch.x, batch.edge_index)
                graph_emb = global_mean_pool(node_emb, batch.batch)
                z         = self.mlp(graph_emb)
                all_z.append(z)
        c = torch.cat(all_z, dim=0).mean(dim=0)
        c[(c.abs() < eps) & (c >= 0)] =  eps
        c[(c.abs() < eps) & (c <  0)] = -eps
        self.centre = c
        print(f"  Centre initialised  norm={c.norm():.4f}")
