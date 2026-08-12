"""
model/localizer.py
==================
Root Cause Localizer

Input:  node_emb  [N, 64]
        batch     [N]
Output: scores    [N, 1]   anomaly contribution per node
"""

import torch
import torch.nn as nn


class Localizer(nn.Module):

    def __init__(self, hidden=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, node_emb, batch):
        scores     = self.attn(node_emb)          # [N, 1]
        scores_exp = torch.exp(scores)

        n_graphs = int(batch.max().item()) + 1
        denom    = torch.zeros(n_graphs, 1, device=node_emb.device)
        denom.scatter_add_(0, batch.unsqueeze(1), scores_exp)

        return scores_exp / (denom[batch] + 1e-8)  # [N, 1]

    def top_service(self, node_emb, batch, node_info_list=None):
        """Return index of highest-scoring node per graph."""
        scores  = self.forward(node_emb, batch).squeeze(1)
        n_graphs = int(batch.max().item()) + 1
        results  = []
        for g in range(n_graphs):
            mask       = (batch == g)
            g_scores   = scores[mask]
            local_idx  = g_scores.argmax().item()
            global_idx = mask.nonzero(as_tuple=True)[0][local_idx].item()
            results.append(global_idx)
        return results
