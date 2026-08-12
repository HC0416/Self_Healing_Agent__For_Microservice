"""
model/baselines.py
==================
Baseline models for comparison with our GNN approach.

M1: Isolation Forest  — classical ML, no deep learning
M2: LSTM Autoencoder  — deep learning, sequence-based (ignores graph)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool


# ══════════════════════════════════════════════════════════════
# M1 — Isolation Forest
# ══════════════════════════════════════════════════════════════

class IsolationForestBaseline:
    """
    M1: Classical ML baseline.
    Treats each trace as a flat feature vector (no deep learning,
    no graph structure). Fits on normal traces only.

    Input:  PyG DataLoader
    Output: anomaly scores (higher = more anomalous)
    """

    def __init__(self, contamination=0.176, random_state=42):
        from sklearn.ensemble import IsolationForest
        self.clf = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.fitted = False

    def _extract_features(self, loader):
        """
        Convert each graph to a fixed-size feature vector by
        computing statistics over all node features:
        mean, max, min, std per feature dimension → 38×4 = 152 dims
        """
        all_feats, all_labels = [], []
        for batch in loader:
            # Process each graph in the batch separately
            ptr = batch.ptr  # graph boundaries
            for i in range(batch.num_graphs):
                node_feats = batch.x[ptr[i]:ptr[i+1]]  # [N_i, 38]
                # Aggregate: mean + max + min + std
                feat = torch.cat([
                    node_feats.mean(dim=0),
                    node_feats.max(dim=0).values,
                    node_feats.min(dim=0).values,
                    node_feats.std(dim=0).nan_to_num(0),
                ], dim=0)  # [152]
                all_feats.append(feat.numpy())
                all_labels.append(batch.y[i].item())
        return np.array(all_feats), np.array(all_labels)

    def fit(self, train_loader):
        """Fit on normal training traces."""
        print("  Extracting features for Isolation Forest...")
        feats, labels = self._extract_features(train_loader)
        normal_feats = feats[labels == 0]
        print(f"  Fitting on {len(normal_feats)} normal traces...")
        self.clf.fit(normal_feats)
        self.fitted = True

    def score(self, loader):
        """
        Returns (scores, labels).
        Scores: higher = more anomalous (negated IF score).
        """
        feats, labels = self._extract_features(loader)
        # IF returns negative scores: more negative = more anomalous
        # Negate so higher = more anomalous
        scores = -self.clf.score_samples(feats)
        return scores, labels


# ══════════════════════════════════════════════════════════════
# M2 — LSTM Autoencoder
# ══════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    """LSTM encoder: sequence of spans → bottleneck vector."""

    def __init__(self, input_dim=38, hidden_dim=64, latent_dim=32,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: [B, seq_len, 38]
        out, (h, _) = self.lstm(x)
        # Use last hidden state
        z = self.fc(h[-1])  # [B, latent_dim]
        return z, out


class LSTMDecoder(nn.Module):
    """LSTM decoder: bottleneck vector → reconstructed sequence."""

    def __init__(self, latent_dim=32, hidden_dim=64,
                 output_dim=38, num_layers=2, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        # z: [B, latent_dim]
        h = self.fc(z)  # [B, hidden_dim]
        # Repeat z for each timestep
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, hidden]
        out, _ = self.lstm(h_seq)
        recon = self.out(out)  # [B, T, 38]
        return recon


class LSTMAutoencoder(nn.Module):
    """
    M2: LSTM Autoencoder baseline.

    Treats each trace as a SEQUENCE of spans (ignores graph edges).
    Anomaly score = reconstruction error on test traces.
    Trained on normal traces only — high error = anomaly.

    Key difference from GNN:
      LSTM: span1 → span2 → span3  (sequence, order matters)
      GNN:  spans connected by edges (graph structure matters)
    """

    def __init__(self, input_dim=38, hidden_dim=64,
                 latent_dim=32, num_layers=2, dropout=0.2,
                 max_seq_len=50):
        super().__init__()
        self.encoder  = LSTMEncoder(input_dim, hidden_dim,
                                    latent_dim, num_layers, dropout)
        self.decoder  = LSTMDecoder(latent_dim, hidden_dim,
                                    input_dim, num_layers, dropout)
        self.max_seq_len = max_seq_len

    def _graphs_to_sequences(self, batch):
        """
        Convert a PyG batch to padded sequences.
        Each graph → padded tensor of shape [max_seq_len, 38].
        """
        ptr    = batch.ptr
        seqs   = []
        labels = []
        for i in range(batch.num_graphs):
            nodes = batch.x[ptr[i]:ptr[i+1]]  # [N_i, 38]
            N = nodes.shape[0]
            # Pad or truncate to max_seq_len
            if N >= self.max_seq_len:
                seq = nodes[:self.max_seq_len]
            else:
                pad = torch.zeros(
                    self.max_seq_len - N, nodes.shape[1],
                    device=nodes.device)
                seq = torch.cat([nodes, pad], dim=0)
            seqs.append(seq)
            labels.append(batch.y[i].item())
        seqs   = torch.stack(seqs, dim=0)   # [B, max_seq_len, 38]
        labels = torch.tensor(labels, dtype=torch.long)
        return seqs, labels

    def forward(self, batch):
        seqs, labels = self._graphs_to_sequences(batch)
        z, _         = self.encoder(seqs)
        recon        = self.decoder(z, self.max_seq_len)
        # Reconstruction error per graph
        recon_error  = ((seqs - recon) ** 2).mean(dim=[1, 2])
        return recon_error, z, labels

    def get_scores(self, loader, device):
        """Run on loader, return (scores, labels)."""
        self.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                scores, _, labels = self.forward(batch)
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.tolist())
        return np.array(all_scores), np.array(all_labels)
