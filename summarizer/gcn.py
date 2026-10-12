from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SentenceGCN(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3, num_layers: int = 2, residual: str = "outer"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        if num_layers == 0:
            self.proj = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()
        else:
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            if num_layers >= 2:
                self.gcn2 = GCNConv(hidden_dim, hidden_dim)
                self.dropout = nn.Dropout(dropout)
            if residual != "none":
                self.skip_proj = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        if self.num_layers == 0:
            return self.proj(x)
        h = self.gcn1(x, edge_index, edge_weight)
        h = F.relu(h)
        if self.num_layers >= 2:
            h = self.dropout(h)
            h = self.gcn2(h, edge_index, edge_weight)
        if self.residual != "none":
            h = h + self.skip_proj(x)
        return h


class MultiSentenceGCN(nn.Module):
    """Dual GCN following Multi-GraS (Jing et al., 2021).

    Each relation gets its own Skip-GCN (with inner skip, Eq 1-2).
    Outputs are concatenated and projected (Eq 3), then outer skip (Eq 4).
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3, num_layers: int = 1, residual: str = "outer"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.gcn_sem = SentenceGCN(in_dim, hidden_dim, dropout, num_layers, residual="none")
        self.gcn_tfidf = SentenceGCN(in_dim, hidden_dim, dropout, num_layers, residual="none")
        # Eq 3: H = tanh(cat(H_sem, H_tfidf) W + b)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        # Eq 4: outer skip
        if residual != "none":
            self.skip_proj = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index_sem, edge_weight_sem, edge_index_tfidf, edge_weight_tfidf):
        h_sem = self.gcn_sem(x, edge_index_sem, edge_weight_sem)
        h_tfidf = self.gcn_tfidf(x, edge_index_tfidf, edge_weight_tfidf)
        h = torch.tanh(self.combine(torch.cat([h_sem, h_tfidf], dim=-1)))
        if self.residual != "none":
            h = h + self.skip_proj(x)
        return h
