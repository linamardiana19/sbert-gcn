from __future__ import annotations

import torch
import torch.nn as nn

from summarizer.gcn import SentenceGCN, MultiSentenceGCN


class SBERTSummarizer(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3, pos_weight: float = 14.0, num_layers: int = 2, residual: str = "outer", scoring: str = "linear", multi_gcn: bool = False, pos_method: str = "none", pos_dim: int = 0):
        super().__init__()
        self.scoring = scoring
        self.multi_gcn = multi_gcn
        self.pos_method = pos_method
        self.pos_dim = pos_dim
        if pos_method == "learned" and pos_dim > 0:
            self.pos_embedding = nn.Embedding(600, pos_dim)
        if multi_gcn:
            self.gcn = MultiSentenceGCN(in_dim, hidden_dim, dropout, num_layers, residual)
        else:
            self.gcn = SentenceGCN(in_dim, hidden_dim, dropout, num_layers, residual)
        self.dropout = nn.Dropout(dropout)

        if scoring == "linear":
            self.score_head = nn.Linear(hidden_dim, 1)
        elif scoring == "mlp":
            self.score_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif scoring == "context":
            self.score_head = nn.Linear(hidden_dim * 2, 1)
        elif scoring == "multigras":
            self.reading = nn.Linear(hidden_dim, hidden_dim)
            self.score_head = nn.Linear(hidden_dim * 3, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, x, edge_index, edge_weight=None, edge_index_tfidf=None, edge_weight_tfidf=None, pos_idx=None):
        if self.pos_method == "learned" and pos_idx is not None:
            pe = self.pos_embedding(pos_idx)
            x = torch.cat([x, pe], dim=-1)
        if self.multi_gcn:
            h = self.gcn(x, edge_index, edge_weight, edge_index_tfidf, edge_weight_tfidf)
        else:
            h = self.gcn(x, edge_index, edge_weight)
        h = self.dropout(h)

        if self.scoring == "context":
            d_doc = h.mean(dim=0, keepdim=True).expand_as(h)
            logits = self.score_head(torch.cat([h, d_doc], dim=-1)).squeeze(-1)
        elif self.scoring == "multigras":
            d_doc = h.mean(dim=0, keepdim=True).expand_as(h)
            h_pre = self.gcn.skip_proj(x) if hasattr(self.gcn, 'skip_proj') else x
            o = torch.tanh(self.reading(h))
            logits = self.score_head(torch.cat([o, d_doc, h_pre], dim=-1)).squeeze(-1)
        else:
            logits = self.score_head(h).squeeze(-1)

        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

    def predict(self, x, edge_index, edge_weight=None, edge_index_tfidf=None, edge_weight_tfidf=None, pos_idx=None):
        logits = self.forward(x, edge_index, edge_weight, edge_index_tfidf, edge_weight_tfidf, pos_idx=pos_idx)
        return torch.sigmoid(logits)
