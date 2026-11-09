"""Sentence-graph fusion: encoder IndoT5 -> pooling kalimat -> GCN -> gated fusion.

    H        : output encoder T5, [B, L, d]
    S        : vektor kalimat = rata-rata token per kalimat, [N, d]   (N = total kalimat di batch)
    G        : SentenceGCN(S (+ fitur tambahan), graf) -> proyeksi ke d
    G_tok    : G disebar balik ke setiap token kalimatnya, [B, L, d]
    gate     : sigmoid(W [H ; G_tok])
    H'       : H + gate * G_tok

``out_proj`` diinisialisasi NOL sehingga pada langkah pertama H' == H:
model mulai persis seperti IndoT5 baseline, lalu perlahan belajar memakai graf.
GCN dijalankan dalam fp32 (di luar autocast) agar stabil dengan PyG.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from summarizer.gcn import SentenceGCN


class SentenceGraphFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        node_extra_dim: int = 0,
        gcn_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual: str = "outer",
    ):
        super().__init__()
        self.d_model = d_model
        self.node_extra_dim = node_extra_dim
        self.gcn = SentenceGCN(
            in_dim=d_model + node_extra_dim,
            hidden_dim=gcn_hidden,
            dropout=dropout,
            num_layers=num_layers,
            residual=residual,
        )
        self.out_proj = nn.Linear(gcn_hidden, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.last_gate_mean: float = 0.0

    def forward(
        self,
        hidden: torch.Tensor,          # [B, L, d]
        sent_index: torch.Tensor,      # [B, L], -1 = bukan token kalimat
        node_offsets: torch.Tensor,    # [B]
        num_nodes: int,
        edge_index: torch.Tensor,      # [2, E] (indeks global)
        edge_weight: torch.Tensor,     # [E]
        node_extra: torch.Tensor | None = None,  # [N, node_extra_dim]
    ) -> torch.Tensor:
        bsz, seq_len, d = hidden.shape
        tok_mask = sent_index >= 0
        node_ids = (sent_index + node_offsets.unsqueeze(1))[tok_mask]  # [T]

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h_tok = hidden[tok_mask].float()                           # [T, d]

            # 1) mean pooling token -> kalimat
            sums = torch.zeros(num_nodes, d, device=hidden.device).index_add_(0, node_ids, h_tok)
            counts = torch.zeros(num_nodes, device=hidden.device).index_add_(
                0, node_ids, torch.ones_like(node_ids, dtype=torch.float)
            )
            sent_vec = sums / counts.clamp(min=1.0).unsqueeze(1)       # [N, d]

            if node_extra is not None:
                sent_vec = torch.cat([sent_vec, node_extra.float()], dim=-1)

            # 2) GCN antar-kalimat (memakai SentenceGCN milik pipeline extractive)
            g = self.gcn(sent_vec, edge_index, edge_weight.float())    # [N, gcn_hidden]
            g = self.out_proj(g)                                       # [N, d]

            # 3) sebar balik ke token
            g_tok = torch.zeros(bsz, seq_len, d, device=hidden.device)
            g_tok[tok_mask] = g[node_ids]

            # 4) gated fusion (residual)
            h32 = hidden.float()
            gate = torch.sigmoid(self.gate(torch.cat([h32, g_tok], dim=-1)))
            fused = h32 + gate * g_tok

        if tok_mask.any():
            self.last_gate_mean = gate[tok_mask].mean().item()
        return fused.to(hidden.dtype)
