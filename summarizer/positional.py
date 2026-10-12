from __future__ import annotations

import math

import torch


def make_simple_features(num_sentences: int) -> torch.Tensor:
    if num_sentences <= 1:
        return torch.zeros(num_sentences, 3)
    positions = torch.arange(num_sentences, dtype=torch.float32)
    n = num_sentences - 1
    normalized = positions / n
    inverse = 1.0 - normalized
    log_decay = 1.0 / torch.log(positions + 2)
    return torch.stack([normalized, inverse, log_decay], dim=-1)


def make_position_encoding(num_sentences: int, pos_dim: int) -> torch.Tensor:
    if pos_dim == 0:
        return torch.zeros(num_sentences, 0)

    pe = torch.zeros(num_sentences, pos_dim)
    positions = torch.arange(num_sentences, dtype=torch.float32).unsqueeze(1)

    if pos_dim == 1:
        if num_sentences <= 1:
            return pe
        pe[:, 0] = positions.squeeze(1) / (num_sentences - 1)
        return pe

    div_term = torch.exp(torch.arange(0, pos_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / pos_dim))
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


def get_pos_dim(pos_method: str, pos_dim: int) -> int:
    if pos_method == "none":
        return 0
    if pos_method == "simple":
        return 3
    return pos_dim
