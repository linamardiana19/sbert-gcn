from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SelectionResult:
    selected_indices: list[int]
    summary: str


def _get_trigrams(sentence: str) -> set[tuple[str, ...]]:
    words = sentence.lower().split()
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def select_sentences(
    sentences: list[str],
    scores: torch.Tensor,
    k: int = 3,
) -> SelectionResult:
    ranked = scores.argsort(descending=True).tolist()

    selected: list[int] = []
    used_trigrams: set[tuple[str, ...]] = set()

    for idx in ranked:
        if len(selected) >= k:
            break

        trigrams = _get_trigrams(sentences[idx])
        if trigrams & used_trigrams:
            continue

        selected.append(idx)
        used_trigrams.update(trigrams)

    selected.sort()
    summary = " ".join(sentences[i] for i in selected)

    return SelectionResult(selected_indices=selected, summary=summary)


def select_sentences_mmr(
    sentences: list[str],
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    k: int = 3,
    lambda_: float = 0.7,
) -> SelectionResult:
    n = len(sentences)
    if n == 0:
        return SelectionResult(selected_indices=[], summary="")

    relevance = scores.float()
    r_min, r_max = relevance.min(), relevance.max()
    if r_max > r_min:
        relevance = (relevance - r_min) / (r_max - r_min)

    norm = F.normalize(embeddings.float(), p=2, dim=1)
    sim_matrix = norm @ norm.t()

    selected: list[int] = []
    candidates = set(range(n))

    for _ in range(min(k, n)):
        best_idx = -1
        best_mmr = float("-inf")

        for idx in candidates:
            rel = relevance[idx].item()
            if selected:
                max_sim = max(sim_matrix[idx, s].item() for s in selected)
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx == -1:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)

    selected.sort()
    summary = " ".join(sentences[i] for i in selected)
    return SelectionResult(selected_indices=selected, summary=summary)
