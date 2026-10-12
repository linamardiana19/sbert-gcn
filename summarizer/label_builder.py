from __future__ import annotations

from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def greedy_rouge_label(
    sentences: list[str],
    reference: str,
    metric: str = "rouge2",
) -> list[bool]:
    n = len(sentences)
    if n == 0:
        return []

    selected: list[int] = []
    remaining = set(range(n))
    current_score = 0.0

    while remaining:
        best_idx = -1
        best_score = current_score

        for idx in remaining:
            candidate = selected + [idx]
            candidate_text = " ".join(sentences[i] for i in candidate)
            scores = _SCORER.score(reference, candidate_text)
            f1 = scores[metric].fmeasure
            if f1 > best_score:
                best_score = f1
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_score = best_score

    labels = [i in selected for i in range(n)]
    return labels
