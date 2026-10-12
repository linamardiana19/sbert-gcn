from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def compute_rouge(reference: str, generated: str) -> dict:
    scores = _SCORER.score(reference, generated)
    return {
        "rouge1": {"precision": scores["rouge1"].precision, "recall": scores["rouge1"].recall, "f1": scores["rouge1"].fmeasure},
        "rouge2": {"precision": scores["rouge2"].precision, "recall": scores["rouge2"].recall, "f1": scores["rouge2"].fmeasure},
        "rougeL": {"precision": scores["rougeL"].precision, "recall": scores["rougeL"].recall, "f1": scores["rougeL"].fmeasure},
    }


def aggregate_rouge(scores_list: list[dict]) -> dict:
    if not scores_list:
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
    n = len(scores_list)
    agg: dict = {}
    for key in ["rouge1", "rouge2", "rougeL"]:
        agg[key] = {
            "precision": sum(s[key]["precision"] for s in scores_list) / n,
            "recall": sum(s[key]["recall"] for s in scores_list) / n,
            "f1": sum(s[key]["f1"] for s in scores_list) / n,
        }
    return agg
