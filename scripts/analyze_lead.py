"""
Compute Lead-N baseline ROUGE scores.

Lead-N selects the first N sentences of each document as the summary.
This provides a trivial baseline for extractive summarization — especially
strong on news articles due to inverted pyramid structure.

Usage:
    uv run python scripts/analyze_lead.py --dataset xlsum
    uv run python scripts/analyze_lead.py --dataset xlsum --split test
    uv run python scripts/analyze_lead.py --dataset xlsum --max_n 5
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.rouge import compute_rouge, aggregate_rouge


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def compute_lead_n(docs: list[dict], max_n: int = 3) -> dict:
    results = {}
    for n in range(1, max_n + 1):
        all_rouge = []

        for doc in tqdm(docs, desc=f"  Lead-{n}"):
            sents = doc["sentences"]
            summary = doc["summary"]

            lead_summary = " ".join(sents[:n])
            all_rouge.append(compute_rouge(summary, lead_summary))

        agg = aggregate_rouge(all_rouge)
        results[f"lead_{n}"] = {
            "num_docs": len(all_rouge),
            "rouge1": agg["rouge1"]["f1"],
            "rouge2": agg["rouge2"]["f1"],
            "rougeL": agg["rougeL"]["f1"],
        }

    return results


def main():
    ap = argparse.ArgumentParser(description="Compute Lead-N baseline ROUGE scores")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--split", choices=["train", "val", "dev", "test"], default=None,
                    help="Specific split to analyze (default: all)")
    ap.add_argument("--max_n", type=int, default=3,
                    help="Maximum N for Lead-N (default: 3, computes Lead-1 through Lead-3)")
    args = ap.parse_args()

    data_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"

    if args.dataset == "xlsum":
        splits = ["train", "val", "test"]
    else:
        splits = ["train", "dev", "test"]

    if args.split:
        splits = [args.split]

    all_results = {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue

        print(f"\n=== {split} ===")
        docs = load_jsonl(path)
        results = compute_lead_n(docs, args.max_n)
        all_results[split] = results

        print(f"\n  {'Method':<10} {'R1':>8} {'R2':>8} {'RL':>8}")
        print(f"  {'-'*34}")
        for key, scores in results.items():
            name = key.replace("_", "-").capitalize()
            print(f"  {name:<10} {scores['rouge1']:>8.4f} {scores['rouge2']:>8.4f} {scores['rougeL']:>8.4f}")

    output_dir = BASE_DIR / "dataset" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "lead_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
