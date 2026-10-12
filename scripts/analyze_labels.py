"""
Analyze positive label distribution from pre-computed labels.

Reads label files produced by run_labeling.py and computes:
1. Total positive vs negative sentences
2. Negative:positive ratio (for pos_weight in BCEWithLogitsLoss)
3. Average positive sentences per document

Usage:
    uv run python scripts/analyze_labels.py --dataset xlsum
    uv run python scripts/analyze_labels.py --dataset xlsum --split train
    uv run python scripts/analyze_labels.py --dataset xlsum --metric rouge1
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def analyze_labels(label_docs: list[dict]) -> dict:
    total_sentences = 0
    total_positive = 0
    ratios = []

    for doc in label_docs:
        labels = doc["labels"]
        n = len(labels)
        n_pos = sum(labels)
        total_sentences += n
        total_positive += n_pos
        if n > 0:
            ratios.append(n_pos / n)

    neg = total_sentences - total_positive
    neg_pos_ratio = neg / total_positive if total_positive > 0 else float("inf")

    return {
        "num_docs": len(label_docs),
        "total_sentences": total_sentences,
        "total_positive": total_positive,
        "total_negative": neg,
        "positive_ratio": total_positive / total_sentences if total_sentences > 0 else 0,
        "negative_positive_ratio": neg_pos_ratio,
        "avg_positive_per_doc": total_positive / len(label_docs) if label_docs else 0,
        "suggested_pos_weight": round(neg_pos_ratio),
        "avg_positive_fraction": float(np.mean(ratios)) if ratios else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze positive label distribution")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--split", choices=["train", "val", "dev", "test"], default=None,
                    help="Specific split to analyze (default: all)")
    ap.add_argument("--metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                    help="ROUGE metric used for labeling (default: rouge1). Must match run_labeling.py --metric.")
    args = ap.parse_args()

    labels_dir = BASE_DIR / "dataset" / args.dataset / "labels" / args.metric

    if args.dataset == "xlsum":
        splits = ["train", "val", "test"]
    else:
        splits = ["train", "dev", "test"]

    if args.split:
        splits = [args.split]

    all_results = {}
    for split in splits:
        path = labels_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"SKIP: {path} not found (run run_labeling.py --metric {args.metric} first)")
            continue

        print(f"\n=== {split} ===")
        label_docs = load_jsonl(path)
        result = analyze_labels(label_docs)
        all_results[split] = result

        print(f"  Number of documents    : {result['num_docs']}")
        print(f"  Total sentences        : {result['total_sentences']}")
        print(f"  Total positive (label=1): {result['total_positive']}")
        print(f"  Positive ratio         : {result['positive_ratio']:.4f}")
        print(f"  Negative:positive ratio: {result['negative_positive_ratio']:.1f}:1")
        print(f"  Avg positive per doc   : {result['avg_positive_per_doc']:.2f}")
        print(f"  Suggested pos_weight   : {result['suggested_pos_weight']}")

    output_dir = BASE_DIR / "dataset" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"label_analysis_{args.metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
