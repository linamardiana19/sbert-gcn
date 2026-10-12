"""
Compute oracle ROUGE scores (extractive upper bound) from pre-computed labels.

Reads label files produced by run_labeling.py and preprocessed data,
then computes ROUGE of the oracle summary (sentences with label=1)
against the reference summary.

Usage:
    uv run python scripts/analyze_oracle.py --dataset xlsum
    uv run python scripts/analyze_oracle.py --dataset xlsum --split test
    uv run python scripts/analyze_oracle.py --dataset xlsum --metric rouge1
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
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def compute_oracle_rouge(prep_docs: list[dict], label_docs: list[dict]) -> dict:
    label_map = {doc["id"]: doc["labels"] for doc in label_docs}

    all_rouge = []

    for doc in tqdm(prep_docs, desc="  Computing oracle ROUGE"):
        doc_id = doc["id"]
        if doc_id not in label_map:
            continue

        sents = doc["sentences"]
        summary = doc["summary"]
        labels = label_map[doc_id]

        selected = [sents[i] for i, l in enumerate(labels) if l == 1]
        if not selected:
            continue

        oracle_summary = " ".join(selected)
        all_rouge.append(compute_rouge(summary, oracle_summary))

    agg = aggregate_rouge(all_rouge)
    return {
        "num_docs": len(all_rouge),
        "rouge1": agg["rouge1"]["f1"],
        "rouge2": agg["rouge2"]["f1"],
        "rougeL": agg["rougeL"]["f1"],
    }


def main():
    ap = argparse.ArgumentParser(description="Compute oracle ROUGE from pre-computed labels")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--split", choices=["train", "val", "dev", "test"], default=None,
                    help="Specific split to analyze (default: all)")
    ap.add_argument("--metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                    help="ROUGE metric used for labeling (default: rouge1). Must match run_labeling.py --metric.")
    args = ap.parse_args()

    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"
    labels_dir = BASE_DIR / "dataset" / args.dataset / "labels" / args.metric

    if args.dataset == "xlsum":
        splits = ["train", "val", "test"]
    else:
        splits = ["train", "dev", "test"]

    if args.split:
        splits = [args.split]

    metric_label = args.metric.upper().replace("ROUGE", "ROUGE-")

    all_results = {}
    for split in splits:
        prep_path = prep_dir / f"{split}.jsonl"
        label_path = labels_dir / f"{split}.jsonl"

        if not prep_path.exists():
            print(f"SKIP: {prep_path} not found (run run_preprocessing.py first)")
            continue
        if not label_path.exists():
            print(f"SKIP: {label_path} not found (run run_labeling.py --metric {args.metric} first)")
            continue

        print(f"\n=== {split} ===")
        prep_docs = load_jsonl(prep_path)
        label_docs = load_jsonl(label_path)

        result = compute_oracle_rouge(prep_docs, label_docs)
        all_results[split] = result

        print(f"  Documents: {result['num_docs']}")
        print(f"  Oracle ROUGE (greedy {metric_label} labels):")
        print(f"    ROUGE-1: {result['rouge1']:.4f}")
        print(f"    ROUGE-2: {result['rouge2']:.4f}")
        print(f"    ROUGE-L: {result['rougeL']:.4f}")

    output_dir = BASE_DIR / "dataset" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"oracle_scores_{args.metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
