import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from summarizer.label_builder import greedy_rouge_label

DATASET_CONFIG = {
    "xlsum": {"splits": ["train", "val", "test"]},
    "liputan6": {"splits": ["train", "dev", "test"]},
}

CHECKPOINT_INTERVAL = 100


def label_split(
    input_path: Path,
    output_path: Path,
    metric: str = "rouge2",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".ckpt")

    done_ids: set[str] = set()
    results: list[dict] = []
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(row)
                done_ids.add(row["id"])
        print(f"  Resumed from checkpoint: {len(results)} documents")

    with open(input_path, encoding="utf-8") as f:
        docs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    remaining = [d for d in docs if d["id"] not in done_ids]

    if not remaining:
        print(f"  All {len(docs)} documents already labeled")
        if checkpoint_path.exists():
            _save_jsonl(results, output_path)
            checkpoint_path.unlink()
            print(f"  Finalized: {output_path}")
        return

    print(f"  Labeling {len(remaining)} documents ({len(done_ids)} already done)")

    for i, doc in enumerate(tqdm(remaining, desc="  Labeling")):
        labels = greedy_rouge_label(doc["sentences"], doc["summary"], metric)
        results.append({
            "id": doc["id"],
            "labels": [int(l) for l in labels],
        })

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_jsonl(results, checkpoint_path)

    _save_jsonl(results, output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"  Saved {len(results)} documents to {output_path}")


def _save_jsonl(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create greedy ROUGE labels for extractive summarization.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--split", default=None, help="Specific split to process. If omitted, processes all.")
    ap.add_argument("--metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"])
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    splits = [args.split] if args.split else cfg["splits"]
    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"
    out_dir = BASE_DIR / "dataset" / args.dataset / "labels" / args.metric

    for split in splits:
        input_path = prep_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue

        print(f"\n=== {split} ===")
        label_split(input_path, out_dir / f"{split}.jsonl", args.metric)


if __name__ == "__main__":
    main()
