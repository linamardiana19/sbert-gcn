"""
Analyze built-in Liputan6 extractive_summary ROUGE.

This script reads raw Liputan6 canonical JSON files:

    dataset/liputan6/canonical/<split>/*.json

It uses:
    - clean_article as source sentences
    - extractive_summary as built-in sentence indices
    - clean_summary as reference summary

Then it reconstructs the built-in extractive summary from clean_article
using extractive_summary indices, and computes ROUGE against clean_summary.

Usage:
    uv run python scripts/analyze_builtin_liputan6_oracle.py --split test

Optional:
    uv run python scripts/analyze_builtin_liputan6_oracle.py --split test --include_document
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.rouge import compute_rouge


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def join_words(words: list[str]) -> str:
    return " ".join(words)


def build_article_sentences(clean_article: list[list[str]]) -> list[str]:
    return [
        join_words(words)
        for words in clean_article
    ]


def build_reference_summary(clean_summary: list[list[str]]) -> str:
    return " ".join(
        join_words(words)
        for words in clean_summary
    )


def select_sentences_by_indices(
    sentences: list[str],
    indices: list[int],
) -> tuple[list[int], list[str], list[int]]:
    valid_indices = []
    selected_sentences = []
    invalid_indices = []

    for idx in indices:
        if 0 <= idx < len(sentences):
            valid_indices.append(idx)
            selected_sentences.append(sentences[idx])
        else:
            invalid_indices.append(idx)

    return valid_indices, selected_sentences, invalid_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ROUGE of built-in Liputan6 extractive_summary against clean_summary."
    )

    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        required=True,
        help="Liputan6 split to analyze.",
    )

    parser.add_argument(
        "--include_document",
        action="store_true",
        help="Include full clean_article sentences in output.",
    )

    args = parser.parse_args()

    input_dir = (
        BASE_DIR
        / "dataset"
        / "liputan6"
        / "canonical"
        / args.split
    )

    output_dir = (
        BASE_DIR
        / "dataset"
        / "liputan6"
        / "builtin_oracle_details"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.split}_details.jsonl"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))

    total_docs = 0
    skipped_missing_field = 0
    skipped_empty_article = 0
    skipped_empty_summary = 0
    skipped_empty_extractive = 0
    total_invalid_indices = 0

    rouge1_total = 0.0
    rouge2_total = 0.0
    rougeL_total = 0.0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for json_file in tqdm(json_files, desc=f"Analyzing Liputan6 {args.split}"):
            try:
                doc = load_json(json_file)

                required_fields = [
                    "id",
                    "clean_article",
                    "clean_summary",
                    "extractive_summary",
                ]

                if any(field not in doc for field in required_fields):
                    skipped_missing_field += 1
                    continue

                sentences = build_article_sentences(doc["clean_article"])
                reference_summary = build_reference_summary(doc["clean_summary"])
                builtin_indices = [int(idx) for idx in doc["extractive_summary"]]

                if not sentences:
                    skipped_empty_article += 1
                    continue

                if not reference_summary.strip():
                    skipped_empty_summary += 1
                    continue

                if not builtin_indices:
                    skipped_empty_extractive += 1
                    continue

                selected_indices, selected_sentences, invalid_indices = (
                    select_sentences_by_indices(
                        sentences=sentences,
                        indices=builtin_indices,
                    )
                )

                total_invalid_indices += len(invalid_indices)

                builtin_extractive_summary = " ".join(selected_sentences)

                rouge = compute_rouge(
                    reference_summary,
                    builtin_extractive_summary,
                )

                row = {
                    "id": str(doc["id"]),
                    "source_file": json_file.name,
                    "num_sentences": len(sentences),
                    "num_selected": len(selected_indices),
                    "builtin_extractive_indices": builtin_indices,
                    "valid_selected_indices": selected_indices,
                    "invalid_indices": invalid_indices,
                    "selected_sentences": selected_sentences,
                    "builtin_extractive_summary": builtin_extractive_summary,
                    "reference_summary": reference_summary,
                    "rouge1_precision": rouge["rouge1"]["precision"],
                    "rouge1_recall": rouge["rouge1"]["recall"],
                    "rouge1_f1": rouge["rouge1"]["f1"],
                    "rouge2_precision": rouge["rouge2"]["precision"],
                    "rouge2_recall": rouge["rouge2"]["recall"],
                    "rouge2_f1": rouge["rouge2"]["f1"],
                    "rougeL_precision": rouge["rougeL"]["precision"],
                    "rougeL_recall": rouge["rougeL"]["recall"],
                    "rougeL_f1": rouge["rougeL"]["f1"],
                }

                if args.include_document:
                    row["sentences"] = sentences
                    row["article_text"] = " ".join(sentences)

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

                rouge1_total += rouge["rouge1"]["f1"]
                rouge2_total += rouge["rouge2"]["f1"]
                rougeL_total += rouge["rougeL"]["f1"]

                total_docs += 1

            except Exception as e:
                skipped_missing_field += 1
                print(f"Failed processing {json_file.name}: {e}")

    print("\n=== Built-in Liputan6 extractive_summary ROUGE analysis saved ===")
    print(f"Output file              : {output_path}")
    print(f"Documents analyzed       : {total_docs}")
    print(f"Skipped missing/error    : {skipped_missing_field}")
    print(f"Skipped empty article    : {skipped_empty_article}")
    print(f"Skipped empty summary    : {skipped_empty_summary}")
    print(f"Skipped empty extractive : {skipped_empty_extractive}")
    print(f"Invalid indices found    : {total_invalid_indices}")

    if total_docs > 0:
        print("\nAverage ROUGE against clean_summary:")
        print(f"  ROUGE-1 F1 : {rouge1_total / total_docs:.4f}")
        print(f"  ROUGE-2 F1 : {rouge2_total / total_docs:.4f}")
        print(f"  ROUGE-L F1 : {rougeL_total / total_docs:.4f}")


if __name__ == "__main__":
    main()