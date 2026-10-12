"""
Detailed oracle ROUGE analysis per document.

This script reads:
1. Preprocessed documents:
   dataset/<dataset>/preprocessed/<split>.jsonl

2. Oracle labels:
   dataset/<dataset>/labels/<metric>/<split>.jsonl

It reconstructs the oracle extractive summary from sentences with label=1.
If --max_words is provided, the reconstructed summary is truncated to the
specified maximum number of words before ROUGE is computed.

Usage examples:

1. Original scenario, without word limit:
   uv run python scripts/analyze_oracle_detail.py --dataset liputan6 --split test --metric rouge1

2. Maximum 31 words:
   uv run python scripts/analyze_oracle_detail.py --dataset liputan6 --split test --metric rouge1 --max_words 31

3. Maximum 31 words and include full document:
   uv run python scripts/analyze_oracle_detail.py --dataset liputan6 --split test --metric rouge1 --max_words 31 --include_document
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.rouge import compute_rouge


def load_jsonl(path: Path) -> list[dict]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    docs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                docs.append(json.loads(line))

    return docs


def truncate_summary_by_words(
    text: str,
    max_words: Optional[int],
) -> tuple[str, int, int, bool]:
    """
    Truncate a summary based on whitespace-separated words.

    Parameters
    ----------
    text:
        Full summary text.

    max_words:
        Maximum number of words.

        If None, no truncation is applied.

    Returns
    -------
    truncated_text:
        Summary after applying the word limit.

    original_word_count:
        Number of words before truncation.

    final_word_count:
        Number of words after truncation.

    was_truncated:
        True if the summary exceeded max_words.
    """
    text = str(text)
    words = text.split()

    original_word_count = len(words)

    if max_words is None:
        return (
            text,
            original_word_count,
            original_word_count,
            False,
        )

    truncated_words = words[:max_words]
    truncated_text = " ".join(truncated_words)

    final_word_count = len(truncated_words)
    was_truncated = original_word_count > max_words

    return (
        truncated_text,
        original_word_count,
        final_word_count,
        was_truncated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute detailed oracle ROUGE per document, "
            "with an optional maximum word limit."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["xlsum", "liputan6"],
        required=True,
        help="Dataset name.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "dev", "test"],
        required=True,
        help="Dataset split.",
    )

    parser.add_argument(
        "--metric",
        default="rouge1",
        choices=["rouge1", "rouge2", "rougeL"],
        help=(
            "Metric used when creating labels. "
            "Must match run_labeling.py --metric."
        ),
    )

    parser.add_argument(
        "--max_words",
        type=int,
        default=None,
        help=(
            "Maximum number of words in the reconstructed oracle summary. "
            "For example: --max_words 31. "
            "If omitted, the full oracle summary is evaluated."
        ),
    )

    parser.add_argument(
        "--include_document",
        action="store_true",
        help=(
            "Include all document sentences, labels, "
            "and the full article text in the output."
        ),
    )

    args = parser.parse_args()

    if args.max_words is not None and args.max_words <= 0:
        parser.error("--max_words must be greater than 0.")

    preprocessed_path = (
        BASE_DIR
        / "dataset"
        / args.dataset
        / "preprocessed"
        / f"{args.split}.jsonl"
    )

    label_path = (
        BASE_DIR
        / "dataset"
        / args.dataset
        / "labels"
        / args.metric
        / f"{args.split}.jsonl"
    )

    output_dir = (
        BASE_DIR
        / "dataset"
        / args.dataset
        / "oracle_details"
        / args.metric
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.max_words is None:
        output_filename = f"{args.split}_details.jsonl"
    else:
        output_filename = (
            f"{args.split}_details_max_{args.max_words}_words.jsonl"
        )

    output_path = output_dir / output_filename

    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}"
        )

    preprocessed_docs = load_jsonl(preprocessed_path)
    label_docs = load_jsonl(label_path)

    label_map = {
        str(doc["id"]): doc["labels"]
        for doc in label_docs
    }

    total_docs = 0

    skipped_missing_label = 0
    skipped_length_mismatch = 0

    truncated_docs = 0
    empty_oracle_summaries = 0

    total_words_before = 0
    total_words_after = 0

    rouge1_total = 0.0
    rouge2_total = 0.0
    rougeL_total = 0.0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f_out:

        for doc in tqdm(
            preprocessed_docs,
            desc=f"Analyzing {args.split}",
        ):
            doc_id = str(doc["id"])

            if doc_id not in label_map:
                skipped_missing_label += 1
                continue

            sentences = doc["sentences"]
            reference_summary = doc["summary"]
            labels = label_map[doc_id]

            if len(sentences) != len(labels):
                skipped_length_mismatch += 1
                continue

            selected_indices = [
                i
                for i, label in enumerate(labels)
                if int(label) == 1
            ]

            selected_sentences = [
                sentences[i]
                for i in selected_indices
            ]

            # Full oracle summary based on all sentences with label = 1.
            oracle_summary_full = " ".join(selected_sentences)

            # Apply word limit only after the selected sentences are combined.
            (
                oracle_summary,
                word_count_before,
                word_count_after,
                was_truncated,
            ) = truncate_summary_by_words(
                text=oracle_summary_full,
                max_words=args.max_words,
            )

            if was_truncated:
                truncated_docs += 1

            if word_count_after == 0:
                empty_oracle_summaries += 1

            # ROUGE is computed using oracle_summary.
            # If max_words is set, this is the truncated summary.
            # If max_words is None, this is the full summary.
            rouge = compute_rouge(
                reference_summary,
                oracle_summary,
            )

            row = {
                "id": doc_id,
                "num_sentences": len(sentences),
                "num_selected": len(selected_indices),
                "selected_indices": selected_indices,
                "selected_sentences": selected_sentences,

                # Summary before word truncation.
                "oracle_summary_full": oracle_summary_full,

                # Summary actually used for ROUGE evaluation.
                "oracle_summary": oracle_summary,

                "reference_summary": reference_summary,

                "max_words": args.max_words,
                "word_count_before": word_count_before,
                "word_count_after": word_count_after,
                "was_truncated": was_truncated,

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
                row["labels"] = labels
                row["article_text"] = " ".join(sentences)

            f_out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            rouge1_total += rouge["rouge1"]["f1"]
            rouge2_total += rouge["rouge2"]["f1"]
            rougeL_total += rouge["rougeL"]["f1"]

            total_words_before += word_count_before
            total_words_after += word_count_after

            total_docs += 1

    print("\n=== Detail oracle analysis saved ===")
    print(f"Output file             : {output_path}")
    print(f"Dataset                 : {args.dataset}")
    print(f"Split                   : {args.split}")
    print(f"Label metric            : {args.metric}")
    print(f"Maximum words           : {args.max_words}")
    print(f"Documents analyzed      : {total_docs}")
    print(f"Documents truncated     : {truncated_docs}")
    print(f"Empty oracle summaries  : {empty_oracle_summaries}")
    print(f"Skipped missing label   : {skipped_missing_label}")
    print(f"Skipped length mismatch : {skipped_length_mismatch}")

    if total_docs > 0:
        average_words_before = total_words_before / total_docs
        average_words_after = total_words_after / total_docs

        average_rouge1 = rouge1_total / total_docs
        average_rouge2 = rouge2_total / total_docs
        average_rougeL = rougeL_total / total_docs

        print("\nAverage summary length:")
        print(
            f"  Before truncation : "
            f"{average_words_before:.2f} words"
        )
        print(
            f"  After truncation  : "
            f"{average_words_after:.2f} words"
        )

        print("\nAverage ROUGE:")
        print(f"  ROUGE-1 F1 : {average_rouge1:.4f}")
        print(f"  ROUGE-2 F1 : {average_rouge2:.4f}")
        print(f"  ROUGE-L F1 : {average_rougeL:.4f}")
    else:
        print("\nNo documents were analyzed.")


if __name__ == "__main__":
    main()