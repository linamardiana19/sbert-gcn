"""
Select best combined oracle labels per document.

This script compares two candidate extractive labels per document:

1. Built-in extractive oracle detail:
   dataset/<dataset>/builtin_oracle_details/<split>_details.jsonl

2. Greedy oracle detail:
   dataset/<dataset>/oracle_details/<metric>/<split>_details.jsonl

For each document, the script selects the label source with the highest
primary metric F1 score, for example ROUGE-1 F1.

Optionally, a secondary tie-break metric can be used. Example:
    primary metric     : rouge1
    tie-break metric   : rouge2

Selection rule:
    1. Choose the source with higher <metric>_f1.
    2. If <metric>_f1 is equal, compare <tie_break_metric>_f1.
    3. If both are equal, use --tie_break, default: greedy.

Main output for modelling:
    dataset/<dataset>/labels/<output_label_name>/<split>.jsonl

Detail output for analysis:
    dataset/<dataset>/best_combine_oracle_details/<output_label_name>/<split>_details.jsonl

Examples:
    uv run python scripts/run_select_best_combine_oracle.py \
        --dataset liputan6 \
        --metric rouge1

    uv run python scripts/run_select_best_combine_oracle.py \
        --dataset liputan6 \
        --metric rouge1 \
        --tie_break_metric rouge2

    uv run python scripts/run_select_best_combine_oracle.py \
        --dataset liputan6 \
        --split test \
        --metric rouge1 \
        --tie_break_metric rouge2

    uv run python scripts/run_select_best_combine_oracle.py \
        --dataset xlsum \
        --metric rouge1 \
        --fallback_to_greedy
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


DATASET_CONFIG = {
    "liputan6": {
        "splits": ["train", "dev", "test"],
    },
    "xlsum": {
        "splits": ["train", "val", "test"],
    },
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_binary_labels(num_sentences: int, selected_indices: list[int]) -> list[int]:
    labels = [0] * num_sentences

    for idx in selected_indices:
        idx = int(idx)
        if 0 <= idx < num_sentences:
            labels[idx] = 1

    return labels


def get_metric_f1(row: dict, metric: str) -> float:
    key = f"{metric}_f1"

    if key not in row:
        raise KeyError(f"Metric field not found: {key}")

    return float(row[key])


def get_metric_value(row: dict, metric: str, value_type: str) -> float:
    key = f"{metric}_{value_type}"
    return float(row.get(key, 0.0))


def get_builtin_indices(row: dict) -> list[int]:
    """
    Supports built-in oracle detail files.

    Expected fields, in priority order:
        1. valid_selected_indices
        2. builtin_extractive_indices
        3. selected_indices
    """

    if "valid_selected_indices" in row:
        return [int(idx) for idx in row["valid_selected_indices"]]

    if "builtin_extractive_indices" in row:
        return [int(idx) for idx in row["builtin_extractive_indices"]]

    if "selected_indices" in row:
        return [int(idx) for idx in row["selected_indices"]]

    raise KeyError(
        "Built-in selected indices not found. "
        "Expected one of: valid_selected_indices, "
        "builtin_extractive_indices, selected_indices."
    )


def get_greedy_indices(row: dict) -> list[int]:
    """
    Supports output from analyze_oracle_detail.py.

    Expected field:
        selected_indices
    """

    if "selected_indices" not in row:
        raise KeyError("Greedy selected_indices not found.")

    return [int(idx) for idx in row["selected_indices"]]


def get_builtin_summary(row: dict) -> str:
    return row.get(
        "builtin_extractive_summary",
        row.get("oracle_summary", ""),
    )


def get_greedy_summary(row: dict) -> str:
    return row.get(
        "oracle_summary",
        row.get("builtin_extractive_summary", ""),
    )


def build_paths(
    dataset: str,
    split: str,
    metric: str,
    output_label_name: str,
) -> dict[str, Path]:
    dataset_dir = BASE_DIR / "dataset" / dataset

    return {
        "builtin_detail": dataset_dir
        / "builtin_oracle_details"
        / f"{split}_details.jsonl",

        "greedy_detail": dataset_dir
        / "oracle_details"
        / metric
        / f"{split}_details.jsonl",

        "greedy_label": dataset_dir
        / "labels"
        / metric
        / f"{split}.jsonl",

        "label_output": dataset_dir
        / "labels"
        / output_label_name
        / f"{split}.jsonl",

        "detail_output": dataset_dir
        / "best_combine_oracle_details"
        / output_label_name
        / f"{split}_details.jsonl",
    }


def copy_greedy_as_best_combine(
    dataset: str,
    split: str,
    metric: str,
    output_label_name: str,
    paths: dict[str, Path],
) -> None:
    """
    Used when a dataset has no built-in extractive oracle details.

    The best-combine label becomes equivalent to the greedy oracle label.
    This is useful for datasets such as XL-Sum.
    """

    greedy_label_path = paths["greedy_label"]
    greedy_detail_path = paths["greedy_detail"]
    label_output_path = paths["label_output"]
    detail_output_path = paths["detail_output"]

    if not greedy_label_path.exists():
        raise FileNotFoundError(f"Greedy label file not found: {greedy_label_path}")

    if not greedy_detail_path.exists():
        raise FileNotFoundError(f"Greedy detail file not found: {greedy_detail_path}")

    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(greedy_label_path, label_output_path)

    greedy_rows = load_jsonl(greedy_detail_path)
    detail_rows = []

    rouge1_total = 0.0
    rouge2_total = 0.0
    rougeL_total = 0.0

    for row in greedy_rows:
        detail_row = {
            "id": str(row["id"]),
            "dataset": dataset,
            "split": split,
            "metric": metric,
            "output_label_name": output_label_name,
            "num_sentences": int(row["num_sentences"]),
            "num_selected": int(row["num_selected"]),

            "selected_source": "greedy_oracle_only",
            "selection_primary_metric": f"{metric}_f1",
            "selection_tie_break_metric": None,
            "final_tie_break": None,
            "selection_reason": "fallback_to_greedy_no_builtin_oracle",

            "selected_indices": get_greedy_indices(row),
            "selected_sentences": row.get("selected_sentences", []),
            "best_summary": get_greedy_summary(row),
            "reference_summary": row.get("reference_summary", ""),

            "best_rouge1_f1": get_metric_value(row, "rouge1", "f1"),
            "best_rouge2_f1": get_metric_value(row, "rouge2", "f1"),
            "best_rougeL_f1": get_metric_value(row, "rougeL", "f1"),

            "builtin_indices": [],
            "builtin_rouge1_f1": None,
            "builtin_rouge2_f1": None,
            "builtin_rougeL_f1": None,

            "greedy_indices": get_greedy_indices(row),
            "greedy_rouge1_f1": get_metric_value(row, "rouge1", "f1"),
            "greedy_rouge2_f1": get_metric_value(row, "rouge2", "f1"),
            "greedy_rougeL_f1": get_metric_value(row, "rougeL", "f1"),
        }

        detail_rows.append(detail_row)

        rouge1_total += detail_row["best_rouge1_f1"]
        rouge2_total += detail_row["best_rouge2_f1"]
        rougeL_total += detail_row["best_rougeL_f1"]

    save_jsonl(detail_rows, detail_output_path)

    total_docs = len(detail_rows)

    print("\n=== Best combine fallback saved ===")
    print(f"Dataset                   : {dataset}")
    print(f"Split                     : {split}")
    print(f"Output label name         : {output_label_name}")
    print(f"Reason                    : Built-in oracle detail not available")
    print(f"Selected source           : greedy_oracle_only")
    print(f"Greedy label input        : {greedy_label_path}")
    print(f"Greedy detail input       : {greedy_detail_path}")
    print(f"Label output file         : {label_output_path}")
    print(f"Detail output file        : {detail_output_path}")
    print(f"Documents processed       : {total_docs}")

    if total_docs > 0:
        print("\nAverage ROUGE:")
        print(f"  ROUGE-1 F1 : {rouge1_total / total_docs:.4f}")
        print(f"  ROUGE-2 F1 : {rouge2_total / total_docs:.4f}")
        print(f"  ROUGE-L F1 : {rougeL_total / total_docs:.4f}")


def decide_source(
    builtin: dict,
    greedy: dict,
    metric: str,
    tie_break_metric: str | None,
    tie_break: str,
) -> tuple[bool, str]:
    """
    Returns:
        choose_builtin: bool
        selection_reason: str
    """

    builtin_primary_f1 = get_metric_f1(builtin, metric)
    greedy_primary_f1 = get_metric_f1(greedy, metric)

    if builtin_primary_f1 > greedy_primary_f1:
        return True, f"builtin_{metric}_f1_higher"

    if builtin_primary_f1 < greedy_primary_f1:
        return False, f"greedy_{metric}_f1_higher"

    if tie_break_metric is not None:
        builtin_secondary_f1 = get_metric_f1(builtin, tie_break_metric)
        greedy_secondary_f1 = get_metric_f1(greedy, tie_break_metric)

        if builtin_secondary_f1 > greedy_secondary_f1:
            return True, (
                f"{metric}_f1_equal_"
                f"builtin_{tie_break_metric}_f1_higher"
            )

        if builtin_secondary_f1 < greedy_secondary_f1:
            return False, (
                f"{metric}_f1_equal_"
                f"greedy_{tie_break_metric}_f1_higher"
            )

        choose_builtin = tie_break == "builtin"
        return (
            choose_builtin,
            f"{metric}_f1_equal_{tie_break_metric}_f1_equal_{tie_break}_tie_break",
        )

    choose_builtin = tie_break == "builtin"
    return choose_builtin, f"{metric}_f1_equal_{tie_break}_tie_break"


def select_best_for_split(
    dataset: str,
    split: str,
    metric: str,
    tie_break_metric: str | None,
    tie_break: str,
    fallback_to_greedy: bool,
    output_label_name: str,
) -> None:
    paths = build_paths(
        dataset=dataset,
        split=split,
        metric=metric,
        output_label_name=output_label_name,
    )

    builtin_path = paths["builtin_detail"]
    greedy_path = paths["greedy_detail"]
    label_output_path = paths["label_output"]
    detail_output_path = paths["detail_output"]

    if not greedy_path.exists():
        raise FileNotFoundError(f"Greedy oracle detail file not found: {greedy_path}")

    if not builtin_path.exists():
        if fallback_to_greedy:
            copy_greedy_as_best_combine(
                dataset=dataset,
                split=split,
                metric=metric,
                output_label_name=output_label_name,
                paths=paths,
            )
            return

        raise FileNotFoundError(
            f"Built-in oracle detail file not found: {builtin_path}\n"
            f"If this dataset has no built-in extractive labels, "
            f"run with --fallback_to_greedy."
        )

    builtin_rows = load_jsonl(builtin_path)
    greedy_rows = load_jsonl(greedy_path)

    builtin_map = {str(row["id"]): row for row in builtin_rows}
    greedy_map = {str(row["id"]): row for row in greedy_rows}

    all_doc_ids = sorted(set(builtin_map.keys()) | set(greedy_map.keys()))

    label_rows = []
    detail_rows = []

    total_docs = 0
    selected_builtin = 0
    selected_greedy = 0
    skipped_missing_builtin = 0
    skipped_missing_greedy = 0
    skipped_sentence_mismatch = 0

    best_rouge1_total = 0.0
    best_rouge2_total = 0.0
    best_rougeL_total = 0.0

    for doc_id in tqdm(
        all_doc_ids,
        desc=f"Selecting best combine labels: {dataset} {split}",
    ):
        if doc_id not in builtin_map:
            skipped_missing_builtin += 1
            continue

        if doc_id not in greedy_map:
            skipped_missing_greedy += 1
            continue

        builtin = builtin_map[doc_id]
        greedy = greedy_map[doc_id]

        builtin_num_sentences = int(builtin["num_sentences"])
        greedy_num_sentences = int(greedy["num_sentences"])

        if builtin_num_sentences != greedy_num_sentences:
            skipped_sentence_mismatch += 1
            continue

        choose_builtin, selection_reason = decide_source(
            builtin=builtin,
            greedy=greedy,
            metric=metric,
            tie_break_metric=tie_break_metric,
            tie_break=tie_break,
        )

        if choose_builtin:
            selected_source = "builtin_extractive_summary"
            selected_indices = get_builtin_indices(builtin)
            selected_sentences = builtin.get("selected_sentences", [])
            best_summary = get_builtin_summary(builtin)

            best_rouge1_f1 = get_metric_value(builtin, "rouge1", "f1")
            best_rouge2_f1 = get_metric_value(builtin, "rouge2", "f1")
            best_rougeL_f1 = get_metric_value(builtin, "rougeL", "f1")

            selected_builtin += 1
        else:
            selected_source = "greedy_oracle"
            selected_indices = get_greedy_indices(greedy)
            selected_sentences = greedy.get("selected_sentences", [])
            best_summary = get_greedy_summary(greedy)

            best_rouge1_f1 = get_metric_value(greedy, "rouge1", "f1")
            best_rouge2_f1 = get_metric_value(greedy, "rouge2", "f1")
            best_rougeL_f1 = get_metric_value(greedy, "rougeL", "f1")

            selected_greedy += 1

        labels = make_binary_labels(
            num_sentences=builtin_num_sentences,
            selected_indices=selected_indices,
        )

        label_row = {
            "id": doc_id,
            "labels": labels,
        }

        detail_row = {
            "id": doc_id,
            "dataset": dataset,
            "split": split,
            "metric": metric,
            "output_label_name": output_label_name,
            "source_file": builtin.get("source_file"),
            "num_sentences": builtin_num_sentences,
            "num_selected": int(sum(labels)),

            "selected_source": selected_source,
            "selection_primary_metric": f"{metric}_f1",
            "selection_tie_break_metric": (
                f"{tie_break_metric}_f1" if tie_break_metric else None
            ),
            "final_tie_break": tie_break,
            "selection_reason": selection_reason,

            "selected_indices": [int(idx) for idx in selected_indices],
            "selected_sentences": selected_sentences,
            "best_summary": best_summary,
            "reference_summary": builtin.get(
                "reference_summary",
                greedy.get("reference_summary", ""),
            ),

            "best_rouge1_f1": best_rouge1_f1,
            "best_rouge2_f1": best_rouge2_f1,
            "best_rougeL_f1": best_rougeL_f1,

            "builtin_indices": get_builtin_indices(builtin),
            "builtin_rouge1_f1": get_metric_value(builtin, "rouge1", "f1"),
            "builtin_rouge2_f1": get_metric_value(builtin, "rouge2", "f1"),
            "builtin_rougeL_f1": get_metric_value(builtin, "rougeL", "f1"),

            "greedy_indices": get_greedy_indices(greedy),
            "greedy_rouge1_f1": get_metric_value(greedy, "rouge1", "f1"),
            "greedy_rouge2_f1": get_metric_value(greedy, "rouge2", "f1"),
            "greedy_rougeL_f1": get_metric_value(greedy, "rougeL", "f1"),
        }

        label_rows.append(label_row)
        detail_rows.append(detail_row)

        best_rouge1_total += best_rouge1_f1
        best_rouge2_total += best_rouge2_f1
        best_rougeL_total += best_rougeL_f1
        total_docs += 1

    save_jsonl(label_rows, label_output_path)
    save_jsonl(detail_rows, detail_output_path)

    print("\n=== Best combine oracle labels saved ===")
    print(f"Dataset                   : {dataset}")
    print(f"Split                     : {split}")
    print(f"Primary selection metric  : {metric}_f1")
    print(
        "Tie-break metric          : "
        f"{tie_break_metric}_f1" if tie_break_metric else "Tie-break metric          : None"
    )
    print(f"Final tie-break           : {tie_break}")
    print(f"Output label name         : {output_label_name}")
    print(f"Built-in detail input     : {builtin_path}")
    print(f"Greedy detail input       : {greedy_path}")
    print(f"Label output file         : {label_output_path}")
    print(f"Detail output file        : {detail_output_path}")
    print(f"Documents processed       : {total_docs}")
    print(f"Selected built-in         : {selected_builtin}")
    print(f"Selected greedy oracle    : {selected_greedy}")
    print(f"Skipped missing built-in  : {skipped_missing_builtin}")
    print(f"Skipped missing greedy    : {skipped_missing_greedy}")
    print(f"Skipped sentence mismatch : {skipped_sentence_mismatch}")

    if total_docs > 0:
        print("\nAverage best-combined ROUGE:")
        print(f"  ROUGE-1 F1 : {best_rouge1_total / total_docs:.4f}")
        print(f"  ROUGE-2 F1 : {best_rouge2_total / total_docs:.4f}")
        print(f"  ROUGE-L F1 : {best_rougeL_total / total_docs:.4f}")

        print("\nSelection ratio:")
        print(f"  Built-in      : {selected_builtin / total_docs:.4f}")
        print(f"  Greedy oracle : {selected_greedy / total_docs:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best combined oracle labels per dataset."
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset name.",
    )

    parser.add_argument(
        "--split",
        default=None,
        help="Specific split to process. If omitted, all dataset splits are processed.",
    )

    parser.add_argument(
        "--metric",
        default="rouge1",
        choices=["rouge1", "rouge2", "rougeL"],
        help="Primary metric used for best-label selection.",
    )

    parser.add_argument(
        "--tie_break_metric",
        choices=["rouge1", "rouge2", "rougeL"],
        default=None,
        help=(
            "Secondary metric used when primary metric F1 is equal. "
            "Example: --metric rouge1 --tie_break_metric rouge2."
        ),
    )

    parser.add_argument(
        "--tie_break",
        choices=["greedy", "builtin"],
        default="greedy",
        help="Source selected when primary and secondary scores are equal.",
    )

    parser.add_argument(
        "--output_label_name",
        default=None,
        help=(
            "Output label folder name under dataset/<dataset>/labels/. "
            "If omitted, default is best_combine_<metric> or "
            "best_combine_<metric>_tie_<tie_break_metric>."
        ),
    )

    parser.add_argument(
        "--fallback_to_greedy",
        action="store_true",
        help=(
            "If built-in oracle details are unavailable, copy greedy labels as "
            "best-combine labels. Useful for datasets such as XL-Sum."
        ),
    )

    args = parser.parse_args()

    if args.output_label_name is None:
        if args.tie_break_metric is None:
            output_label_name = f"best_combine_{args.metric}"
        else:
            output_label_name = f"best_combine_{args.metric}_tie_{args.tie_break_metric}"
    else:
        output_label_name = args.output_label_name

    valid_splits = DATASET_CONFIG[args.dataset]["splits"]
    splits = [args.split] if args.split else valid_splits

    for split in splits:
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split '{split}' for dataset '{args.dataset}'. "
                f"Valid splits: {valid_splits}"
            )

        print(f"\n=== {args.dataset} | {split} | {args.metric} ===")

        select_best_for_split(
            dataset=args.dataset,
            split=split,
            metric=args.metric,
            tie_break_metric=args.tie_break_metric,
            tie_break=args.tie_break,
            fallback_to_greedy=args.fallback_to_greedy,
            output_label_name=output_label_name,
        )


if __name__ == "__main__":
    main()