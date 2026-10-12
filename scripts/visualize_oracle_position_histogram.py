"""
Visualize oracle sentence position histogram for XLSum and Liputan6.

This script reads:
1. Preprocessed documents:
   dataset/<dataset>/preprocessed/<split>.jsonl

2. Oracle labels from run_labeling.py:
   dataset/<dataset>/labels/<metric>/<split>.jsonl

It collects sentence indices where label = 1, converts them into sentence
positions, and visualizes their distribution as a histogram.

Modes:
    relative:
        Position is normalized from 0.0 to 1.0.
        0.0 = first sentence, 1.0 = last sentence.

    absolute:
        Position uses human-readable numbering.
        1 = first sentence, 2 = second sentence, etc.
        By default, only sentence positions 1-30 are shown.

Default:
    dataset = both
    split   = test
    metric  = rouge1
    mode    = relative

Usage examples:
    uv run python scripts/visualize_oracle_position_histogram.py

    uv run python scripts/visualize_oracle_position_histogram.py --dataset xlsum

    uv run python scripts/visualize_oracle_position_histogram.py --dataset liputan6

    uv run python scripts/visualize_oracle_position_histogram.py --dataset both

    uv run python scripts/visualize_oracle_position_histogram.py --dataset xlsum --split test --metric rouge1 --mode relative

    uv run python scripts/visualize_oracle_position_histogram.py --dataset liputan6 --split test --metric rouge1 --mode absolute --max_absolute_position 30

    uv run python scripts/visualize_oracle_position_histogram.py --dataset both --split train --metric rouge1 --mode absolute --max_absolute_position 30
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib

# Use non-GUI backend so this works from terminal and saves PNG files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


DATASET_SPLITS = {
    "xlsum": ["train", "val", "test"],
    "liputan6": ["train", "dev", "test"],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def resolve_splits(dataset: str, split: str) -> list[str]:
    """
    Resolve split name for each dataset.

    XLSum uses:
        train, val, test

    Liputan6 uses:
        train, dev, test
    """

    if split == "all":
        return DATASET_SPLITS[dataset]

    if split == "validation":
        if dataset == "xlsum":
            return ["val"]
        if dataset == "liputan6":
            return ["dev"]

    return [split]


def binary_labels_to_indices(labels: list[int]) -> list[int]:
    return [
        i
        for i, label in enumerate(labels)
        if int(label) == 1
    ]


def calculate_position(index: int, num_sentences: int, mode: str) -> float:
    """
    mode = relative:
        position is normalized into range 0.0 - 1.0

        first sentence  = 0.0
        last sentence   = 1.0

    mode = absolute:
        position uses human-readable numbering:
        sentence 1, sentence 2, sentence 3, ...
    """

    if mode == "absolute":
        return float(index + 1)

    if num_sentences <= 1:
        return 0.0

    return index / (num_sentences - 1)


def collect_positions_for_dataset(
    dataset: str,
    split: str,
    metric: str,
    mode: str,
    max_absolute_position: int,
) -> dict:
    prep_dir = BASE_DIR / "dataset" / dataset / "preprocessed"
    label_dir = BASE_DIR / "dataset" / dataset / "labels" / metric

    splits = resolve_splits(dataset, split)

    all_positions = []
    total_docs = 0
    total_docs_with_label = 0
    total_selected_sentences = 0
    total_selected_sentences_before_filter = 0

    skipped_missing_file = 0
    skipped_missing_label = 0
    skipped_length_mismatch = 0
    skipped_empty_document = 0
    skipped_outside_absolute_limit = 0

    for current_split in splits:
        prep_path = prep_dir / f"{current_split}.jsonl"
        label_path = label_dir / f"{current_split}.jsonl"

        if not prep_path.exists():
            print(f"SKIP {dataset}-{current_split}: preprocessed file not found: {prep_path}")
            skipped_missing_file += 1
            continue

        if not label_path.exists():
            print(f"SKIP {dataset}-{current_split}: label file not found: {label_path}")
            skipped_missing_file += 1
            continue

        prep_docs = load_jsonl(prep_path)
        label_docs = load_jsonl(label_path)

        label_map = {
            str(doc["id"]): doc["labels"]
            for doc in label_docs
        }

        total_docs += len(prep_docs)

        for doc in prep_docs:
            doc_id = str(doc["id"])

            if doc_id not in label_map:
                skipped_missing_label += 1
                continue

            sentences = doc["sentences"]
            labels = label_map[doc_id]

            if not sentences:
                skipped_empty_document += 1
                continue

            if len(sentences) != len(labels):
                skipped_length_mismatch += 1
                continue

            selected_indices = binary_labels_to_indices(labels)
            total_selected_sentences_before_filter += len(selected_indices)

            for idx in selected_indices:
                position = calculate_position(
                    index=idx,
                    num_sentences=len(sentences),
                    mode=mode,
                )

                if mode == "absolute" and position > max_absolute_position:
                    skipped_outside_absolute_limit += 1
                    continue

                all_positions.append(position)

            total_docs_with_label += 1
            total_selected_sentences = len(all_positions)

    return {
        "dataset": dataset,
        "positions": all_positions,
        "total_docs": total_docs,
        "total_docs_with_label": total_docs_with_label,
        "total_selected_sentences": total_selected_sentences,
        "total_selected_sentences_before_filter": total_selected_sentences_before_filter,
        "skipped_missing_file": skipped_missing_file,
        "skipped_missing_label": skipped_missing_label,
        "skipped_length_mismatch": skipped_length_mismatch,
        "skipped_empty_document": skipped_empty_document,
        "skipped_outside_absolute_limit": skipped_outside_absolute_limit,
    }


def summarize_positions(positions: list[float], mode: str) -> dict:
    if not positions:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "early_30_percent": None,
            "middle_40_percent": None,
            "late_30_percent": None,
        }

    summary = {
        "count": len(positions),
        "mean": mean(positions),
        "median": median(positions),
    }

    if mode == "relative":
        early = [p for p in positions if p <= 0.30]
        middle = [p for p in positions if 0.30 < p <= 0.70]
        late = [p for p in positions if p > 0.70]

        n = len(positions)

        summary["early_30_percent"] = len(early) / n
        summary["middle_40_percent"] = len(middle) / n
        summary["late_30_percent"] = len(late) / n
    else:
        summary["early_30_percent"] = None
        summary["middle_40_percent"] = None
        summary["late_30_percent"] = None

    return summary


def plot_histogram(
    results: list[dict],
    metric: str,
    split: str,
    mode: str,
    bins: int,
    normalize: bool,
    output_path: Path,
    max_absolute_position: int,
) -> None:
    num_plots = len(results)

    fig, axes = plt.subplots(
        1,
        num_plots,
        figsize=(7 * num_plots, 5),
        squeeze=False,
    )

    axes = axes[0]

    for ax, result in zip(axes, results):
        dataset = result["dataset"]
        positions = result["positions"]

        if not positions:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
            )
            ax.set_title(dataset.upper())
            continue

        if normalize:
            weights = [1 / len(positions)] * len(positions)
            y_label = "Proportion"
        else:
            weights = None
            y_label = "Count"

        if mode == "relative":
            ax.hist(
                positions,
                bins=bins,
                range=(0, 1),
                weights=weights,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Relative sentence position")
        else:
            # Absolute mode uses sentence numbering 1, 2, 3, ...
            # and only shows positions up to max_absolute_position.
            ax.hist(
                positions,
                bins=max_absolute_position,
                range=(1, max_absolute_position + 1),
                weights=weights,
            )
            ax.set_xlim(1, max_absolute_position)
            ax.set_xlabel("Sentence position")

            # Make x-axis easier to read.
            if max_absolute_position <= 30:
                ax.set_xticks(range(1, max_absolute_position + 1, 1))
                ax.tick_params(axis="x", labelrotation=90)
            else:
                step = max(1, max_absolute_position // 15)
                ax.set_xticks(range(1, max_absolute_position + 1, step))

        summary = summarize_positions(positions, mode)

        if mode == "absolute":
            title = (
                f"{dataset.upper()}\n"
                f"Docs={result['total_docs_with_label']}, "
                f"Selected shown={result['total_selected_sentences']}"
            )
        else:
            title = (
                f"{dataset.upper()}\n"
                f"Docs={result['total_docs_with_label']}, "
                f"Selected={result['total_selected_sentences']}"
            )

        ax.set_title(title)
        ax.set_ylabel(y_label)

        if summary["count"] > 0:
            if mode == "relative":
                stats_text = (
                    f"Mean={summary['mean']:.3f}\n"
                    f"Median={summary['median']:.3f}\n"
                    f"Early <=30%={summary['early_30_percent']:.2%}"
                )
            else:
                stats_text = (
                    f"Mean={summary['mean']:.2f}\n"
                    f"Median={summary['median']:.2f}\n"
                    f"Shown <= {max_absolute_position}"
                )

            ax.text(
                0.98,
                0.95,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "alpha": 0.15},
            )

    if mode == "absolute":
        title = (
            f"Oracle sentence position histogram | metric={metric}, "
            f"split={split}, mode={mode}, shown=1-{max_absolute_position}"
        )
    else:
        title = (
            f"Oracle sentence position histogram | metric={metric}, "
            f"split={split}, mode={mode}"
        )

    fig.suptitle(title, fontsize=13)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_json(
    results: list[dict],
    metric: str,
    split: str,
    mode: str,
    output_path: Path,
    max_absolute_position: int,
) -> None:
    rows = {}

    for result in results:
        dataset = result["dataset"]
        positions = result["positions"]

        rows[dataset] = {
            "metric": metric,
            "split": split,
            "mode": mode,
            "max_absolute_position": max_absolute_position if mode == "absolute" else None,
            "total_docs": result["total_docs"],
            "total_docs_with_label": result["total_docs_with_label"],
            "total_selected_sentences_shown": result["total_selected_sentences"],
            "total_selected_sentences_before_filter": result[
                "total_selected_sentences_before_filter"
            ],
            "skipped_missing_file": result["skipped_missing_file"],
            "skipped_missing_label": result["skipped_missing_label"],
            "skipped_length_mismatch": result["skipped_length_mismatch"],
            "skipped_empty_document": result["skipped_empty_document"],
            "skipped_outside_absolute_limit": result["skipped_outside_absolute_limit"],
            "position_summary": summarize_positions(positions, mode),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize oracle sentence position histogram for XLSum and Liputan6."
    )

    parser.add_argument(
        "--dataset",
        default="both",
        choices=["both", "xlsum", "liputan6"],
        help="Dataset to visualize. Default: both.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["all", "train", "val", "dev", "validation", "test"],
        help=(
            "Split to visualize. "
            "Use 'validation' to map XLSum val and Liputan6 dev. "
            "Default: test."
        ),
    )

    parser.add_argument(
        "--metric",
        default="rouge1",
        choices=["rouge1", "rouge2", "rougeL"],
        help="Metric used when generating oracle labels. Default: rouge1.",
    )

    parser.add_argument(
        "--mode",
        default="relative",
        choices=["relative", "absolute"],
        help="Sentence position mode. Default: relative.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins for relative mode. Default: 20.",
    )

    parser.add_argument(
        "--max_absolute_position",
        type=int,
        default=30,
        help="Maximum sentence position shown in absolute mode. Default: 30.",
    )

    parser.add_argument(
        "--count",
        action="store_true",
        help="Use raw count instead of normalized proportion.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path.",
    )

    args = parser.parse_args()

    if args.dataset == "both":
        datasets = ["xlsum", "liputan6"]
    else:
        datasets = [args.dataset]

    results = []

    for dataset in datasets:
        result = collect_positions_for_dataset(
            dataset=dataset,
            split=args.split,
            metric=args.metric,
            mode=args.mode,
            max_absolute_position=args.max_absolute_position,
        )
        results.append(result)

    output_dir = BASE_DIR / "dataset" / "oracle_position_histogram"

    if args.output:
        png_output_path = Path(args.output)
    else:
        if args.mode == "absolute":
            png_output_path = (
                output_dir
                / (
                    f"oracle_position_histogram_{args.dataset}_{args.metric}_"
                    f"{args.mode}_{args.split}_1-{args.max_absolute_position}.png"
                )
            )
        else:
            png_output_path = (
                output_dir
                / (
                    f"oracle_position_histogram_{args.dataset}_{args.metric}_"
                    f"{args.mode}_{args.split}.png"
                )
            )

    json_output_path = png_output_path.with_suffix(".json")

    plot_histogram(
        results=results,
        metric=args.metric,
        split=args.split,
        mode=args.mode,
        bins=args.bins,
        normalize=not args.count,
        output_path=png_output_path,
        max_absolute_position=args.max_absolute_position,
    )

    save_summary_json(
        results=results,
        metric=args.metric,
        split=args.split,
        mode=args.mode,
        output_path=json_output_path,
        max_absolute_position=args.max_absolute_position,
    )

    print("\n=== Oracle position histogram saved ===")
    print(f"PNG output  : {png_output_path}")
    print(f"JSON output : {json_output_path}")

    for result in results:
        dataset = result["dataset"]
        summary = summarize_positions(result["positions"], args.mode)

        print(f"\n[{dataset}]")
        print(f"Documents found                  : {result['total_docs']}")
        print(f"Documents with label             : {result['total_docs_with_label']}")
        print(f"Selected sentences shown         : {result['total_selected_sentences']}")
        print(
            "Selected sentences before filter : "
            f"{result['total_selected_sentences_before_filter']}"
        )
        print(f"Skipped missing file             : {result['skipped_missing_file']}")
        print(f"Skipped missing label            : {result['skipped_missing_label']}")
        print(f"Skipped length mismatch          : {result['skipped_length_mismatch']}")
        print(f"Skipped empty document           : {result['skipped_empty_document']}")

        if args.mode == "absolute":
            print(
                "Skipped outside absolute limit   : "
                f"{result['skipped_outside_absolute_limit']}"
            )

        if summary["count"] > 0:
            print(f"Mean position                    : {summary['mean']:.4f}")
            print(f"Median position                  : {summary['median']:.4f}")

            if args.mode == "relative":
                print(f"Early <= 30%                     : {summary['early_30_percent']:.2%}")
                print(f"Middle 30%-70%                   : {summary['middle_40_percent']:.2%}")
                print(f"Late > 70%                       : {summary['late_30_percent']:.2%}")


if __name__ == "__main__":
    main()