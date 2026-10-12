"""
Visualize sentence position histogram from model prediction results.

This script reads:
1. Preprocessed documents:
   dataset/<dataset>/preprocessed/<split>.jsonl

2. Prediction results from run_evaluation.py:
   results/<dataset>/<scenario_folder>/

The prediction file should contain selected sentence indices if available.

Supported prediction fields:
    - selected_indices
    - predicted_indices
    - pred_indices
    - indices
    - selected_ids

Fallback:
    If indices are not available, the script will try to infer indices from
    selected_sentences / predicted_sentences by matching them against the
    preprocessed document sentences.

Output:
    results/<dataset>/prediction_position_histogram/*.png
    results/<dataset>/prediction_position_histogram/*.json

Usage:
    uv run python scripts/visualize_prediction_position_histogram.py \
      --dataset liputan6 \
      --split dev \
      --results_dir results/liputan6/sbert_gcn_10ep_1e-3_256_bs16_tadaptive_Empnet_d00_Smlp_Psin16_Ybest_combine_rouge1_tie_rouge2 \
      --mode absolute \
      --max_absolute_position 30

      

If you already know the exact prediction file:
    uv run python scripts/visualize_prediction_position_histogram.py \
      --dataset liputan6 \
      --split dev \
      --prediction_path results/liputan6/<scenario>/dev_predictions.jsonl \
      --mode absolute \
      --max_absolute_position 30

Optional manual title override:
    --title_suffix "My model name"
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


def load_json_or_jsonl(path: Path) -> list[dict]:
    """
    Load prediction result file.

    Supports:
        - JSONL: one JSON object per line
        - JSON : either a list of objects, or a dict containing one of:
                 predictions, results, details, rows, data
    """

    if path.suffix == ".jsonl":
        return load_jsonl(path)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["predictions", "results", "details", "rows", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

            return [data]

    raise ValueError(f"Unsupported prediction file format: {path}")


def find_prediction_file(results_dir: Path, split: str) -> Path:
    """
    Auto-detect prediction file inside results/<dataset>/<scenario>/.

    The script prioritizes files whose names contain:
        split + prediction/pred/detail/result
    """

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    patterns = [
        f"*{split}*prediction*.jsonl",
        f"*{split}*pred*.jsonl",
        f"*{split}*detail*.jsonl",
        f"*{split}*result*.jsonl",
        f"*{split}*.jsonl",
        f"*{split}*prediction*.json",
        f"*{split}*pred*.json",
        f"*{split}*detail*.json",
        f"*{split}*result*.json",
        f"*{split}*.json",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(sorted(results_dir.glob(pattern)))

    unique_candidates = []
    seen = set()

    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)

    if not unique_candidates:
        raise FileNotFoundError(
            f"No prediction file found in {results_dir} for split '{split}'. "
            "Please use --prediction_path with the exact file path."
        )

    if len(unique_candidates) > 1:
        print("\nMultiple candidate prediction files found:")
        for i, path in enumerate(unique_candidates, start=1):
            print(f"  {i}. {path}")
        print(f"\nUsing first candidate: {unique_candidates[0]}")

    return unique_candidates[0]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def get_doc_id(row: dict) -> str:
    for key in ["id", "doc_id", "document_id", "source_id"]:
        if key in row:
            return str(row[key])

    raise KeyError(
        "Prediction row does not contain document id. "
        "Expected one of: id, doc_id, document_id, source_id."
    )


def get_indices_from_prediction(row: dict) -> list[int] | None:
    """
    Try to retrieve predicted selected sentence indices.

    Supported field names:
        selected_indices
        predicted_indices
        pred_indices
        indices
        selected_ids
    """

    for key in [
        "selected_indices",
        "predicted_indices",
        "pred_indices",
        "indices",
        "selected_ids",
    ]:
        if key in row and row[key] is not None:
            return [int(idx) for idx in row[key]]

    return None


def get_sentences_from_prediction(row: dict) -> list[str] | None:
    """
    Fallback if selected indices are not available.

    The script will try to infer sentence indices by matching selected
    sentence text against preprocessed document sentences.
    """

    for key in [
        "selected_sentences",
        "predicted_sentences",
        "sentences_selected",
        "summary_sentences",
    ]:
        if key in row and row[key] is not None:
            return [str(sent) for sent in row[key]]

    return None


def infer_indices_from_sentences(
    document_sentences: list[str],
    selected_sentences: list[str],
) -> list[int]:
    """
    Infer selected indices by matching predicted selected sentences with
    original preprocessed sentences.

    This is a fallback. The most reliable input is still selected_indices.
    """

    normalized_doc = [normalize_text(sent) for sent in document_sentences]
    used_indices = set()
    inferred_indices = []

    for selected in selected_sentences:
        selected_norm = normalize_text(selected)
        matched_idx = None

        # Exact match first.
        for idx, sent_norm in enumerate(normalized_doc):
            if idx in used_indices:
                continue

            if selected_norm == sent_norm:
                matched_idx = idx
                break

        # Loose containment match as fallback.
        if matched_idx is None:
            for idx, sent_norm in enumerate(normalized_doc):
                if idx in used_indices:
                    continue

                if selected_norm and (
                    selected_norm in sent_norm or sent_norm in selected_norm
                ):
                    matched_idx = idx
                    break

        if matched_idx is not None:
            inferred_indices.append(matched_idx)
            used_indices.add(matched_idx)

    return inferred_indices


def calculate_position(index: int, num_sentences: int, mode: str) -> float:
    """
    mode = absolute:
        1 = first sentence, 2 = second sentence, etc.

    mode = relative:
        0.0 = first sentence, 1.0 = last sentence
    """

    if mode == "absolute":
        return float(index + 1)

    if num_sentences <= 1:
        return 0.0

    return index / (num_sentences - 1)


def collect_prediction_positions(
    dataset: str,
    split: str,
    prediction_path: Path,
    mode: str,
    max_absolute_position: int,
) -> dict:
    prep_path = (
        BASE_DIR
        / "dataset"
        / dataset
        / "preprocessed"
        / f"{split}.jsonl"
    )

    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {prep_path}")

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    prep_docs = load_jsonl(prep_path)
    prediction_rows = load_json_or_jsonl(prediction_path)

    sentences_map = {
        str(doc["id"]): doc["sentences"]
        for doc in prep_docs
    }

    prediction_map = {}

    skipped_prediction_without_id = 0

    for row in prediction_rows:
        try:
            doc_id = get_doc_id(row)
            prediction_map[doc_id] = row
        except KeyError:
            skipped_prediction_without_id += 1
            continue

    all_positions = []

    total_docs = len(prep_docs)
    total_docs_with_prediction = 0
    total_selected_sentences = 0
    total_selected_sentences_before_filter = 0

    skipped_missing_prediction = 0
    skipped_empty_document = 0
    skipped_invalid_index = 0
    skipped_outside_absolute_limit = 0
    skipped_no_indices = 0
    inferred_from_sentences = 0

    for doc_id, sentences in sentences_map.items():
        if doc_id not in prediction_map:
            skipped_missing_prediction += 1
            continue

        if not sentences:
            skipped_empty_document += 1
            continue

        prediction = prediction_map[doc_id]

        selected_indices = get_indices_from_prediction(prediction)

        if selected_indices is None:
            selected_sentences = get_sentences_from_prediction(prediction)

            if selected_sentences is None:
                skipped_no_indices += 1
                continue

            selected_indices = infer_indices_from_sentences(
                document_sentences=sentences,
                selected_sentences=selected_sentences,
            )
            inferred_from_sentences += 1

        total_docs_with_prediction += 1
        total_selected_sentences_before_filter += len(selected_indices)

        for idx in selected_indices:
            if idx < 0 or idx >= len(sentences):
                skipped_invalid_index += 1
                continue

            position = calculate_position(
                index=idx,
                num_sentences=len(sentences),
                mode=mode,
            )

            if mode == "absolute" and position > max_absolute_position:
                skipped_outside_absolute_limit += 1
                continue

            all_positions.append(position)

        total_selected_sentences = len(all_positions)

    return {
        "dataset": dataset,
        "split": split,
        "prediction_path": str(prediction_path),
        "positions": all_positions,
        "total_docs": total_docs,
        "total_prediction_rows": len(prediction_rows),
        "total_docs_with_prediction": total_docs_with_prediction,
        "total_selected_sentences": total_selected_sentences,
        "total_selected_sentences_before_filter": total_selected_sentences_before_filter,
        "skipped_prediction_without_id": skipped_prediction_without_id,
        "skipped_missing_prediction": skipped_missing_prediction,
        "skipped_empty_document": skipped_empty_document,
        "skipped_invalid_index": skipped_invalid_index,
        "skipped_outside_absolute_limit": skipped_outside_absolute_limit,
        "skipped_no_indices": skipped_no_indices,
        "inferred_from_sentences": inferred_from_sentences,
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
    result: dict,
    mode: str,
    bins: int,
    normalize: bool,
    output_path: Path,
    max_absolute_position: int,
    title_suffix: str | None,
) -> None:
    positions = result["positions"]

    fig, ax = plt.subplots(figsize=(10, 5))

    if not positions:
        ax.text(
            0.5,
            0.5,
            "No prediction positions found",
            ha="center",
            va="center",
        )
    else:
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
            ax.hist(
                positions,
                bins=max_absolute_position,
                range=(1, max_absolute_position + 1),
                weights=weights,
            )
            ax.set_xlim(1, max_absolute_position)
            ax.set_xlabel("Sentence position")

            if max_absolute_position <= 30:
                ax.set_xticks(range(1, max_absolute_position + 1))
                ax.tick_params(axis="x", labelrotation=90)
            else:
                step = max(1, max_absolute_position // 15)
                ax.set_xticks(range(1, max_absolute_position + 1, step))

        summary = summarize_positions(positions, mode)

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

    title = (
        f"Predicted sentence position histogram | "
        f"{result['dataset']} | split={result['split']} | mode={mode}"
    )

    if title_suffix:
        title = f"{title}\n{title_suffix}"

    ax.set_title(title)
    ax.set_ylabel("Proportion" if normalize else "Count")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_json(
    result: dict,
    mode: str,
    output_path: Path,
    max_absolute_position: int,
) -> None:
    summary = {
        "dataset": result["dataset"],
        "split": result["split"],
        "prediction_path": result["prediction_path"],
        "mode": mode,
        "max_absolute_position": max_absolute_position if mode == "absolute" else None,
        "total_docs": result["total_docs"],
        "total_prediction_rows": result["total_prediction_rows"],
        "total_docs_with_prediction": result["total_docs_with_prediction"],
        "total_selected_sentences_shown": result["total_selected_sentences"],
        "total_selected_sentences_before_filter": result[
            "total_selected_sentences_before_filter"
        ],
        "skipped_prediction_without_id": result["skipped_prediction_without_id"],
        "skipped_missing_prediction": result["skipped_missing_prediction"],
        "skipped_empty_document": result["skipped_empty_document"],
        "skipped_invalid_index": result["skipped_invalid_index"],
        "skipped_outside_absolute_limit": result["skipped_outside_absolute_limit"],
        "skipped_no_indices": result["skipped_no_indices"],
        "inferred_from_sentences": result["inferred_from_sentences"],
        "position_summary": summarize_positions(result["positions"], mode),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def resolve_title_suffix(
    title_suffix_arg: str | None,
    results_dir_arg: str | None,
    prediction_path: Path,
) -> str:
    """
    Title priority:
        1. Use --title_suffix if provided.
        2. Else use scenario folder name from --results_dir.
        3. Else use parent folder name of --prediction_path.
    """

    if title_suffix_arg:
        return title_suffix_arg

    if results_dir_arg:
        return Path(results_dir_arg).name

    return prediction_path.parent.name


def resolve_output_path(
    dataset: str,
    split: str,
    mode: str,
    max_absolute_position: int,
    output_arg: str | None,
    results_dir_arg: str | None,
    prediction_path: Path,
) -> Path:
    if output_arg:
        return Path(output_arg)

    output_dir = BASE_DIR / "results" / dataset / "prediction_position_histogram"

    scenario_name = (
        Path(results_dir_arg).name
        if results_dir_arg
        else prediction_path.parent.name
    )

    if mode == "absolute":
        return (
            output_dir
            / (
                f"prediction_position_histogram_{scenario_name}_"
                f"{split}_1-{max_absolute_position}.png"
            )
        )

    return (
        output_dir
        / (
            f"prediction_position_histogram_{scenario_name}_"
            f"{split}_{mode}.png"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize predicted sentence position histogram from model results."
    )

    parser.add_argument(
        "--dataset",
        choices=["xlsum", "liputan6"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "dev", "test"],
        required=True,
    )

    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Directory under results/<dataset>/<scenario>. "
            "If provided, the script auto-detects the prediction file."
        ),
    )

    parser.add_argument(
        "--prediction_path",
        default=None,
        help="Exact prediction JSON/JSONL path. Overrides --results_dir.",
    )

    parser.add_argument(
        "--mode",
        default="absolute",
        choices=["relative", "absolute"],
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for relative mode.",
    )

    parser.add_argument(
        "--max_absolute_position",
        type=int,
        default=30,
        help="Maximum sentence position shown in absolute mode.",
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

    parser.add_argument(
        "--title_suffix",
        default=None,
        help=(
            "Optional title suffix. If omitted, the scenario folder name "
            "from --results_dir is used automatically."
        ),
    )

    args = parser.parse_args()

    if args.prediction_path:
        prediction_path = Path(args.prediction_path)
    elif args.results_dir:
        prediction_path = find_prediction_file(
            results_dir=Path(args.results_dir),
            split=args.split,
        )
    else:
        raise ValueError("You must provide either --prediction_path or --results_dir.")

    title_suffix = resolve_title_suffix(
        title_suffix_arg=args.title_suffix,
        results_dir_arg=args.results_dir,
        prediction_path=prediction_path,
    )

    result = collect_prediction_positions(
        dataset=args.dataset,
        split=args.split,
        prediction_path=prediction_path,
        mode=args.mode,
        max_absolute_position=args.max_absolute_position,
    )

    png_output_path = resolve_output_path(
        dataset=args.dataset,
        split=args.split,
        mode=args.mode,
        max_absolute_position=args.max_absolute_position,
        output_arg=args.output,
        results_dir_arg=args.results_dir,
        prediction_path=prediction_path,
    )

    json_output_path = png_output_path.with_suffix(".json")

    plot_histogram(
        result=result,
        mode=args.mode,
        bins=args.bins,
        normalize=not args.count,
        output_path=png_output_path,
        max_absolute_position=args.max_absolute_position,
        title_suffix=title_suffix,
    )

    save_summary_json(
        result=result,
        mode=args.mode,
        output_path=json_output_path,
        max_absolute_position=args.max_absolute_position,
    )

    print("\n=== Prediction position histogram saved ===")
    print(f"Prediction input : {prediction_path}")
    print(f"PNG output       : {png_output_path}")
    print(f"JSON output      : {json_output_path}")

    summary = summarize_positions(result["positions"], args.mode)

    print(f"\n[{args.dataset}]")
    print(f"Split                              : {result['split']}")
    print(f"Documents found                    : {result['total_docs']}")
    print(f"Prediction rows                    : {result['total_prediction_rows']}")
    print(f"Documents with prediction           : {result['total_docs_with_prediction']}")
    print(f"Selected sentences shown            : {result['total_selected_sentences']}")
    print(
        "Selected sentences before filter   : "
        f"{result['total_selected_sentences_before_filter']}"
    )
    print(f"Skipped prediction without id        : {result['skipped_prediction_without_id']}")
    print(f"Skipped missing prediction          : {result['skipped_missing_prediction']}")
    print(f"Skipped empty document              : {result['skipped_empty_document']}")
    print(f"Skipped invalid index               : {result['skipped_invalid_index']}")
    print(f"Skipped no indices/sentences         : {result['skipped_no_indices']}")
    print(f"Inferred from selected_sentences     : {result['inferred_from_sentences']}")

    if args.mode == "absolute":
        print(
            "Skipped outside absolute limit     : "
            f"{result['skipped_outside_absolute_limit']}"
        )

    if summary["count"] > 0:
        print(f"Mean position                      : {summary['mean']:.4f}")
        print(f"Median position                    : {summary['median']:.4f}")

        if args.mode == "relative":
            print(f"Early <= 30%                       : {summary['early_30_percent']:.2%}")
            print(f"Middle 30%-70%                     : {summary['middle_40_percent']:.2%}")
            print(f"Late > 70%                         : {summary['late_30_percent']:.2%}")


if __name__ == "__main__":
    main()