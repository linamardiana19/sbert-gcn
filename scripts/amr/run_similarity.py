"""
AMR Similarity Pipeline - Multiprocessing
=========================================

Purpose
-------
Calculate pairwise SMATCH similarity between sentence-level AMR graphs.

Processing is parallelized PER ARTICLE using multiple CPU processes.

Input
-----
dataset/<dataset>/amr/parsed/<split>.pt

Expected document:
{
    "id": "...",
    "sentences": [...],
    "amr": [...]
}

Output
------
dataset/<dataset>/amr/similarity/smatch/<split>.pt

Output document:
{
    "id": "...",
    "num_nodes": N,
    "pairs": [
        (src, dst, smatch_score),
        ...
    ],
    "num_pairs": ...,
    "errors": [...]
}

Run example
-----------

Validation:

    uv run python scripts/amr/run_similarity.py \
        --dataset xlsum \
        --method smatch \
        --split val \
        --workers 4

Checkpoint
----------
Progress is saved periodically to:

dataset/<dataset>/amr/similarity/smatch/<split>.ckpt
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import torch
from tqdm import tqdm


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

sys.path.insert(
    0,
    str(BASE_DIR)
)

import smatch


# ============================================================
# CONFIG
# ============================================================

DATASET_CONFIG = {
    "xlsum": {
        "splits": [
            "train",
            "val",
            "test",
        ]
    },

    "liputan6": {
        "splits": [
            "train",
            "dev",
            "test",
        ]
    },
}


DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_WORKERS = 4


# ============================================================
# CLEAN AMR
# ============================================================

def clean_amr_for_smatch(amr_text):
    """
    Remove metadata/comment lines before SMATCH.

    Example removed:
        # ::id ...
        # ::snt ...
        # ::tok ...

    Also removes blank lines.
    """

    if amr_text is None:
        return ""

    lines = []

    for line in str(amr_text).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        lines.append(line)

    return " ".join(lines)


# ============================================================
# CALCULATE SMATCH
# ============================================================

def calculate_smatch(
    amr1,
    amr2,
):
    """
    Calculate SMATCH F1 between two AMRs.

    Returns
    -------
    Success:
        (score, None)

    Failure:
        (None, error_message)
    """

    try:

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        amr1 = clean_amr_for_smatch(
            amr1
        )

        amr2 = clean_amr_for_smatch(
            amr2
        )

        # ----------------------------------------------------
        # EMPTY CHECK
        # ----------------------------------------------------

        if not amr1:
            return (
                None,
                "AMR1 empty after cleaning",
            )

        if not amr2:
            return (
                None,
                "AMR2 empty after cleaning",
            )

        # ----------------------------------------------------
        # RESET SMATCH CACHE
        # ----------------------------------------------------

        if hasattr(
            smatch,
            "match_triple_dict",
        ):
            smatch.match_triple_dict.clear()

        # ----------------------------------------------------
        # SMATCH
        # ----------------------------------------------------

        (
            match_num,
            test_num,
            gold_num,
        ) = smatch.get_amr_match(
            amr1,
            amr2,
        )

        (
            precision,
            recall,
            f1,
        ) = smatch.compute_f(
            match_num,
            test_num,
            gold_num,
        )

        return (
            float(f1),
            None,
        )

    except Exception as exc:

        return (
            None,
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# VALIDATE ARTICLE
# ============================================================

def validate_article(item):
    """
    Basic validation before SMATCH.
    """

    if not isinstance(
        item,
        dict,
    ):
        return (
            False,
            "record is not dict",
        )

    if "id" not in item:
        return (
            False,
            "missing id",
        )

    if "sentences" not in item:
        return (
            False,
            "missing sentences",
        )

    if "amr" not in item:
        return (
            False,
            "missing amr",
        )

    if not isinstance(
        item["amr"],
        (list, tuple),
    ):
        return (
            False,
            "amr is not list/tuple",
        )

    if len(
        item["sentences"]
    ) != len(
        item["amr"]
    ):
        return (
            False,
            (
                "sentence-AMR mismatch: "
                f"{len(item['sentences'])} "
                f"vs "
                f"{len(item['amr'])}"
            ),
        )

    return (
        True,
        None,
    )


# ============================================================
# PROCESS ONE ARTICLE
# ============================================================

def process_article(item):
    """
    Worker function.

    For N AMRs, calculate only unique pairs:

        (0,1)
        (0,2)
        ...
        (N-2,N-1)

    Reverse pair is not calculated because similarity
    is treated as symmetric.
    """

    valid, validation_error = (
        validate_article(
            item
        )
    )

    if not valid:

        return {
            "_status":
                "skipped",

            "id":
                item.get(
                    "id",
                    None,
                )
                if isinstance(
                    item,
                    dict
                )
                else None,

            "error":
                validation_error,
        }

    article_id = str(
        item["id"]
    )

    amr_graphs = item[
        "amr"
    ]

    num_nodes = len(
        amr_graphs
    )

    pairs = []

    errors = []

    # ========================================================
    # UNIQUE PAIRWISE SMATCH
    # ========================================================

    for i, j in combinations(
        range(num_nodes),
        2,
    ):

        score, error_message = (
            calculate_smatch(
                amr_graphs[i],
                amr_graphs[j],
            )
        )

        if score is None:

            errors.append({
                "src": i,
                "dst": j,
                "error": error_message,
            })

            continue

        pairs.append(
            (
                i,
                j,
                score,
            )
        )

    return {
        "_status":
            "success",

        "id":
            article_id,

        "num_nodes":
            num_nodes,

        "pairs":
            pairs,

        "num_pairs":
            len(pairs),

        "errors":
            errors,
    }


# ============================================================
# CLEAN INTERNAL STATUS
# ============================================================

def clean_result(result):

    return {
        "id":
            result["id"],

        "num_nodes":
            result["num_nodes"],

        "pairs":
            result["pairs"],

        "num_pairs":
            result["num_pairs"],

        "errors":
            result["errors"],
    }


# ============================================================
# SHOW SAMPLE ERRORS
# ============================================================

def show_sample_errors(
    results,
    max_errors=5,
):

    sample_errors = []

    for result in results:

        article_id = result[
            "id"
        ]

        for error in result[
            "errors"
        ]:

            sample_errors.append({
                "id":
                    article_id,

                "src":
                    error.get(
                        "src"
                    ),

                "dst":
                    error.get(
                        "dst"
                    ),

                "error":
                    error.get(
                        "error"
                    ),
            })

            if (
                len(sample_errors)
                >= max_errors
            ):
                break

        if (
            len(sample_errors)
            >= max_errors
        ):
            break

    if not sample_errors:
        return

    print()
    print("=" * 80)
    print("SAMPLE SMATCH ERRORS")
    print("=" * 80)

    for index, error in enumerate(
        sample_errors,
        start=1,
    ):

        print()

        print(
            f"Error #{index}"
        )

        print(
            "Article ID :",
            error["id"],
        )

        print(
            "Pair       :",
            f"S{error['src']} "
            f"vs "
            f"S{error['dst']}",
        )

        print(
            "Message    :",
            error["error"],
        )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    input_path,
    output_path,
    checkpoint_interval,
    workers,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_path.with_suffix(
            ".ckpt"
        )
    )

    # ========================================================
    # LOAD INPUT
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"INPUT: {input_path}"
    )
    print("=" * 80)

    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Total articles     :",
        f"{len(data):,}",
    )

    # ========================================================
    # KEEP ORIGINAL ORDER
    # ========================================================

    order_map = {

        str(item["id"]):
            index

        for index, item in enumerate(
            data
        )

        if (
            isinstance(
                item,
                dict
            )
            and "id" in item
        )
    }

    # ========================================================
    # RESUME CHECKPOINT
    # ========================================================

    if checkpoint_path.exists():

        print()
        print(
            "Checkpoint found:"
        )

        print(
            checkpoint_path
        )

        results = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        done_ids = {
            str(
                item["id"]
            )
            for item in results
        }

        print(
            "Already processed  :",
            f"{len(done_ids):,}",
        )

    else:

        results = []

        done_ids = set()

    # ========================================================
    # REMAINING ARTICLES
    # ========================================================

    remaining = [

        item

        for item in data

        if (
            isinstance(
                item,
                dict,
            )
            and "id" in item
            and str(
                item["id"]
            ) not in done_ids
        )
    ]

    print(
        "Remaining articles :",
        f"{len(remaining):,}",
    )

    print(
        "CPU workers        :",
        workers,
    )

    # ========================================================
    # MULTIPROCESSING
    # ========================================================

    newly_completed = 0

    skipped_articles = []

    if len(
        remaining
    ) > 0:

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {

                executor.submit(
                    process_article,
                    item,
                ):
                str(
                    item["id"]
                )

                for item in remaining
            }

            for future in tqdm(
                as_completed(
                    futures
                ),
                total=len(
                    futures
                ),
                desc="Calculating SMATCH",
            ):

                article_id = futures[
                    future
                ]

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    skipped_articles.append({
                        "id":
                            article_id,

                        "error":
                            (
                                f"Worker error: "
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                    })

                    continue

                # ------------------------------------------------
                # SKIPPED
                # ------------------------------------------------

                if (
                    result.get(
                        "_status"
                    )
                    == "skipped"
                ):

                    skipped_articles.append({
                        "id":
                            result.get(
                                "id"
                            ),

                        "error":
                            result.get(
                                "error"
                            ),
                    })

                    continue

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                result = clean_result(
                    result
                )

                results.append(
                    result
                )

                done_ids.add(
                    str(
                        result["id"]
                    )
                )

                newly_completed += 1

                # =================================================
                # CHECKPOINT
                # =================================================

                if (
                    newly_completed
                    % checkpoint_interval
                    == 0
                ):

                    results.sort(
                        key=lambda x:
                            order_map.get(
                                str(
                                    x["id"]
                                ),
                                10**15,
                            )
                    )

                    torch.save(
                        results,
                        checkpoint_path,
                    )

    # ========================================================
    # RESTORE ORIGINAL ORDER
    # ========================================================

    results.sort(
        key=lambda x:
            order_map.get(
                str(
                    x["id"]
                ),
                10**15,
            )
    )

    # ========================================================
    # SAVE FINAL
    # ========================================================

    torch.save(
        results,
        output_path,
    )

    # ========================================================
    # REMOVE CHECKPOINT
    # ========================================================

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ========================================================
    # SUMMARY
    # ========================================================

    total_possible_pairs = sum(

        (
            item["num_nodes"]
            * (
                item["num_nodes"]
                - 1
            )
            // 2
        )

        for item in results
    )

    total_successful_pairs = sum(

        item[
            "num_pairs"
        ]

        for item in results
    )

    total_errors = sum(

        len(
            item[
                "errors"
            ]
        )

        for item in results
    )

    print()
    print("-" * 80)

    print(
        "Processed articles :",
        f"{len(results):,}",
    )

    print(
        "Possible pairs     :",
        f"{total_possible_pairs:,}",
    )

    print(
        "Successful pairs   :",
        f"{total_successful_pairs:,}",
    )

    print(
        "SMATCH errors      :",
        f"{total_errors:,}",
    )

    if (
        total_possible_pairs
        > 0
    ):

        success_rate = (
            total_successful_pairs
            / total_possible_pairs
            * 100
        )

        print(
            "Success rate      :",
            f"{success_rate:.4f}%",
        )

    print(
        "Skipped articles   :",
        f"{len(skipped_articles):,}",
    )

    print(
        "Workers used       :",
        workers,
    )

    print(
        "Saved              :",
        output_path,
    )

    # ========================================================
    # CONSISTENCY CHECK
    # ========================================================

    if (
        total_successful_pairs
        + total_errors
        != total_possible_pairs
    ):

        print()
        print(
            "[WARNING] "
            "successful + errors != possible pairs"
        )

        print(
            "Possible   :",
            total_possible_pairs,
        )

        print(
            "Successful :",
            total_successful_pairs,
        )

        print(
            "Errors     :",
            total_errors,
        )

    # ========================================================
    # ERROR SAMPLE
    # ========================================================

    if total_errors > 0:

        show_sample_errors(
            results,
            max_errors=5,
        )

    # ========================================================
    # SKIPPED SAMPLE
    # ========================================================

    if skipped_articles:

        print()
        print("=" * 80)
        print("SAMPLE SKIPPED ARTICLES")
        print("=" * 80)

        for item in skipped_articles[:5]:
            print(item)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate pairwise AMR "
            "SMATCH similarity using "
            "multiprocessing."
        )
    )

    # ========================================================
    # DATASET
    # ========================================================

    parser.add_argument(
        "--dataset",
        choices=[
            "xlsum",
            "liputan6",
        ],
        required=True,
    )

    # ========================================================
    # METHOD
    # ========================================================

    parser.add_argument(
        "--method",
        choices=[
            "smatch",
        ],
        default="smatch",
    )

    # ========================================================
    # SPLIT
    # ========================================================

    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Process one split only. "
            "If omitted, process all splits."
        ),
    )

    # ========================================================
    # WORKERS
    # ========================================================

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of CPU worker processes. "
            "Default: 4."
        ),
    )

    # ========================================================
    # CHECKPOINT INTERVAL
    # ========================================================

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=(
            "Save checkpoint after N "
            "newly completed articles."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE WORKERS
    # ========================================================

    if args.workers < 1:

        raise ValueError(
            "--workers must be >= 1"
        )

    # ========================================================
    # DATASET CONFIG
    # ========================================================

    config = DATASET_CONFIG[
        args.dataset
    ]

    if args.split:

        if (
            args.split
            not in config["splits"]
        ):

            raise ValueError(
                f"Invalid split "
                f"'{args.split}' "
                f"for dataset "
                f"'{args.dataset}'. "
                f"Available: "
                f"{config['splits']}"
            )

        splits = [
            args.split
        ]

    else:

        splits = config[
            "splits"
        ]

    # ========================================================
    # PATHS
    # ========================================================

    dataset_dir = (
        BASE_DIR
        / "dataset"
        / args.dataset
    )

    input_dir = (
        dataset_dir
        / "amr"
        / "parsed"
    )

    output_dir = (
        dataset_dir
        / "amr"
        / "similarity"
        / args.method
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print(
        "AMR PAIRWISE SMATCH - MULTIPROCESSING"
    )
    print("=" * 80)

    print(
        "Dataset       :",
        args.dataset,
    )

    print(
        "Method        :",
        args.method,
    )

    print(
        "Input folder  :",
        input_dir,
    )

    print(
        "Output folder :",
        output_dir,
    )

    print(
        "Workers       :",
        args.workers,
    )

    print(
        "Checkpoint    : every",
        args.checkpoint_interval,
        "completed articles",
    )

    # ========================================================
    # RUN SPLITS
    # ========================================================

    for split in splits:

        input_path = (
            input_dir
            / f"{split}.pt"
        )

        output_path = (
            output_dir
            / f"{split}.pt"
        )

        if not input_path.exists():

            print()
            print(
                f"[SKIP] {split}: "
                f"input not found"
            )

            print(
                input_path
            )

            continue

        print()
        print("=" * 80)
        print(
            f"SMATCH - {split.upper()}"
        )
        print("=" * 80)

        process_split(
            input_path=
                input_path,

            output_path=
                output_path,

            checkpoint_interval=
                args.checkpoint_interval,

            workers=
                args.workers,
        )

    print()
    print("=" * 80)
    print("SMATCH FINISHED")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()