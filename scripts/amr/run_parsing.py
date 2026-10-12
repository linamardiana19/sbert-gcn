"""
AMR Parsing Pipeline
====================

Purpose
-------
Convert each Indonesian sentence into an AMR graph using
the Indonesian AMRBART parser.

Default Input
-------------
dataset/<dataset>/preprocessed/<split>.jsonl

Optional Input
--------------
You can provide a custom input directory using:

    --input_dir

Example:

    dataset/xlsum/amr/reparse_input/<split>.jsonl

Default Output
--------------
dataset/<dataset>/amr/parsed/<split>.pt

Optional Output
---------------
You can provide a custom output directory using:

    --output_dir

Example:

    dataset/xlsum/amr/reparsed_fajri/<split>.pt


Expected Input Format
---------------------

Each JSONL document must contain:

    {
        "id": "...",
        "sentences": [
            "Kalimat pertama.",
            "Kalimat kedua."
        ]
    }


Output Format
-------------

Each output document contains:

    {
        "id": "...",
        "sentences": [...],
        "amr": [
            "(z0 / ...)",
            "(z0 / ...)"
        ]
    }


Failed Parsing
--------------

If one sentence fails to parse, its AMR value is stored as:

    None

The script does not replace failed AMR parsing with
a dummy graph.


Checkpoint
----------

Progress is saved every CHECKPOINT_INTERVAL documents.

Checkpoint path:

    <output_dir>/<split>.ckpt

If execution is interrupted, rerunning the same command
will resume from the checkpoint.


How to Run
----------

1. Normal parsing using default preprocessing input:

    python3.10 scripts/amr/run_parsing.py \
        --dataset xlsum \
        --split test \
        --device cuda:0 \
        --batch_size 4


2. Re-parse mismatch/missing articles:

    python3.10 scripts/amr/run_parsing.py \
        --dataset xlsum \
        --split test \
        --input_dir dataset/xlsum/amr/reparse_input \
        --output_dir dataset/xlsum/amr/reparsed_fajri \
        --device cuda:0 \
        --batch_size 4


3. Re-parse train:

    python3.10 scripts/amr/run_parsing.py \
        --dataset xlsum \
        --split train \
        --input_dir dataset/xlsum/amr/reparse_input \
        --output_dir dataset/xlsum/amr/reparsed_fajri \
        --device cuda:0 \
        --batch_size 4


4. Re-parse validation:

    python3.10 scripts/amr/run_parsing.py \
        --dataset xlsum \
        --split val \
        --input_dir dataset/xlsum/amr/reparse_input \
        --output_dir dataset/xlsum/amr/reparsed_fajri \
        --device cuda:0 \
        --batch_size 4


5. Process all available splits:

    python3.10 scripts/amr/run_parsing.py \
        --dataset xlsum \
        --input_dir dataset/xlsum/amr/reparse_input \
        --output_dir dataset/xlsum/amr/reparsed_fajri \
        --device cuda:0 \
        --batch_size 4
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

AMR_REPO_DIR = (
    BASE_DIR
    / "models"
    / "amr_abdi"
)

AMRBART_DIR = (
    AMR_REPO_DIR
    / "AMRBART-id"
)

sys.path.insert(
    0,
    str(BASE_DIR)
)

sys.path.insert(
    0,
    str(AMR_REPO_DIR)
)


import penman

from text_to_amr import TextToAMR


# ============================================================
# DATASET CONFIG
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


# Save checkpoint every N documents
CHECKPOINT_INTERVAL = 50


# ============================================================
# HELPER
# ============================================================

def graph_to_penman(graph) -> str | None:
    """
    Convert penman.Graph into PENMAN string.

    If graph is None or encoding fails,
    return None.
    """

    if graph is None:
        return None

    try:
        return penman.encode(graph)

    except Exception:
        return None


# ============================================================
# PARSE ONE SPLIT
# ============================================================

def parse_split(
    parser: TextToAMR,
    input_path: Path,
    output_path: Path,
    batch_size: int = 8,
) -> None:

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Example:
    #
    # test.pt
    # ->
    # test.ckpt

    checkpoint_path = (
        output_path.with_suffix(".ckpt")
    )

    # ========================================================
    # LOAD INPUT DOCUMENTS
    # ========================================================

    docs = []

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            docs.append(
                json.loads(line)
            )

    print(
        f"  Input           : {input_path}"
    )

    print(
        f"  Output          : {output_path}"
    )

    print(
        f"  Total documents : {len(docs):,}"
    )

    # ========================================================
    # RESUME CHECKPOINT
    # ========================================================

    results = []
    done_ids = set()

    if checkpoint_path.exists():

        print()
        print(
            f"  Checkpoint found: "
            f"{checkpoint_path}"
        )

        results = torch.load(
            checkpoint_path,
            weights_only=False,
        )

        done_ids = {
            str(item["id"])
            for item in results
        }

        if len(docs) > 0:

            progress_pct = (
                len(done_ids)
                / len(docs)
            ) * 100

        else:

            progress_pct = 0.0

        print(
            f"  Resumed from checkpoint: "
            f"{len(done_ids):,}/{len(docs):,} "
            f"articles already parsed "
            f"({progress_pct:.2f}%)"
        )

    # ========================================================
    # FIND REMAINING DOCUMENTS
    # ========================================================

    remaining = [
        doc
        for doc in docs
        if str(doc["id"]) not in done_ids
    ]

    print(
        f"  Remaining       : "
        f"{len(remaining):,}"
    )

    # ========================================================
    # ALL DOCUMENTS ALREADY DONE
    # ========================================================

    if not remaining:

        print()
        print(
            "  All documents already parsed."
        )

        # If a checkpoint exists but final .pt
        # has not yet been written, finalize it.

        if checkpoint_path.exists():

            torch.save(
                results,
                output_path,
            )

            checkpoint_path.unlink()

            print(
                f"  Final output saved: "
                f"{output_path}"
            )

        return

    # ========================================================
    # STATISTICS
    # ========================================================

    failed_sentences = 0
    total_sentences = 0

    # ========================================================
    # PARSE DOCUMENTS
    # ========================================================

    for doc_idx, doc in enumerate(

        tqdm(
            remaining,
            desc="  Parsing AMR",
        )

    ):

        doc_id = str(
            doc["id"]
        )

        sentences = (
            doc["sentences"]
        )

        total_sentences += (
            len(sentences)
        )

        amr_strings = []

        # ----------------------------------------------------
        # EMPTY DOCUMENT
        # ----------------------------------------------------

        if len(sentences) == 0:

            results.append({
                "id": doc_id,
                "sentences": sentences,
                "amr": [],
            })

            # Even empty documents contribute
            # to checkpoint progress.

            if (
                (doc_idx + 1)
                % CHECKPOINT_INTERVAL
                == 0
            ):

                torch.save(
                    results,
                    checkpoint_path,
                )

                total_parsed = (
                    len(results)
                )

                total_docs = (
                    len(docs)
                )

                progress_pct = (
                    total_parsed
                    / total_docs
                    * 100
                )

                print(
                    f"\n  Checkpoint saved: "
                    f"{total_parsed:,}/"
                    f"{total_docs:,} "
                    f"articles parsed "
                    f"({progress_pct:.2f}%)"
                )

            continue

        # ====================================================
        # BATCH PARSING
        # ====================================================

        for start in range(
            0,
            len(sentences),
            batch_size,
        ):

            batch = sentences[
                start:start + batch_size
            ]

            try:

                with torch.no_grad():

                    graphs = parser(
                        batch,
                        method=1,
                    )

                # --------------------------------------------
                # Safety check
                # --------------------------------------------

                if len(graphs) != len(batch):

                    print()

                    print(
                        f"WARNING: "
                        f"document={doc_id}, "
                        f"batch_start={start}, "
                        f"expected={len(batch)}, "
                        f"got={len(graphs)}"
                    )

                # --------------------------------------------
                # Convert graph -> PENMAN string
                # --------------------------------------------

                for graph in graphs:

                    amr_string = (
                        graph_to_penman(
                            graph
                        )
                    )

                    if amr_string is None:

                        failed_sentences += 1

                    amr_strings.append(
                        amr_string
                    )

                # --------------------------------------------
                # Parser produced fewer graphs
                # --------------------------------------------

                missing = (
                    len(batch)
                    - len(graphs)
                )

                if missing > 0:

                    amr_strings.extend(
                        [None] * missing
                    )

                    failed_sentences += (
                        missing
                    )

                # --------------------------------------------
                # Parser produced MORE graphs
                # --------------------------------------------

                if len(graphs) > len(batch):

                    extra = (
                        len(graphs)
                        - len(batch)
                    )

                    print(
                        f"\nWARNING: "
                        f"document={doc_id} "
                        f"returned {extra} "
                        f"extra graph(s)."
                    )

                    # Keep strict sentence alignment.
                    #
                    # Remove graphs beyond batch size.

                    amr_strings = (
                        amr_strings[:-extra]
                    )

            except Exception as exc:

                print()

                print(
                    f"ERROR parsing "
                    f"document={doc_id}, "
                    f"batch_start={start}: "
                    f"{exc}"
                )

                # Do not create dummy graph.
                #
                # Failed sentences are represented
                # as None.

                amr_strings.extend(
                    [None] * len(batch)
                )

                failed_sentences += (
                    len(batch)
                )

        # ====================================================
        # ALIGNMENT VALIDATION
        # ====================================================

        if (
            len(amr_strings)
            != len(sentences)
        ):

            raise RuntimeError(
                f"AMR alignment error "
                f"for document {doc_id}: "
                f"{len(sentences)} sentences "
                f"but "
                f"{len(amr_strings)} AMRs."
            )

        # ====================================================
        # SAVE DOCUMENT RESULT
        # ====================================================

        results.append({
            "id": doc_id,
            "sentences": sentences,
            "amr": amr_strings,
        })

        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (
            (doc_idx + 1)
            % CHECKPOINT_INTERVAL
            == 0
        ):

            torch.save(
                results,
                checkpoint_path,
            )

            total_parsed = (
                len(results)
            )

            total_docs = (
                len(docs)
            )

            progress_pct = (
                total_parsed
                / total_docs
                * 100
            )

            print()

            print(
                f"  Checkpoint saved: "
                f"{total_parsed:,}/"
                f"{total_docs:,} "
                f"articles parsed "
                f"({progress_pct:.2f}%)"
            )

    # ========================================================
    # FINAL SAVE
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

    print()

    print(
        f"  Saved documents : "
        f"{len(results):,}"
    )

    print(
        f"  Total sentences : "
        f"{total_sentences:,}"
    )

    print(
        f"  Failed sentences: "
        f"{failed_sentences:,}"
    )

    print(
        f"  Output           : "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Parse Indonesian sentences "
            "into AMR using AMRBART."
        )
    )

    # ========================================================
    # DATASET
    # ========================================================

    ap.add_argument(
        "--dataset",
        choices=[
            "xlsum",
            "liputan6",
        ],
        required=True,
        help=(
            "Dataset name."
        ),
    )

    # ========================================================
    # SPLIT
    # ========================================================

    ap.add_argument(
        "--split",
        default=None,
        help=(
            "Specific split to process. "
            "If omitted, process all "
            "configured splits."
        ),
    )

    # ========================================================
    # CUSTOM INPUT DIRECTORY
    # ========================================================

    ap.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory containing "
            "<split>.jsonl. "
            "If omitted, defaults to "
            "dataset/<dataset>/preprocessed."
        ),
    )

    # ========================================================
    # CUSTOM OUTPUT DIRECTORY
    # ========================================================

    ap.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory where "
            "<split>.pt and checkpoint "
            "will be saved. "
            "If omitted, defaults to "
            "dataset/<dataset>/amr/parsed."
        ),
    )

    # ========================================================
    # DEVICE
    # ========================================================

    ap.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "PyTorch device. "
            "Example: cuda:0 or cpu."
        ),
    )

    # ========================================================
    # BATCH SIZE
    # ========================================================

    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Number of sentences parsed "
            "at once."
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    ap.add_argument(
        "--model_name",
        default="mbart-en-id-smaller-fted",
        help=(
            "AMRBART model directory name."
        ),
    )

    args = ap.parse_args()

    # ========================================================
    # VALIDATE BATCH SIZE
    # ========================================================

    if args.batch_size <= 0:

        raise ValueError(
            "--batch_size must be > 0"
        )

    # ========================================================
    # DATASET CONFIGURATION
    # ========================================================

    cfg = (
        DATASET_CONFIG[
            args.dataset
        ]
    )

    # Validate manually supplied split

    if (
        args.split is not None
        and args.split not in cfg["splits"]
    ):

        raise ValueError(
            f"Invalid split "
            f"'{args.split}' "
            f"for dataset "
            f"'{args.dataset}'. "
            f"Available splits: "
            f"{cfg['splits']}"
        )

    splits = (
        [args.split]
        if args.split
        else cfg["splits"]
    )

    # ========================================================
    # DATASET ROOT
    # ========================================================

    dataset_dir = (
        BASE_DIR
        / "dataset"
        / args.dataset
    )

    # ========================================================
    # INPUT DIRECTORY
    # ========================================================

    if args.input_dir:

        input_dir = Path(
            args.input_dir
        )

        # Relative paths are resolved relative
        # to project root.

        if not input_dir.is_absolute():

            input_dir = (
                BASE_DIR
                / input_dir
            )

    else:

        input_dir = (
            dataset_dir
            / "preprocessed"
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

        if not output_dir.is_absolute():

            output_dir = (
                BASE_DIR
                / output_dir
            )

    else:

        output_dir = (
            dataset_dir
            / "amr"
            / "parsed"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SHOW CONFIGURATION
    # ========================================================

    print()

    print("=" * 80)
    print("AMR PARSING")
    print("=" * 80)

    print(
        f"AMR repository : "
        f"{AMR_REPO_DIR}"
    )

    print(
        f"AMRBART root   : "
        f"{AMRBART_DIR}"
    )

    print(
        f"Model path     : "
        f"{AMRBART_DIR / 'models' / args.model_name}"
    )

    print(
        f"Model          : "
        f"{args.model_name}"
    )

    print(
        f"Device         : "
        f"{args.device}"
    )

    print(
        f"Batch size     : "
        f"{args.batch_size}"
    )

    print(
        f"Input dir      : "
        f"{input_dir}"
    )

    print(
        f"Output dir     : "
        f"{output_dir}"
    )

    print(
        f"Splits         : "
        f"{', '.join(splits)}"
    )

    # ========================================================
    # VALIDATE INPUT DIRECTORY
    # ========================================================

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory "
            f"does not exist:\n"
            f"{input_dir}"
        )

    # ========================================================
    # LOAD AMR PARSER
    # ========================================================

    print()
    print("Loading AMR parser...")

    parser = TextToAMR(
        root_dir=str(
            AMRBART_DIR
        ),
        model_name=args.model_name,
        use_prefix=True,
    )

    parser.model = (
        parser.model.to(
            args.device
        )
    )

    parser.model.eval()

    print(
        f"Parser loaded on "
        f"{args.device}"
    )

    # ========================================================
    # PROCESS SPLITS
    # ========================================================

    for split in splits:

        input_path = (
            input_dir
            / f"{split}.jsonl"
        )

        output_path = (
            output_dir
            / f"{split}.pt"
        )

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not input_path.exists():

            print()

            print(
                f"Skipping {split}: "
                f"{input_path} "
                f"not found"
            )

            continue

        # ----------------------------------------------------
        # Parse
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print(
            f"=== {split.upper()} ==="
        )
        print("=" * 80)

        parse_split(
            parser=parser,
            input_path=input_path,
            output_path=output_path,
            batch_size=args.batch_size,
        )

    print()

    print("=" * 80)
    print("FINISHED")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()