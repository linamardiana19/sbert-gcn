"""
Run factuality-based artifact filtering for SBERT+GCN summarization.

This script filters train labels, embeddings, and graphs based on factuality
metadata while preserving evaluation splits unchanged.

Supported datasets:
- liputan6: train/dev/test
- xlsum: train/val/test

Example:
    uv run python scripts/run_factuality_filter.py
        --dataset liputan6
        --label_name best_combine_rouge1_tie_rouge2
        --encoder mpnet
        --graph_name hybrid_tadaptive
        --factuality_file dataset/faktualitas/liputan6/liputan6_faktualitas.xlsx
        --factuality_value 1
        --artifact all
        --split train 
        --strict
"""


import argparse
import json
import math
import numbers
import shutil
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import torch


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from summarizer.encoder import ENCODER_CONFIGS


DATASET_CONFIG = {
    "xlsum": {
        "splits": ["train", "val", "test"],
        "train_split": "train",
    },
    "liputan6": {
        "splits": ["train", "dev", "test"],
        "train_split": "train",
    },
}


SUPPORTED_FACTUALITY_EXTENSIONS = {
    ".xlsx",
    ".xls",
    ".csv",
}


# ============================================================
# ID NORMALIZATION
# ============================================================

def normalize_id(value) -> str:
    """
    Normalize document ID safely.

    Compatible with:
        100000
        100000.0
        "100000"
        "majalah-39638082"
        "berita-politik-001"

    Important:
    - String IDs remain case-sensitive.
    - Hyphens are preserved.
    - Underscores are preserved.
    - Internal spaces are preserved.
    - Only surrounding whitespace is removed.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    # Integer numeric ID
    if isinstance(value, numbers.Integral):
        return str(int(value))

    # Float numeric ID
    if isinstance(value, numbers.Real):
        numeric_value = float(value)

        if (
            math.isfinite(numeric_value)
            and numeric_value.is_integer()
        ):
            return str(int(numeric_value))

        return unicodedata.normalize(
            "NFC",
            str(value).strip(),
        )

    # String ID
    return unicodedata.normalize(
        "NFC",
        str(value).strip(),
    )


def normalize_value(value) -> str:
    """
    Normalize categorical scalar values such as factuality.

    Examples:
        1       -> "1"
        1.0     -> "1"
        "1"     -> "1"
        "1.0"   -> "1"

        0       -> "0"
        0.0     -> "0"
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    try:
        numeric_value = float(text)

        if (
            math.isfinite(numeric_value)
            and numeric_value.is_integer()
        ):
            return str(int(numeric_value))

    except (ValueError, TypeError):
        pass

    return text.casefold()


# ============================================================
# FACTUALITY FILE UTILITIES
# ============================================================

def load_factuality_file(
    path: Path,
) -> pd.DataFrame:
    """
    Read factuality metadata safely.

    All columns are read as strings to avoid unwanted
    type coercion for document IDs.

    Supported:
        .csv
        .xlsx
        .xls
    """

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(
            path,
            dtype=str,
            keep_default_na=False,
        )

    raise ValueError(
        f"Unsupported factuality file extension: {suffix}\n"
        f"Supported extensions: "
        f"{sorted(SUPPORTED_FACTUALITY_EXTENSIONS)}"
    )


def find_factuality_file(
    dataset: str,
) -> Path:
    """
    Automatically find factuality file from:

        dataset/faktualitas/<dataset>/

    Exactly one CSV/Excel factuality file must exist.
    """

    factuality_dir = (
        BASE_DIR
        / "dataset"
        / "faktualitas"
        / dataset
    )

    if not factuality_dir.exists():
        raise FileNotFoundError(
            "Factuality directory not found:\n"
            f"  {factuality_dir}"
        )

    candidates = sorted(
        path
        for path in factuality_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_FACTUALITY_EXTENSIONS
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No factuality Excel/CSV file found in:\n"
            f"  {factuality_dir}"
        )

    if len(candidates) > 1:

        print(
            "\nMultiple factuality files found:"
        )

        for path in candidates:
            print(f"  - {path}")

        raise RuntimeError(
            "\nMultiple factuality files detected. "
            "Please specify the intended file explicitly "
            "using --factuality_file."
        )

    return candidates[0]


# ============================================================
# FACTUAL TRAIN ID SELECTION
# ============================================================

def get_factual_train_ids(
    factuality_path: Path,
    id_column: str,
    split_column: str,
    factuality_column: str,
    factuality_value: str,
    train_split: str = "train",
    strict: bool = False,
) -> tuple[list[str], set[str]]:
    """
    Select IDs where:

        split == train_split
        AND
        factuality == factuality_value

    Returns:
        selected_id_order:
            Ordered unique IDs following factuality file order.

        selected_id_set:
            Same IDs as a set for efficient lookup.
    """

    df = load_factuality_file(
        factuality_path
    )

    required_columns = {
        id_column,
        split_column,
        factuality_column,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns in factuality file:\n"
            f"  Missing   : {sorted(missing_columns)}\n"
            f"  Available : {list(df.columns)}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Normalize important columns
    # --------------------------------------------------------

    df["_normalized_id"] = (
        df[id_column]
        .map(normalize_id)
    )

    df["_normalized_split"] = (
        df[split_column]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    df["_normalized_factuality"] = (
        df[factuality_column]
        .map(normalize_value)
    )

    # --------------------------------------------------------
    # Validate empty IDs
    # --------------------------------------------------------

    empty_id_count = int(
        (
            df["_normalized_id"] == ""
        ).sum()
    )

    if empty_id_count > 0:
        raise ValueError(
            f"Found {empty_id_count:,} empty/null IDs "
            f"in factuality metadata:\n"
            f"  {factuality_path}"
        )

    # --------------------------------------------------------
    # Print metadata statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FACTUALITY METADATA")
    print("=" * 70)

    print(f"File              : {factuality_path}")
    print(f"Total rows        : {len(df):,}")

    print("\nSplit distribution:")

    print(
        df["_normalized_split"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nFactuality distribution:")

    print(
        df["_normalized_factuality"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Apply factuality selection
    # --------------------------------------------------------

    train_split_normalized = (
        str(train_split)
        .strip()
        .casefold()
    )

    factuality_value_normalized = (
        normalize_value(
            factuality_value
        )
    )

    selected_df = df[
        (
            df["_normalized_split"]
            == train_split_normalized
        )
        &
        (
            df["_normalized_factuality"]
            == factuality_value_normalized
        )
    ].copy()

    if selected_df.empty:
        raise ValueError(
            "No documents matched factuality filtering:\n"
            f"  split      = {train_split!r}\n"
            f"  factuality = {factuality_value!r}"
        )

    # --------------------------------------------------------
    # Detect duplicate selected IDs
    # --------------------------------------------------------

    duplicate_mask = (
        selected_df[
            "_normalized_id"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_ids = (
        selected_df.loc[
            duplicate_mask,
            "_normalized_id",
        ]
        .unique()
        .tolist()
    )

    if duplicate_ids:

        print(
            "\nWARNING: Duplicate selected factuality IDs found."
        )

        print(
            f"Duplicate unique IDs: "
            f"{len(duplicate_ids):,}"
        )

        print("Examples:")

        for doc_id in duplicate_ids[:10]:
            print(f"  - {doc_id}")

        if strict:
            raise ValueError(
                f"Found {len(duplicate_ids):,} duplicate "
                "selected factuality IDs."
            )

    # --------------------------------------------------------
    # Preserve factuality file order while removing duplicates
    # --------------------------------------------------------

    selected_id_order = list(
        dict.fromkeys(
            selected_df[
                "_normalized_id"
            ].tolist()
        )
    )

    selected_id_set = set(
        selected_id_order
    )

    # --------------------------------------------------------
    # Print selection statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED FACTUAL TRAIN DATA")
    print("=" * 70)

    print(f"Split             : {train_split}")
    print(f"Factuality value  : {factuality_value}")
    print(f"Selected rows     : {len(selected_df):,}")
    print(f"Unique IDs        : {len(selected_id_set):,}")
    print(f"Duplicate IDs     : {len(duplicate_ids):,}")

    print("\nExample selected IDs:")

    for doc_id in selected_id_order[:10]:
        print(
            f"  - {doc_id!r}"
        )

    return (
        selected_id_order,
        selected_id_set,
    )


# ============================================================
# JSONL UTILITIES
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:
    """
    Read JSONL file into list[dict].
    """

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON:\n"
                    f"  File : {path}\n"
                    f"  Line : {line_number}\n"
                    f"  Error: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise TypeError(
                    f"Expected JSON object/dict in:\n"
                    f"  File : {path}\n"
                    f"  Line : {line_number}\n"
                    f"  Type : {type(row)}"
                )

            rows.append(row)

    return rows


def save_jsonl(
    data: list[dict],
    path: Path,
) -> None:
    """
    Save list[dict] as JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in data:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# GENERIC RECORD FILTERING
# ============================================================

def filter_records_by_selected_ids(
    records: list[dict],
    selected_id_order: list[str],
    selected_id_set: set[str],
    artifact_name: str,
    id_field: str = "id",
    strict: bool = False,
) -> tuple[list[dict], dict]:
    """
    Filter list[dict] based on selected factuality IDs.

    Important:
    Output order follows selected_id_order from the factuality file.

    This guarantees consistent train order between:
        labels
        embeddings
        graphs
    """

    record_map: dict[str, dict] = {}

    duplicate_ids: set[str] = set()

    for index, item in enumerate(
        records
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"{artifact_name} item at index {index} "
                f"is not a dict.\n"
                f"Actual type: {type(item)}"
            )

        if id_field not in item:
            raise KeyError(
                f"Field {id_field!r} not found in "
                f"{artifact_name} item at index {index}.\n"
                f"Available fields: {list(item.keys())}"
            )

        doc_id = normalize_id(
            item[id_field]
        )

        if not doc_id:
            raise ValueError(
                f"Empty ID found in {artifact_name} "
                f"at item index {index}."
            )

        if doc_id in record_map:

            duplicate_ids.add(
                doc_id
            )

            # Keep first occurrence in non-strict mode.
            continue

        record_map[
            doc_id
        ] = item

    source_id_set = set(
        record_map.keys()
    )

    missing_ids = (
        selected_id_set
        - source_id_set
    )

    excluded_source_ids = (
        source_id_set
        - selected_id_set
    )

    if duplicate_ids:

        print(
            f"\nWARNING: Duplicate IDs found "
            f"in {artifact_name}."
        )

        print(
            f"Duplicate unique IDs: "
            f"{len(duplicate_ids):,}"
        )

        print("Examples:")

        for doc_id in sorted(
            duplicate_ids
        )[:10]:
            print(f"  - {doc_id}")

        if strict:
            raise ValueError(
                f"Found {len(duplicate_ids):,} "
                f"duplicate IDs in {artifact_name}."
            )

    # --------------------------------------------------------
    # Build output in factuality reference order
    # --------------------------------------------------------

    filtered_records = [
        record_map[doc_id]
        for doc_id in selected_id_order
        if doc_id in record_map
    ]

    filtered_ids = [
        normalize_id(
            item[id_field]
        )
        for item in filtered_records
    ]

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        f"\n=== Train {artifact_name} Filtering ==="
    )

    print(
        f"Original documents   : {len(records):,}"
    )

    print(
        f"Original unique ID   : {len(source_id_set):,}"
    )

    print(
        f"Selected fact IDs    : {len(selected_id_set):,}"
    )

    print(
        f"Matched documents    : {len(filtered_records):,}"
    )

    print(
        f"Matched unique ID    : {len(set(filtered_ids)):,}"
    )

    print(
        f"Missing fact IDs     : {len(missing_ids):,}"
    )

    print(
        f"Excluded documents   : {len(excluded_source_ids):,}"
    )

    print(
        f"Duplicate source ID  : {len(duplicate_ids):,}"
    )

    # --------------------------------------------------------
    # Missing IDs
    # --------------------------------------------------------

    if missing_ids:

        print(
            f"\nWARNING: Selected factuality IDs "
            f"not found in train {artifact_name.lower()}."
        )

        print("Examples:")

        for doc_id in sorted(
            missing_ids
        )[:10]:
            print(f"  - {doc_id}")

        if strict:
            raise ValueError(
                f"{len(missing_ids):,} factuality IDs "
                f"were not found in "
                f"train {artifact_name.lower()}."
            )

    if not filtered_records:
        raise ValueError(
            f"{artifact_name} filtering "
            "produced zero train documents."
        )

    stats = {
        "original_count": len(
            records
        ),
        "original_unique_count": len(
            source_id_set
        ),
        "selected_factuality_count": len(
            selected_id_set
        ),
        "matched_count": len(
            filtered_records
        ),
        "matched_unique_count": len(
            set(filtered_ids)
        ),
        "missing_count": len(
            missing_ids
        ),
        "excluded_count": len(
            excluded_source_ids
        ),
        "duplicate_source_count": len(
            duplicate_ids
        ),
    }

    return (
        filtered_records,
        stats,
    )


# ============================================================
# LABEL FILTERING
# ============================================================

def filter_train_labels(
    input_path: Path,
    output_path: Path,
    selected_id_order: list[str],
    selected_id_set: set[str],
    id_field: str = "id",
    strict: bool = False,
) -> dict:
    """
    Filter train JSONL labels.
    """

    print(f"\nSource:")
    print(f"  {input_path}")

    print(f"Target:")
    print(f"  {output_path}")

    records = read_jsonl(
        input_path
    )

    (
        filtered_records,
        stats,
    ) = filter_records_by_selected_ids(
        records=records,
        selected_id_order=selected_id_order,
        selected_id_set=selected_id_set,
        artifact_name="Labels",
        id_field=id_field,
        strict=strict,
    )

    save_jsonl(
        filtered_records,
        output_path,
    )

    print(
        f"\nSaved {len(filtered_records):,} "
        f"filtered label documents to:"
    )

    print(
        f"  {output_path}"
    )

    return stats


# ============================================================
# PT ARTIFACT FILTERING
# ============================================================

def filter_train_pt_artifact(
    input_path: Path,
    output_path: Path,
    selected_id_order: list[str],
    selected_id_set: set[str],
    artifact_name: str,
    id_field: str = "id",
    strict: bool = False,
) -> dict:
    """
    Filter generic .pt artifact containing list[dict].

    Compatible with embeddings:

        [
            {
                "id": "...",
                "embeddings": Tensor(...)
            }
        ]

    Compatible with graphs:

        [
            {
                "id": "...",
                "edge_index": Tensor(...),
                "edge_weight": Tensor(...)
            }
        ]
    """

    print(f"\nSource:")
    print(f"  {input_path}")

    print(f"Target:")
    print(f"  {output_path}")

    print(
        f"\nLoading train "
        f"{artifact_name.lower()}..."
    )

    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        data,
        list,
    ):
        raise TypeError(
            f"Expected {artifact_name} file "
            f"to contain list[dict].\n"
            f"File       : {input_path}\n"
            f"Actual type: {type(data)}"
        )

    (
        filtered_data,
        stats,
    ) = filter_records_by_selected_ids(
        records=data,
        selected_id_order=selected_id_order,
        selected_id_set=selected_id_set,
        artifact_name=artifact_name,
        id_field=id_field,
        strict=strict,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        filtered_data,
        output_path,
    )

    print(
        f"\nSaved {len(filtered_data):,} "
        f"filtered {artifact_name.lower()} to:"
    )

    print(
        f"  {output_path}"
    )

    return stats


# ============================================================
# EVALUATION ARTIFACT COPYING
# ============================================================

def copy_file_unchanged(
    input_path: Path,
    output_path: Path,
    artifact_name: str,
) -> None:
    """
    Copy dev/val/test unchanged.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        input_path,
        output_path,
    )

    print(
        f"\nCopied {artifact_name} unchanged:"
    )

    print(
        f"  Source : {input_path}"
    )

    print(
        f"  Target : {output_path}"
    )


# ============================================================
# ARTIFACT ID EXTRACTION
# ============================================================

def get_label_id_sequence(
    path: Path,
    id_field: str = "id",
) -> list[str]:
    """
    Get ordered IDs from JSONL labels.
    """

    rows = read_jsonl(
        path
    )

    return [
        normalize_id(
            row[id_field]
        )
        for row in rows
    ]


def get_pt_id_sequence(
    path: Path,
    id_field: str = "id",
) -> list[str]:
    """
    Get ordered IDs from .pt artifact.
    """

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        data,
        list,
    ):
        raise TypeError(
            f"Expected list[dict] in:\n"
            f"  {path}\n"
            f"Actual type: {type(data)}"
        )

    return [
        normalize_id(
            item[id_field]
        )
        for item in data
    ]


# ============================================================
# CROSS-ARTIFACT ALIGNMENT VALIDATION
# ============================================================

def validate_train_alignment(
    selected_id_order: list[str],
    label_path: Path,
    embedding_path: Path,
    graph_path: Path,
    label_id_field: str = "id",
    strict: bool = False,
) -> bool:
    """
    Validate both:

    1. ID set equality

        factuality IDs
        ==
        label IDs
        ==
        embedding IDs
        ==
        graph IDs

    2. ID order equality

        factuality_order
        ==
        label_order
        ==
        embedding_order
        ==
        graph_order
    """

    label_ids = get_label_id_sequence(
        label_path,
        id_field=label_id_field,
    )

    embedding_ids = get_pt_id_sequence(
        embedding_path
    )

    graph_ids = get_pt_id_sequence(
        graph_path
    )

    factuality_ids = list(
        selected_id_order
    )

    factuality_set = set(
        factuality_ids
    )

    label_set = set(
        label_ids
    )

    embedding_set = set(
        embedding_ids
    )

    graph_set = set(
        graph_ids
    )

    print("\n" + "=" * 70)
    print("TRAIN CROSS-ARTIFACT ALIGNMENT VALIDATION")
    print("=" * 70)

    print(
        f"Selected factuality IDs : "
        f"{len(factuality_ids):,}"
    )

    print(
        f"Label IDs               : "
        f"{len(label_ids):,}"
    )

    print(
        f"Embedding IDs           : "
        f"{len(embedding_ids):,}"
    )

    print(
        f"Graph IDs               : "
        f"{len(graph_ids):,}"
    )

    # --------------------------------------------------------
    # Set-level validation
    # --------------------------------------------------------

    set_comparisons = {
        "factuality_vs_labels":
            factuality_set ^ label_set,

        "factuality_vs_embeddings":
            factuality_set ^ embedding_set,

        "factuality_vs_graphs":
            factuality_set ^ graph_set,

        "labels_vs_embeddings":
            label_set ^ embedding_set,

        "labels_vs_graphs":
            label_set ^ graph_set,

        "embeddings_vs_graphs":
            embedding_set ^ graph_set,
    }

    print("\nID set differences:")

    set_aligned = True

    for (
        comparison_name,
        difference,
    ) in set_comparisons.items():

        print(
            f"  {comparison_name:<28}: "
            f"{len(difference):,}"
        )

        if difference:
            set_aligned = False

    # --------------------------------------------------------
    # Order-level validation
    # --------------------------------------------------------

    order_comparisons = {
        "factuality_vs_labels":
            factuality_ids == label_ids,

        "factuality_vs_embeddings":
            factuality_ids == embedding_ids,

        "factuality_vs_graphs":
            factuality_ids == graph_ids,

        "labels_vs_embeddings":
            label_ids == embedding_ids,

        "labels_vs_graphs":
            label_ids == graph_ids,

        "embeddings_vs_graphs":
            embedding_ids == graph_ids,
    }

    print("\nID order equality:")

    order_aligned = True

    for (
        comparison_name,
        is_equal,
    ) in order_comparisons.items():

        status = (
            "OK"
            if is_equal
            else "MISMATCH"
        )

        print(
            f"  {comparison_name:<28}: "
            f"{status}"
        )

        if not is_equal:
            order_aligned = False

    # --------------------------------------------------------
    # Final alignment result
    # --------------------------------------------------------

    all_aligned = (
        set_aligned
        and order_aligned
    )

    if all_aligned:

        print("\nAlignment status: OK")

        print(
            "Factuality IDs, labels, embeddings, "
            "and graphs contain identical train IDs "
            "in identical order."
        )

    else:

        print(
            "\nAlignment status: MISMATCH"
        )

        # Print set difference examples
        for (
            comparison_name,
            difference,
        ) in set_comparisons.items():

            if not difference:
                continue

            print(
                f"\nExamples for "
                f"{comparison_name}:"
            )

            for doc_id in sorted(
                difference
            )[:10]:
                print(f"  - {doc_id}")

        # Find first order mismatch examples
        sequences = {
            "factuality": factuality_ids,
            "labels": label_ids,
            "embeddings": embedding_ids,
            "graphs": graph_ids,
        }

        minimum_length = min(
            len(sequence)
            for sequence in sequences.values()
        )

        mismatch_indexes = []

        for index in range(
            minimum_length
        ):

            values = {
                name: sequence[index]
                for (
                    name,
                    sequence,
                ) in sequences.items()
            }

            if len(
                set(values.values())
            ) > 1:

                mismatch_indexes.append(
                    (
                        index,
                        values,
                    )
                )

            if len(
                mismatch_indexes
            ) >= 5:
                break

        if mismatch_indexes:

            print(
                "\nFirst order mismatch examples:"
            )

            for (
                index,
                values,
            ) in mismatch_indexes:

                print(
                    f"\n  Index {index}:"
                )

                for (
                    name,
                    value,
                ) in values.items():

                    print(
                        f"    {name:<11}: "
                        f"{value!r}"
                    )

        if strict:
            raise ValueError(
                "Train factuality IDs, labels, "
                "embeddings, and graphs "
                "are not fully aligned."
            )

    return all_aligned


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Filter SBERT+GCN labels, embeddings, "
            "and graphs using factuality metadata."
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ap.add_argument(
        "--dataset",
        choices=[
            "xlsum",
            "liputan6",
        ],
        required=True,
        help=(
            "Dataset name. "
            "Liputan6 uses train/dev/test; "
            "XL-Sum uses train/val/test."
        ),
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ap.add_argument(
        "--label_name",
        required=True,
        help=(
            "Label directory name. Example: "
            "best_combine_rouge1_tie_rouge2"
        ),
    )

    ap.add_argument(
        "--label_id_field",
        default="id",
        help=(
            "ID field inside label JSONL. "
            "Default: id"
        ),
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    ap.add_argument(
        "--encoder",
        default="mpnet",
        choices=list(
            ENCODER_CONFIGS.keys()
        ),
        help=(
            "Encoder preset. Default: mpnet"
        ),
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    ap.add_argument(
        "--graph_name",
        default=None,
        help=(
            "Graph directory name inside "
            "sbert_<encoder>/graphs/. "
            "Examples: hybrid_tadaptive, "
            "tadaptive, tfidf, "
            "hybrid_intersect_tadaptive. "
            "Required for --artifact all or graphs."
        ),
    )

    # --------------------------------------------------------
    # Factuality metadata
    # --------------------------------------------------------

    ap.add_argument(
        "--factuality_file",
        default=None,
        help=(
            "Explicit factuality .xlsx/.xls/.csv path. "
            "If omitted, auto-detect from "
            "dataset/faktualitas/<dataset>/"
        ),
    )

    ap.add_argument(
        "--factuality_value",
        default="1",
        help=(
            "Factuality value to retain. "
            "Default: 1"
        ),
    )

    ap.add_argument(
        "--id_column",
        default="id",
        help=(
            "ID column in factuality file. "
            "Default: id"
        ),
    )

    ap.add_argument(
        "--split_column",
        default="split",
        help=(
            "Split column in factuality file. "
            "Default: split"
        ),
    )

    ap.add_argument(
        "--factuality_column",
        default="factuality",
        help=(
            "Factuality column in metadata. "
            "Default: factuality"
        ),
    )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    ap.add_argument(
        "--split",
        default=None,
        help=(
            "Specific split to process. "
            "If omitted, process all dataset splits."
        ),
    )

    ap.add_argument(
        "--artifact",
        choices=[
            "all",
            "labels",
            "embeddings",
            "graphs",
        ],
        default="all",
        help=(
            "Artifact type to process. "
            "Default: all"
        ),
    )

    ap.add_argument(
        "--copy_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy dev/val/test unchanged. "
            "Default: true."
        ),
    )

    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Raise error when missing IDs, "
            "duplicate IDs, or alignment "
            "mismatches are detected."
        ),
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Validate graph_name
    # --------------------------------------------------------

    if (
        args.artifact
        in {
            "all",
            "graphs",
        }
        and not args.graph_name
    ):
        ap.error(
            "--graph_name is required when "
            "--artifact is 'all' or 'graphs'."
        )

    # --------------------------------------------------------
    # Resolve dataset config
    # --------------------------------------------------------

    cfg = DATASET_CONFIG[
        args.dataset
    ]

    enc_cfg = ENCODER_CONFIGS[
        args.encoder
    ]

    train_split = cfg[
        "train_split"
    ]

    # --------------------------------------------------------
    # Resolve splits
    # --------------------------------------------------------

    if args.split is not None:

        if args.split not in cfg["splits"]:

            ap.error(
                f"Invalid split {args.split!r} "
                f"for dataset {args.dataset!r}. "
                f"Available splits: "
                f"{cfg['splits']}"
            )

        splits = [
            args.split
        ]

    else:

        splits = list(
            cfg["splits"]
        )

    sbert_subdir = enc_cfg[
        "sbert_subdir"
    ]

    # ========================================================
    # SOURCE DIRECTORIES
    # ========================================================

    source_dataset_dir = (
        BASE_DIR
        / "dataset"
        / args.dataset
    )

    source_label_dir = (
        source_dataset_dir
        / "labels"
        / args.label_name
    )

    source_embedding_dir = (
        source_dataset_dir
        / sbert_subdir
        / "embeddings"
    )

    source_graph_dir = None

    if args.graph_name:

        source_graph_dir = (
            source_dataset_dir
            / sbert_subdir
            / "graphs"
            / args.graph_name
        )

    # ========================================================
    # TARGET DIRECTORIES
    # ========================================================

    factuality_dataset_dir = (
        BASE_DIR
        / "dataset"
        / "faktualitas"
        / args.dataset
    )

    target_label_dir = (
        factuality_dataset_dir
        / "labels"
        / args.label_name
    )

    target_embedding_dir = (
        factuality_dataset_dir
        / sbert_subdir
        / "embeddings"
    )

    target_graph_dir = None

    if args.graph_name:

        target_graph_dir = (
            factuality_dataset_dir
            / sbert_subdir
            / "graphs"
            / args.graph_name
        )

    # ========================================================
    # RESOLVE FACTUALITY FILE
    # ========================================================

    if args.factuality_file:

        factuality_path = Path(
            args.factuality_file
        )

        if not factuality_path.is_absolute():

            factuality_path = (
                BASE_DIR
                / factuality_path
            )

    else:

        factuality_path = (
            find_factuality_file(
                args.dataset
            )
        )

    if not factuality_path.exists():
        raise FileNotFoundError(
            "Factuality file not found:\n"
            f"  {factuality_path}"
        )

    # ========================================================
    # VALIDATE REQUIRED SOURCE DIRECTORIES
    # ========================================================

    if (
        args.artifact
        in {
            "all",
            "labels",
        }
        and not source_label_dir.exists()
    ):
        raise FileNotFoundError(
            "Source label directory not found:\n"
            f"  {source_label_dir}"
        )

    if (
        args.artifact
        in {
            "all",
            "embeddings",
        }
        and not source_embedding_dir.exists()
    ):
        raise FileNotFoundError(
            "Source embedding directory not found:\n"
            f"  {source_embedding_dir}"
        )

    if (
        args.artifact
        in {
            "all",
            "graphs",
        }
        and (
            source_graph_dir is None
            or not source_graph_dir.exists()
        )
    ):
        raise FileNotFoundError(
            "Source graph directory not found:\n"
            f"  {source_graph_dir}"
        )

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print("\n" + "=" * 70)
    print("SBERT+GCN FACTUALITY ARTIFACT FILTER")
    print("=" * 70)

    print(f"Dataset           : {args.dataset}")
    print(f"Splits            : {splits}")
    print(f"Label name        : {args.label_name}")
    print(f"Encoder           : {args.encoder}")
    print(f"SBERT subdir      : {sbert_subdir}")
    print(f"Graph name        : {args.graph_name}")
    print(f"Factuality file   : {factuality_path}")
    print(f"Factuality value  : {args.factuality_value}")
    print(f"Artifact          : {args.artifact}")
    print(f"Strict validation : {args.strict}")

    # ========================================================
    # SELECT FACTUAL TRAIN IDS
    # ========================================================

    (
        selected_id_order,
        selected_id_set,
    ) = get_factual_train_ids(
        factuality_path=factuality_path,
        id_column=args.id_column,
        split_column=args.split_column,
        factuality_column=args.factuality_column,
        factuality_value=args.factuality_value,
        train_split=train_split,
        strict=args.strict,
    )

    summary = {}

    # ========================================================
    # PROCESS LABELS
    # ========================================================

    if args.artifact in {
        "all",
        "labels",
    }:

        print("\n" + "=" * 70)
        print("PROCESSING LABELS")
        print("=" * 70)

        label_summary = {}

        for split in splits:

            input_path = (
                source_label_dir
                / f"{split}.jsonl"
            )

            output_path = (
                target_label_dir
                / f"{split}.jsonl"
            )

            if not input_path.exists():

                print(
                    f"\nSkipping label split {split}:"
                )

                print(
                    f"  {input_path} not found"
                )

                continue

            print(
                f"\n--- Labels: {split} ---"
            )

            if split == train_split:

                label_summary[
                    split
                ] = filter_train_labels(
                    input_path=input_path,
                    output_path=output_path,
                    selected_id_order=selected_id_order,
                    selected_id_set=selected_id_set,
                    id_field=args.label_id_field,
                    strict=args.strict,
                )

            elif args.copy_eval:

                copy_file_unchanged(
                    input_path=input_path,
                    output_path=output_path,
                    artifact_name=(
                        f"{split} labels"
                    ),
                )

                label_summary[
                    split
                ] = {
                    "status":
                        "copied_unchanged"
                }

        summary[
            "labels"
        ] = label_summary

    # ========================================================
    # PROCESS EMBEDDINGS
    # ========================================================

    if args.artifact in {
        "all",
        "embeddings",
    }:

        print("\n" + "=" * 70)
        print("PROCESSING EMBEDDINGS")
        print("=" * 70)

        embedding_summary = {}

        for split in splits:

            input_path = (
                source_embedding_dir
                / f"{split}.pt"
            )

            output_path = (
                target_embedding_dir
                / f"{split}.pt"
            )

            if not input_path.exists():

                print(
                    f"\nSkipping embedding split {split}:"
                )

                print(
                    f"  {input_path} not found"
                )

                continue

            print(
                f"\n--- Embeddings: {split} ---"
            )

            if split == train_split:

                embedding_summary[
                    split
                ] = filter_train_pt_artifact(
                    input_path=input_path,
                    output_path=output_path,
                    selected_id_order=selected_id_order,
                    selected_id_set=selected_id_set,
                    artifact_name="Embeddings",
                    id_field="id",
                    strict=args.strict,
                )

            elif args.copy_eval:

                copy_file_unchanged(
                    input_path=input_path,
                    output_path=output_path,
                    artifact_name=(
                        f"{split} embeddings"
                    ),
                )

                embedding_summary[
                    split
                ] = {
                    "status":
                        "copied_unchanged"
                }

        summary[
            "embeddings"
        ] = embedding_summary

    # ========================================================
    # PROCESS GRAPHS
    # ========================================================

    if args.artifact in {
        "all",
        "graphs",
    }:

        print("\n" + "=" * 70)
        print("PROCESSING GRAPHS")
        print("=" * 70)

        graph_summary = {}

        for split in splits:

            input_path = (
                source_graph_dir
                / f"{split}.pt"
            )

            output_path = (
                target_graph_dir
                / f"{split}.pt"
            )

            if not input_path.exists():

                print(
                    f"\nSkipping graph split {split}:"
                )

                print(
                    f"  {input_path} not found"
                )

                continue

            print(
                f"\n--- Graphs: {split} ---"
            )

            if split == train_split:

                graph_summary[
                    split
                ] = filter_train_pt_artifact(
                    input_path=input_path,
                    output_path=output_path,
                    selected_id_order=selected_id_order,
                    selected_id_set=selected_id_set,
                    artifact_name="Graphs",
                    id_field="id",
                    strict=args.strict,
                )

            elif args.copy_eval:

                copy_file_unchanged(
                    input_path=input_path,
                    output_path=output_path,
                    artifact_name=(
                        f"{split} graphs"
                    ),
                )

                graph_summary[
                    split
                ] = {
                    "status":
                        "copied_unchanged"
                }

        summary[
            "graphs"
        ] = graph_summary

    # ========================================================
    # CROSS-ARTIFACT VALIDATION
    # ========================================================

    if (
        args.artifact == "all"
        and train_split in splits
    ):

        train_label_path = (
            target_label_dir
            / f"{train_split}.jsonl"
        )

        train_embedding_path = (
            target_embedding_dir
            / f"{train_split}.pt"
        )

        train_graph_path = (
            target_graph_dir
            / f"{train_split}.pt"
        )

        required_train_outputs = [
            train_label_path,
            train_embedding_path,
            train_graph_path,
        ]

        missing_outputs = [
            path
            for path
            in required_train_outputs
            if not path.exists()
        ]

        if missing_outputs:

            print(
                "\nCross-artifact validation skipped "
                "because output files are missing:"
            )

            for path in missing_outputs:
                print(f"  - {path}")

            if args.strict:
                raise FileNotFoundError(
                    "Cross-artifact validation could "
                    "not run because one or more "
                    "output train artifacts are missing."
                )

        else:

            alignment_ok = (
                validate_train_alignment(
                    selected_id_order=selected_id_order,
                    label_path=train_label_path,
                    embedding_path=train_embedding_path,
                    graph_path=train_graph_path,
                    label_id_field=args.label_id_field,
                    strict=args.strict,
                )
            )

            summary[
                "train_alignment"
            ] = {
                "status": (
                    "ok"
                    if alignment_ok
                    else "mismatch"
                )
            }

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nFactuality artifact filtering completed."
    )


if __name__ == "__main__":
    main()