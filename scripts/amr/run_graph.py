"""
AMR Sentence Graph Construction
===============================

Purpose
-------
Convert pairwise AMR sentence similarities into sentence-level graphs
for the GCN summarization model.

Input
-----
dataset/<dataset>/amr/similarity/<method>/<split>.pt

Example:
dataset/xlsum/amr/similarity/smatch/val.pt

Expected input document:
{
    "id": "...",
    "num_nodes": N,
    "pairs": [
        (src, dst, similarity_score),
        ...
    ],
    "num_pairs": ...,
    "errors": [...]
}

Only unique pairs i < j are stored by run_similarity.py.

Output
------
dataset/<dataset>/amr/graphs/<method>_t<threshold>/<split>.pt

Example:

Adaptive:
dataset/xlsum/amr/graphs/smatch_tadaptive/val.pt

Graph format:
{
    "id": "...",
    "edge_index": Tensor[2, E],
    "edge_weight": Tensor[E]
}

Adaptive Threshold
------------------
For each document, use all valid pairwise similarities i < j:

    mean = mean(pairwise similarity)
    std  = population standard deviation

Then:

    threshold = max(mean - std, 0)

An edge is retained when:

    similarity(i, j) > threshold

Each retained undirected relationship becomes two directed edges:

    i -> j
    j -> i

Both directions use the same SMATCH score as edge weight.

No self-loops are manually added here.

How to Run
----------

Adaptive:

    uv run python scripts/amr/run_graph.py \
        --dataset xlsum \
        --method smatch \
        --threshold adaptive \
        --split val

Fixed threshold 0.3:

    uv run python scripts/amr/run_graph.py \
        --dataset xlsum \
        --method smatch \
        --threshold 0.3 \
        --split val

Fully connected:

    uv run python scripts/amr/run_graph.py \
        --dataset xlsum \
        --method smatch \
        --threshold fully_connected \
        --split val

Changing threshold does NOT require rerunning SMATCH.
"""


import argparse
import sys
from pathlib import Path

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


# ============================================================
# HELPERS
# ============================================================

def expected_pair_count(
    num_nodes: int,
) -> int:
    """
    Number of unique unordered pairs:

        N choose 2
        = N(N-1)/2
    """

    return (
        num_nodes
        * (num_nodes - 1)
        // 2
    )


def get_pair_scores(
    pairs,
) -> torch.Tensor:
    """
    Extract similarity scores from:

        [(i, j, score), ...]

    Returns:
        Tensor[num_pairs]
    """

    if len(pairs) == 0:

        return torch.empty(
            (0,),
            dtype=torch.float32,
        )

    return torch.tensor(
        [
            float(score)
            for _, _, score in pairs
        ],
        dtype=torch.float32,
    )


# ============================================================
# THRESHOLD
# ============================================================

def resolve_threshold(
    pair_scores: torch.Tensor,
    threshold,
):
    """
    Resolve graph threshold.

    adaptive:
        max(mean - population_std, 0)

    fully_connected:
        None

    fixed:
        float(threshold)
    """

    # --------------------------------------------------------
    # FULLY CONNECTED
    # --------------------------------------------------------

    if threshold == "fully_connected":

        return None

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    if threshold == "adaptive":

        if pair_scores.numel() == 0:

            return 0.0

        mean_similarity = (
            pair_scores
            .mean()
            .item()
        )

        # Population std:
        #
        # correction=0
        #
        # Sama dengan np.std(..., ddof=0)
        if pair_scores.numel() > 1:

            std_similarity = (
                pair_scores
                .std(
                    correction=0
                )
                .item()
            )

        else:

            std_similarity = 0.0

        resolved = (
            mean_similarity
            - std_similarity
        )

        return max(
            resolved,
            0.0,
        )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    return float(
        threshold
    )


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph_from_pairs(
    num_nodes: int,
    pairs,
    threshold,
):
    """
    Convert unique pairs (i < j) into PyG-style graph.

    Returns:
        edge_index
        edge_weight
        resolved_threshold
        mean_similarity
        std_similarity
    """

    # ========================================================
    # VALIDATE NUM NODES
    # ========================================================

    if num_nodes < 0:

        raise ValueError(
            f"num_nodes invalid: {num_nodes}"
        )

    # ========================================================
    # SCORES
    # ========================================================

    pair_scores = get_pair_scores(
        pairs
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    if pair_scores.numel() > 0:

        mean_similarity = (
            pair_scores
            .mean()
            .item()
        )

    else:

        mean_similarity = 0.0

    if pair_scores.numel() > 1:

        std_similarity = (
            pair_scores
            .std(
                correction=0
            )
            .item()
        )

    else:

        std_similarity = 0.0

    # ========================================================
    # THRESHOLD
    # ========================================================

    resolved_threshold = (
        resolve_threshold(
            pair_scores,
            threshold,
        )
    )

    # ========================================================
    # BUILD EDGES
    # ========================================================

    src = []
    dst = []
    weights = []

    for i, j, score in pairs:

        i = int(i)
        j = int(j)
        score = float(score)

        # ----------------------------------------------------
        # INDEX VALIDATION
        # ----------------------------------------------------

        if not (
            0 <= i < num_nodes
        ):

            raise ValueError(
                f"Invalid src node {i}; "
                f"num_nodes={num_nodes}"
            )

        if not (
            0 <= j < num_nodes
        ):

            raise ValueError(
                f"Invalid dst node {j}; "
                f"num_nodes={num_nodes}"
            )

        # Similarity input should contain i < j
        if i >= j:

            raise ValueError(
                f"Expected unique pair i < j, "
                f"received ({i}, {j})"
            )

        # ----------------------------------------------------
        # EDGE SELECTION
        # ----------------------------------------------------

        if threshold == "fully_connected":

            keep_edge = True

        else:

            keep_edge = (
                score
                > resolved_threshold
            )

        if not keep_edge:
            continue

        # ----------------------------------------------------
        # UNDIRECTED RELATIONSHIP
        #
        # Represented as two directed PyG edges.
        # ----------------------------------------------------

        src.append(i)
        dst.append(j)
        weights.append(score)

        src.append(j)
        dst.append(i)
        weights.append(score)

    # ========================================================
    # TORCH TENSORS
    # ========================================================

    if len(src) > 0:

        edge_index = torch.tensor(
            [
                src,
                dst,
            ],
            dtype=torch.long,
        )

        edge_weight = torch.tensor(
            weights,
            dtype=torch.float32,
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_weight = torch.empty(
            (0,),
            dtype=torch.float32,
        )

    return (
        edge_index,
        edge_weight,
        resolved_threshold,
        mean_similarity,
        std_similarity,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_similarity_item(
    item,
):
    """
    Validate one document from run_similarity.py.
    """

    required_keys = [
        "id",
        "num_nodes",
        "pairs",
    ]

    for key in required_keys:

        if key not in item:

            raise KeyError(
                f"Missing key '{key}' "
                f"in similarity item."
            )

    doc_id = str(
        item["id"]
    )

    num_nodes = int(
        item["num_nodes"]
    )

    pairs = item[
        "pairs"
    ]

    errors = item.get(
        "errors",
        []
    )

    expected_pairs = (
        expected_pair_count(
            num_nodes
        )
    )

    actual_pairs = len(
        pairs
    )

    num_errors = len(
        errors
    )

    # ========================================================
    # PAIR CONSISTENCY
    # ========================================================

    if (
        actual_pairs
        + num_errors
        != expected_pairs
    ):

        raise RuntimeError(
            f"Pair count inconsistency "
            f"for {doc_id}: "
            f"num_nodes={num_nodes}, "
            f"expected={expected_pairs}, "
            f"successful={actual_pairs}, "
            f"errors={num_errors}"
        )

    return (
        doc_id,
        num_nodes,
        pairs,
        errors,
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    input_path: Path,
    output_path: Path,
    threshold,
    stats_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD SIMILARITY
    # ========================================================

    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Similarity articles:",
        f"{len(data):,}",
    )

    results = []
    stats = []

    total_edges = 0
    empty_graphs = 0
    documents_with_smatch_errors = 0
    total_smatch_errors = 0

    # ========================================================
    # BUILD GRAPHS
    # ========================================================

    for item in tqdm(
        data,
        desc="Building AMR graphs",
    ):

        (
            doc_id,
            num_nodes,
            pairs,
            errors,
        ) = validate_similarity_item(
            item
        )

        # ----------------------------------------------------
        # SMATCH ERRORS
        # ----------------------------------------------------

        if len(errors) > 0:

            documents_with_smatch_errors += 1
            total_smatch_errors += len(
                errors
            )

        # ----------------------------------------------------
        # BUILD GRAPH
        # ----------------------------------------------------

        (
            edge_index,
            edge_weight,
            resolved_threshold,
            mean_similarity,
            std_similarity,
        ) = build_graph_from_pairs(
            num_nodes=
                num_nodes,

            pairs=
                pairs,

            threshold=
                threshold,
        )

        # ====================================================
        # GRAPH VALIDATION
        # ====================================================

        if edge_index.ndim != 2:

            raise RuntimeError(
                f"{doc_id}: "
                f"edge_index must be 2D."
            )

        if edge_index.shape[0] != 2:

            raise RuntimeError(
                f"{doc_id}: "
                f"edge_index shape invalid: "
                f"{edge_index.shape}"
            )

        if (
            edge_index.shape[1]
            != edge_weight.shape[0]
        ):

            raise RuntimeError(
                f"{doc_id}: "
                "edge_index and "
                "edge_weight size mismatch."
            )

        if edge_index.numel() > 0:

            if (
                edge_index.max().item()
                >= num_nodes
            ):

                raise RuntimeError(
                    f"{doc_id}: "
                    "edge index exceeds "
                    "num_nodes."
                )

            if (
                edge_index.min().item()
                < 0
            ):

                raise RuntimeError(
                    f"{doc_id}: "
                    "negative edge index."
                )

        # ====================================================
        # FINAL GRAPH
        # ====================================================

        results.append({
            "id":
                doc_id,

            "edge_index":
                edge_index,

            "edge_weight":
                edge_weight,
        })

        # ====================================================
        # STATISTICS
        # ====================================================

        num_edges = int(
            edge_index.shape[1]
        )

        possible_directed_edges = (
            num_nodes
            * (
                num_nodes - 1
            )
        )

        if possible_directed_edges > 0:

            density = (
                num_edges
                / possible_directed_edges
            )

        else:

            density = 0.0

        if num_edges == 0:
            empty_graphs += 1

        total_edges += num_edges

        stats.append({
            "id":
                doc_id,

            "num_sentences":
                num_nodes,

            "num_pairs":
                len(pairs),

            "num_edges":
                num_edges,

            "density":
                density,

            "mean_similarity":
                mean_similarity,

            "std_similarity":
                std_similarity,

            "threshold":
                resolved_threshold,

            "smatch_errors":
                len(errors),
        })

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    torch.save(
        results,
        output_path,
    )

    # ========================================================
    # SAVE ANALYSIS
    # ========================================================

    torch.save(
        stats,
        stats_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    num_graphs = len(
        results
    )

    if num_graphs > 0:

        avg_edges = (
            total_edges
            / num_graphs
        )

        avg_density = (
            sum(
                x["density"]
                for x in stats
            )
            / num_graphs
        )

        avg_threshold_values = [
            x["threshold"]
            for x in stats
            if x["threshold"] is not None
        ]

        if avg_threshold_values:

            avg_threshold = (
                sum(
                    avg_threshold_values
                )
                / len(
                    avg_threshold_values
                )
            )

        else:

            avg_threshold = None

    else:

        avg_edges = 0.0
        avg_density = 0.0
        avg_threshold = None

    print()
    print("-" * 80)

    print(
        "Graphs                :",
        f"{num_graphs:,}",
    )

    print(
        "Total edges           :",
        f"{total_edges:,}",
    )

    print(
        "Average edges / graph :",
        f"{avg_edges:.2f}",
    )

    print(
        "Average density       :",
        f"{avg_density:.4f}",
    )

    print(
        "Empty graphs          :",
        f"{empty_graphs:,}",
    )

    if avg_threshold is not None:

        print(
            "Average threshold     :",
            f"{avg_threshold:.6f}",
        )

    else:

        print(
            "Average threshold     :",
            "N/A",
        )

    print(
        "Docs with SMATCH error:",
        f"{documents_with_smatch_errors:,}",
    )

    print(
        "Total SMATCH errors   :",
        f"{total_smatch_errors:,}",
    )

    print()

    print(
        "Graph saved to:"
    )

    print(
        output_path
    )

    print()

    print(
        "Analysis saved to:"
    )

    print(
        stats_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build sentence graphs from "
            "pairwise AMR similarities."
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
        default="smatch",
        help=(
            "AMR similarity method."
        ),
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    parser.add_argument(
        "--threshold",
        default="adaptive",
        help=(
            "adaptive, fully_connected, "
            "or numeric value such as 0.3"
        ),
    )

    # ========================================================
    # SPLIT
    # ========================================================

    parser.add_argument(
        "--split",
        default=None,
    )

    args = parser.parse_args()

    # ========================================================
    # THRESHOLD ARGUMENT
    # ========================================================

    if args.threshold in (
        "adaptive",
        "fully_connected",
    ):

        threshold = (
            args.threshold
        )

        threshold_str = (
            args.threshold
        )

    else:

        threshold = float(
            args.threshold
        )

        # 0.3 -> 03
        # 0.2 -> 02
        threshold_str = (
            str(threshold)
            .replace(
                ".",
                ""
            )
        )

    # ========================================================
    # SPLITS
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
                f"'{args.split}'. "
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

    similarity_dir = (
        dataset_dir
        / "amr"
        / "similarity"
        / args.method
    )

    graph_name = (
        f"{args.method}_t"
        f"{threshold_str}"
    )

    output_dir = (
        dataset_dir
        / "amr"
        / "graphs"
        / graph_name
    )

    stats_dir = (
        dataset_dir
        / "amr"
        / "analysis"
        / graph_name
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print("AMR SENTENCE GRAPH CONSTRUCTION")
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
        "Threshold     :",
        threshold,
    )

    print(
        "Similarity dir:",
        similarity_dir,
    )

    print(
        "Graph dir     :",
        output_dir,
    )

    print(
        "Analysis dir  :",
        stats_dir,
    )

    # ========================================================
    # PROCESS
    # ========================================================

    for split in splits:

        input_path = (
            similarity_dir
            / f"{split}.pt"
        )

        output_path = (
            output_dir
            / f"{split}.pt"
        )

        stats_path = (
            stats_dir
            / f"{split}.pt"
        )

        if not input_path.exists():

            print()
            print(
                f"[SKIP] {split}: "
                f"{input_path} not found"
            )

            continue

        print()
        print("=" * 80)
        print(
            f"GRAPH - {split.upper()}"
        )
        print("=" * 80)

        process_split(
            input_path=
                input_path,

            output_path=
                output_path,

            threshold=
                threshold,

            stats_path=
                stats_path,
        )

    print()
    print("=" * 80)
    print("GRAPH CONSTRUCTION FINISHED")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()