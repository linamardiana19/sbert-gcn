"""
Visualize GOLD (oracle) vs PRED sentence position histogram.

BLUE:
    GOLD oracle sentence positions from Rouge labels.

RED:
    Predicted sentence positions from model results.

Output:
    Saved inside the same model result directory.

Example:

uv run python scripts/visualize_oracle_vs_prediction_position_histogram.py \
 --dataset xlsum \
 --split test \
 --metric rouge1 \
 --results_dir results/xlsum/sbert_gcn_10ep_1e-3_256_bs16_thybrid_tadaptive_Empnet_d00_Smlp_Psin16 \
 --mode absolute \
 --max_absolute_position 30
"""


import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent



# ============================================================
# Utility
# ============================================================

def load_jsonl(path):

    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:

            line=line.strip()

            if line:
                rows.append(json.loads(line))

    return rows



def calculate_position(index, num_sentences, mode):

    if mode=="absolute":
        return index + 1

    if num_sentences <= 1:
        return 0

    return index/(num_sentences-1)



# ============================================================
# GOLD / ORACLE
# ============================================================

def collect_oracle_positions(
        dataset,
        split,
        metric,
        mode,
        max_absolute_position
):

    prep_path = (
        BASE_DIR /
        "dataset" /
        dataset /
        "preprocessed" /
        f"{split}.jsonl"
    )


    label_path = (
        BASE_DIR /
        "dataset" /
        dataset /
        "labels" /
        metric /
        f"{split}.jsonl"
    )


    documents = load_jsonl(prep_path)
    labels = load_jsonl(label_path)


    label_map = {
        str(row["id"]): row["labels"]
        for row in labels
    }


    positions=[]


    for doc in documents:


        doc_id=str(doc["id"])


        if doc_id not in label_map:
            continue


        sentences=doc["sentences"]
        doc_labels=label_map[doc_id]


        if len(sentences)!=len(doc_labels):
            continue


        for idx,label in enumerate(doc_labels):

            if int(label)==1:

                pos=calculate_position(
                    idx,
                    len(sentences),
                    mode
                )


                if (
                    mode=="absolute"
                    and pos > max_absolute_position
                ):
                    continue


                positions.append(pos)


    return positions



# ============================================================
# PRED
# ============================================================


def find_prediction_file(results_dir, split):

    patterns=[

        f"*{split}*prediction*.jsonl",
        f"*{split}*pred*.jsonl",
        f"*{split}*.jsonl"

    ]


    candidates=[]


    for pattern in patterns:

        candidates.extend(
            results_dir.glob(pattern)
        )


    if not candidates:

        raise FileNotFoundError(
            "Prediction file not found"
        )


    print("\nPrediction file used:")
    print(candidates[0])


    return candidates[0]



def get_doc_id(row):

    for key in [
        "id",
        "doc_id",
        "document_id",
        "source_id"
    ]:

        if key in row:
            return str(row[key])


    return None



def get_prediction_indices(row):


    for key in [

        "selected_indices",
        "predicted_indices",
        "pred_indices",
        "indices",
        "selected_ids"

    ]:


        if key in row and row[key] is not None:

            return [
                int(x)
                for x in row[key]
            ]


    return []



def collect_prediction_positions(
        dataset,
        split,
        results_dir,
        mode,
        max_absolute_position
):


    prediction_file=find_prediction_file(
        results_dir,
        split
    )


    prediction_rows=load_jsonl(
        prediction_file
    )


    prep_path=(

        BASE_DIR /
        "dataset" /
        dataset /
        "preprocessed" /
        f"{split}.jsonl"

    )


    documents=load_jsonl(
        prep_path
    )


    sentence_map={

        str(doc["id"]):
        doc["sentences"]

        for doc in documents

    }


    prediction_map={}


    for row in prediction_rows:

        doc_id=get_doc_id(row)

        if doc_id:

            prediction_map[doc_id]=row



    positions=[]


    for doc_id,sentences in sentence_map.items():


        if doc_id not in prediction_map:
            continue


        indices=get_prediction_indices(
            prediction_map[doc_id]
        )


        for idx in indices:


            if idx < 0 or idx >= len(sentences):
                continue


            pos=calculate_position(
                idx,
                len(sentences),
                mode
            )


            if (
                mode=="absolute"
                and pos > max_absolute_position
            ):
                continue


            positions.append(pos)



    return positions



# ============================================================
# Plot
# ============================================================


def plot_histogram(
        gold,
        pred,
        dataset,
        split,
        metric,
        max_absolute_position,
        output_path
):

    import numpy as np

    # Hitung frequency tiap posisi
    gold_counts = [
        gold.count(i)
        for i in range(1, max_absolute_position + 1)
    ]

    pred_counts = [
        pred.count(i)
        for i in range(1, max_absolute_position + 1)
    ]


    positions = np.arange(
        1,
        max_absolute_position + 1
    )


    # Lebar masing-masing bar
    width = 0.35


    fig, ax = plt.subplots(
        figsize=(12,5)
    )


    # GOLD bar (kiri)
    ax.bar(
        positions - width/2,
        gold_counts,
        width,
        color="blue",
        alpha=0.7,
        label="GOLD"
    )


    # PRED bar (kanan)
    ax.bar(
        positions + width/2,
        pred_counts,
        width,
        color="red",
        alpha=0.7,
        label="PRED"
    )



    ax.set_xlabel(
        "Position"
    )


    ax.set_ylabel(
        "Count"
    )


    ax.set_title(
        f"Sentence Position Distribution\n"
        f"{dataset.upper()} | {split} | {metric}"
    )


    ax.set_xticks(
        positions
    )


    ax.set_xlim(
        0.5,
        max_absolute_position + 0.5
    )


    ax.legend()


    # grid ringan agar mudah dibaca
    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# Main
# ============================================================


def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        required=True
    )


    parser.add_argument(
        "--split",
        required=True
    )


    parser.add_argument(
        "--metric",
        default="rouge1"
    )


    parser.add_argument(
        "--results_dir",
        required=True
    )


    parser.add_argument(
        "--mode",
        default="absolute"
    )


    parser.add_argument(
        "--max_absolute_position",
        type=int,
        default=30
    )


    args=parser.parse_args()



    results_dir=Path(
        args.results_dir
    )



    gold_positions=collect_oracle_positions(

        args.dataset,

        args.split,

        args.metric,

        args.mode,

        args.max_absolute_position

    )



    pred_positions=collect_prediction_positions(

        args.dataset,

        args.split,

        results_dir,

        args.mode,

        args.max_absolute_position

    )



    output_path=(

        results_dir /

        (
            f"oracle_vs_prediction_"
            f"{args.split}_"
            f"{args.metric}_"
            f"position_histogram.png"
        )

    )



    plot_histogram(

        gold_positions,

        pred_positions,

        args.dataset,

        args.split,

        args.metric,

        args.max_absolute_position,

        output_path

    )



    print("\n=== DONE ===")
    print("Saved:")
    print(output_path)


    print("\nGOLD sentences:")
    print(len(gold_positions))


    print("PRED sentences:")
    print(len(pred_positions))



if __name__=="__main__":

    main()