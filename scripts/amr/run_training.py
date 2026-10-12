# uv run python scripts/amr/run_training.py \
#     --dataset xlsum \
#     --encoder mpnet \
#     --graph_name smatch_tadaptive \
#     --epochs 10 \
#     --lr 1e-3 \
#     --hidden_dim 256 \
#     --batch_size 16 \
#     --num_layers 2 \
#     --residual outer \
#     --dropout 0.0 \
#     --scoring mlp \
#     --pos_weight 11 \
#     --pos_method sinusoidal \
#     --pos_dim 16 \
#     --top_k 2 \
#     --label_metric rouge1 \
#     --selection_metric rouge1


# ============================================================
# scripts/amr/run_training.py
#
# AMR-GCN EXTRACTIVE SUMMARIZATION
#
# Node feature:
#     SBERT sentence embedding
#
# Graph:
#     AMR Smatch graph
#     edge_index + edge_weight
#
# Model:
#     existing SBERTSummarizer / GCN
# ============================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path


# ============================================================
# BASE DIRECTORY
#
# File berada di:
# scripts/amr/run_training.py
#
# parents[0] = scripts/amr
# parents[1] = scripts
# parents[2] = root repository
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from summarizer.encoder import ENCODER_CONFIGS
from summarizer.model import SBERTSummarizer
from summarizer.positional import (
    make_position_encoding,
    make_simple_features,
    get_pos_dim,
)
from summarizer.selector import (
    select_sentences,
    select_sentences_mmr,
)
from evaluation.rouge import (
    compute_rouge,
    aggregate_rouge,
)


# ============================================================
# DATASET CONFIG
# ============================================================

DATASET_CONFIG = {

    "xlsum": {
        "train_split": "train",
        "val_split": "val",
    },

    "liputan6": {
        "train_split": "train",
        "val_split": "dev",
    },
}


# ============================================================
# HELPER
# ============================================================

def _lr_to_scientific(
    lr: float
) -> str:

    if lr == 0:
        return "0"

    exp = int(
        math.floor(
            math.log10(
                abs(lr)
            )
        )
    )

    mantissa = (
        lr
        / (10 ** exp)
    )

    if mantissa == 1.0:
        return f"1e{exp}"

    return (
        f"{mantissa:g}e{exp}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def _load_dataset(
    sbert_dir: Path,
    amr_graph_dir: Path,
    prep_dir: Path,
    split: str,
    limit: int | None = None,
    labels_dir: Path | None = None,
    pos_method: str = "none",
    pos_dim: int = 0,
) -> tuple[
    list[Data],
    dict[str, list[str]],
    dict[str, str],
]:

    print()
    print(
        f"Loading split: {split}"
    )

    # ========================================================
    # LOAD SBERT EMBEDDINGS
    # ========================================================

    embedding_path = (
        sbert_dir
        / "embeddings"
        / f"{split}.pt"
    )

    if not embedding_path.exists():

        raise FileNotFoundError(
            "SBERT embedding tidak ditemukan:\n"
            f"{embedding_path}"
        )

    embeddings_list = torch.load(
        embedding_path,
        weights_only=False,
        map_location="cpu",
    )


    # ========================================================
    # LOAD AMR GRAPH
    # ========================================================

    graph_path = (
        amr_graph_dir
        / f"{split}.pt"
    )

    if not graph_path.exists():

        raise FileNotFoundError(
            "AMR graph tidak ditemukan:\n"
            f"{graph_path}"
        )

    graphs_list = torch.load(
        graph_path,
        weights_only=False,
        map_location="cpu",
    )


    # ========================================================
    # LOAD LABELS
    # ========================================================

    labels_map: dict[
        str,
        list[int]
    ] = {}

    if labels_dir is None:

        raise ValueError(
            "labels_dir harus diberikan."
        )

    labels_path = (
        labels_dir
        / f"{split}.jsonl"
    )

    if not labels_path.exists():

        raise FileNotFoundError(
            "Label file tidak ditemukan:\n"
            f"{labels_path}"
        )

    with open(
        labels_path,
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            labels_map[
                row["id"]
            ] = row[
                "labels"
            ]


    # ========================================================
    # CREATE MAP BY ID
    # ========================================================

    emb_map = {

        item["id"]:
            item["embeddings"]

        for item
        in embeddings_list
    }


    graph_map = {

        item["id"]:
            item

        for item
        in graphs_list
    }


    # ========================================================
    # LOAD PREPROCESSED DATA
    #
    # Digunakan untuk:
    # - sentences
    # - gold/reference summary
    # ========================================================

    sentences_map: dict[
        str,
        list[str]
    ] = {}

    summary_map: dict[
        str,
        str
    ] = {}


    prep_path = (
        prep_dir
        / f"{split}.jsonl"
    )

    if not prep_path.exists():

        raise FileNotFoundError(
            "Preprocessed file tidak ditemukan:\n"
            f"{prep_path}"
        )

    with open(
        prep_path,
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            doc = json.loads(
                line
            )

            sentences_map[
                doc["id"]
            ] = doc[
                "sentences"
            ]

            summary_map[
                doc["id"]
            ] = doc[
                "summary"
            ]


    # ========================================================
    # COMMON IDS
    #
    # Article harus tersedia pada:
    #
    # SBERT embedding
    # AMR graph
    # labels
    # preprocessing
    # ========================================================

    common_ids = (
        set(emb_map)
        & set(graph_map)
        & set(labels_map)
        & set(sentences_map)
    )


    print(
        "Embedding IDs :",
        f"{len(emb_map):,}"
    )

    print(
        "AMR graph IDs :",
        f"{len(graph_map):,}"
    )

    print(
        "Label IDs     :",
        f"{len(labels_map):,}"
    )

    print(
        "Preprocess IDs:",
        f"{len(sentences_map):,}"
    )

    print(
        "Common IDs    :",
        f"{len(common_ids):,}"
    )


    # ========================================================
    # LIMIT
    # ========================================================

    if limit is not None:

        common_ids = set(
            list(
                common_ids
            )[:limit]
        )


    # ========================================================
    # BUILD PYTORCH GEOMETRIC DATA
    # ========================================================

    dataset: list[
        Data
    ] = []


    skipped_length = 0
    skipped_empty_graph = 0
    skipped_invalid_edge = 0


    for doc_id in common_ids:

        # ----------------------------------------------------
        # FEATURE MATRIX
        #
        # x = SBERT sentence embeddings
        # ----------------------------------------------------

        x = emb_map[
            doc_id
        ]


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        labels = torch.tensor(
            labels_map[
                doc_id
            ],
            dtype=torch.float,
        )


        # ----------------------------------------------------
        # AMR GRAPH
        # ----------------------------------------------------

        graph_item = graph_map[
            doc_id
        ]

        edge_index = graph_item[
            "edge_index"
        ]

        edge_weight = graph_item[
            "edge_weight"
        ]


        # ----------------------------------------------------
        # NUMBER OF SENTENCES
        # ----------------------------------------------------

        n_sents = x.size(
            0
        )


        # ====================================================
        # VALIDATE NODE LENGTH
        # ====================================================

        if len(labels) != n_sents:

            skipped_length += 1

            continue


        # Jika graph memiliki num_nodes,
        # cek juga dengan embedding.

        graph_num_nodes = graph_item.get(
            "num_nodes",
            n_sents,
        )


        if graph_num_nodes != n_sents:

            skipped_length += 1

            continue


        # ====================================================
        # VALIDATE EDGE
        # ====================================================

        if edge_index.numel() == 0:

            skipped_empty_graph += 1

            continue


        if (
            edge_index.shape[0]
            != 2
        ):

            skipped_invalid_edge += 1

            continue


        if (
            edge_index.shape[1]
            != edge_weight.shape[0]
        ):

            skipped_invalid_edge += 1

            continue


        if (
            edge_index.max().item()
            >= n_sents
        ):

            skipped_invalid_edge += 1

            continue


        # ====================================================
        # POSITIONAL FEATURES
        # ====================================================

        if pos_method == "simple":

            x = torch.cat(
                [
                    x,
                    make_simple_features(
                        n_sents
                    ),
                ],
                dim=-1,
            )


        elif pos_method == "sinusoidal":

            x = torch.cat(
                [
                    x,
                    make_position_encoding(
                        n_sents,
                        pos_dim,
                    ),
                ],
                dim=-1,
            )


        # ====================================================
        # CREATE DATA OBJECT
        # ====================================================

        data = Data(

            # -----------------------------------------------
            # SBERT NODE FEATURE
            # -----------------------------------------------

            x=x,

            # -----------------------------------------------
            # AMR GRAPH
            # -----------------------------------------------

            edge_index=edge_index,

            edge_weight=edge_weight,

            # -----------------------------------------------
            # ORACLE LABEL
            # -----------------------------------------------

            y=labels,
        )


        # ====================================================
        # LEARNED POSITION
        # ====================================================

        if pos_method == "learned":

            data.pos_idx = torch.arange(
                n_sents
            )


        data.doc_id = doc_id


        dataset.append(
            data
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "Loaded dataset      :",
        f"{len(dataset):,}"
    )

    print(
        "Skipped length      :",
        f"{skipped_length:,}"
    )

    print(
        "Skipped empty graph :",
        f"{skipped_empty_graph:,}"
    )

    print(
        "Skipped invalid edge:",
        f"{skipped_invalid_edge:,}"
    )


    return (
        dataset,
        sentences_map,
        summary_map,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model: SBERTSummarizer,
    val_data: list[Data],
    sentences_map: dict[
        str,
        list[str]
    ],
    summary_map: dict[
        str,
        str
    ],
    top_k: int,
    selection: str = "topk",
    lambda_mmr: float = 0.7,
) -> tuple[
    float,
    dict
]:

    model.eval()

    total_loss = 0.0

    n_docs = 0

    all_rouge = []


    with torch.no_grad():

        for data in val_data:

            # =================================================
            # GCN
            #
            # x           = SBERT embedding
            # edge_index  = AMR graph
            # edge_weight = Smatch score
            # =================================================

            logits = model(

                data.x,

                data.edge_index,

                data.edge_weight,

                None,

                None,

                pos_idx=getattr(
                    data,
                    "pos_idx",
                    None,
                ),
            )


            loss = model.compute_loss(
                logits,
                data.y,
            )


            total_loss += (
                loss.item()
            )

            n_docs += 1


            # =================================================
            # PROBABILITY
            # =================================================

            scores = torch.sigmoid(
                logits
            )


            sentences = sentences_map.get(
                data.doc_id,
                [],
            )


            reference = summary_map.get(
                data.doc_id,
                "",
            )


            if (
                sentences
                and
                reference
            ):

                # =============================================
                # SENTENCE SELECTION
                # =============================================

                if selection == "mmr":

                    result = (
                        select_sentences_mmr(
                            sentences,
                            scores.cpu(),
                            data.x.cpu(),
                            k=top_k,
                            lambda_=lambda_mmr,
                        )
                    )

                else:

                    result = (
                        select_sentences(
                            sentences,
                            scores.cpu(),
                            k=top_k,
                        )
                    )


                # =============================================
                # ROUGE
                # =============================================

                rouge = compute_rouge(
                    reference,
                    result.summary,
                )

                all_rouge.append(
                    rouge
                )


    avg_loss = (

        total_loss
        / n_docs

        if n_docs > 0

        else 0.0
    )


    agg = aggregate_rouge(
        all_rouge
    )


    return (
        avg_loss,
        agg,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def _save_checkpoint(
    model,
    optimizer,
    epoch,
    in_dim,
    hidden_dim,
    num_layers,
    residual,
    scoring,
    best_rouge_f1,
    pos_method,
    pos_dim,
    encoder,
    graph_name,
    selection_metric,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "in_dim":
                in_dim,

            "hidden_dim":
                hidden_dim,

            "num_layers":
                num_layers,

            "residual":
                residual,

            "scoring":
                scoring,

            # Keep compatibility
            "multi_gcn":
                False,

            "best_rouge_f1":
                best_rouge_f1,

            "selection_metric":
                selection_metric,

            "pos_method":
                pos_method,

            "pos_dim":
                pos_dim,

            "encoder":
                encoder,

            "graph_source":
                "amr",

            "graph_name":
                graph_name,

        },

        path,
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    model: SBERTSummarizer,
    train_loader: DataLoader,
    val_data: list[Data],
    val_sentences: dict[
        str,
        list[str]
    ],
    val_summaries: dict[
        str,
        str
    ],
    epochs: int,
    lr: float,
    top_k: int,
    scenario_dir: Path,
    device: torch.device,
    encoder: str,
    graph_name: str,
    patience: int = 3,
    selection_metric: str = "rouge1",
    pos_method: str = "none",
    pos_dim: int = 0,
    selection: str = "topk",
    lambda_mmr: float = 0.7,
) -> None:

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )


    # ========================================================
    # BEST TRACKER
    # ========================================================

    best_rouge = 0.0

    epochs_without_improvement = 0


    in_dim = model.gcn.in_dim

    hidden_dim = model.gcn.hidden_dim

    num_layers = model.gcn.num_layers

    residual = model.gcn.residual

    scoring = model.scoring


    metric_label = (
        selection_metric
        .upper()
        .replace(
            "ROUGE",
            "ROUGE-"
        )
    )


    best_path = (
        scenario_dir
        / "best.pt"
    )

    last_path = (
        scenario_dir
        / "last.pt"
    )


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()


        total_loss = 0.0

        n_batches = 0


        # ====================================================
        # TRAIN BATCH
        # ====================================================

        for batch in tqdm(
            train_loader,
            desc=(
                f"  Epoch "
                f"{epoch}/{epochs}"
            ),
        ):

            batch = batch.to(
                device
            )


            optimizer.zero_grad()


            # =================================================
            # FORWARD
            #
            # x           = SBERT
            # edge_index  = AMR
            # edge_weight = Smatch
            # =================================================

            logits = model(

                batch.x,

                batch.edge_index,

                batch.edge_weight,

                None,

                None,

                pos_idx=getattr(
                    batch,
                    "pos_idx",
                    None,
                ),
            )


            loss = model.compute_loss(
                logits,
                batch.y,
            )


            if torch.isnan(
                loss
            ):

                continue


            # =================================================
            # BACKPROP
            # =================================================

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


            total_loss += (
                loss.item()
            )


            n_batches += 1


        # ====================================================
        # TRAIN LOSS
        # ====================================================

        avg_train_loss = (

            total_loss
            / n_batches

            if n_batches > 0

            else 0.0
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            val_rouge,
        ) = validate(

            model,

            val_data,

            val_sentences,

            val_summaries,

            top_k,

            selection=selection,

            lambda_mmr=lambda_mmr,
        )


        r1 = (
            val_rouge[
                "rouge1"
            ]["f1"]
        )


        r2 = (
            val_rouge[
                "rouge2"
            ]["f1"]
        )


        rL = (
            val_rouge[
                "rougeL"
            ]["f1"]
        )


        print(
            f"  Epoch {epoch}: "
            f"train_loss="
            f"{avg_train_loss:.4f} "
            f"val_loss="
            f"{val_loss:.4f} "
            f"R1="
            f"{r1:.4f} "
            f"R2="
            f"{r2:.4f} "
            f"RL="
            f"{rL:.4f}"
        )


        # ====================================================
        # BEST MODEL SELECTION
        # ====================================================

        r_sel = (
            val_rouge[
                selection_metric
            ]["f1"]
        )


        if r_sel > best_rouge:

            best_rouge = (
                r_sel
            )

            epochs_without_improvement = 0


            _save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                in_dim=in_dim,

                hidden_dim=hidden_dim,

                num_layers=num_layers,

                residual=residual,

                scoring=scoring,

                best_rouge_f1=r_sel,

                pos_method=pos_method,

                pos_dim=pos_dim,

                encoder=encoder,

                graph_name=graph_name,

                selection_metric=selection_metric,

                path=best_path,
            )


            print(
                f"  Saved best model "
                f"({metric_label}: "
                f"{r_sel:.4f})"
            )


        else:

            epochs_without_improvement += 1


            if patience > 0:

                print(
                    f"  No improvement for "
                    f"{epochs_without_improvement}/"
                    f"{patience} epochs"
                )


                if (
                    epochs_without_improvement
                    >= patience
                ):

                    print(
                        "  Early stopping "
                        f"triggered at epoch "
                        f"{epoch}"
                    )

                    break


    # ========================================================
    # SAVE LAST
    # ========================================================

    _save_checkpoint(

        model=model,

        optimizer=optimizer,

        epoch=epoch,

        in_dim=in_dim,

        hidden_dim=hidden_dim,

        num_layers=num_layers,

        residual=residual,

        scoring=scoring,

        best_rouge_f1=r_sel,

        pos_method=pos_method,

        pos_dim=pos_dim,

        encoder=encoder,

        graph_name=graph_name,

        selection_metric=selection_metric,

        path=last_path,
    )


    print(
        f"  Saved last model "
        f"(epoch {epoch})"
    )


    print()

    print(
        f"Training complete. "
        f"Best {metric_label}: "
        f"{best_rouge:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # ARGUMENTS
    # ========================================================

    ap = argparse.ArgumentParser(
        description=(
            "Train SBERT feature + "
            "AMR Smatch GCN "
            "extractive summarizer."
        )
    )


    ap.add_argument(
        "--dataset",
        choices=[
            "xlsum",
            "liputan6",
        ],
        required=True,
    )


    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
    )


    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )


    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )


    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )


    ap.add_argument(
        "--top_k",
        type=int,
        default=3,
    )


    ap.add_argument(
        "--num_layers",
        type=int,
        default=2,
    )


    ap.add_argument(
        "--residual",
        choices=[
            "outer",
            "none",
        ],
        default="outer",
    )


    ap.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )


    ap.add_argument(
        "--scoring",
        choices=[
            "linear",
            "mlp",
            "context",
            "multigras",
        ],
        default="linear",
    )


    # ========================================================
    # SBERT FEATURE ENCODER
    # ========================================================

    ap.add_argument(
        "--encoder",
        default="mpnet",
        choices=list(
            ENCODER_CONFIGS.keys()
        ),
        help=(
            "SBERT embedding source. "
            "Dipakai sebagai node feature."
        ),
    )


    # ========================================================
    # AMR GRAPH
    # ========================================================

    ap.add_argument(
        "--graph_name",
        default="smatch_tadaptive",
        help=(
            "AMR graph folder name under "
            "dataset/<dataset>/amr/graphs/. "
            "Default: smatch_tadaptive"
        ),
    )


    ap.add_argument(
        "--graph_dir",
        default=None,
        help=(
            "Optional custom AMR graph directory. "
            "Jika tidak diberikan, menggunakan "
            "dataset/<dataset>/amr/graphs/"
            "<graph_name>"
        ),
    )


    # ========================================================
    # LOSS
    # ========================================================

    ap.add_argument(
        "--pos_weight",
        type=float,
        default=14.0,
    )


    # ========================================================
    # VALIDATION / EARLY STOPPING
    # ========================================================

    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "0 = disabled."
        ),
    )


    ap.add_argument(
        "--selection_metric",
        default="rouge1",
        choices=[
            "rouge1",
            "rouge2",
            "rougeL",
        ],
    )


    # ========================================================
    # POSITION
    # ========================================================

    ap.add_argument(
        "--pos_method",
        default="none",
        choices=[
            "none",
            "simple",
            "sinusoidal",
            "learned",
        ],
    )


    ap.add_argument(
        "--pos_dim",
        type=int,
        default=0,
    )


    # ========================================================
    # LABEL
    # ========================================================

    ap.add_argument(
        "--label_metric",
        default="rouge1",
        help=(
            "Label folder under "
            "dataset/<dataset>/labels/. "
            "Examples: rouge1, rouge2, "
            "best_combine_rouge1."
        ),
    )


    # ========================================================
    # SENTENCE SELECTION
    # ========================================================

    ap.add_argument(
        "--selection",
        choices=[
            "topk",
            "mmr",
        ],
        default="topk",
    )


    ap.add_argument(
        "--lambda_mmr",
        type=float,
        default=0.7,
    )


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    # ========================================================
    # DEBUG LIMIT
    # ========================================================

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
    )


    # ========================================================
    # OPTIONAL CUSTOM DATA FOLDER
    # ========================================================

    ap.add_argument(
        "--data_folder",
        default=None,
        help=(
            "Optional dataset base folder. "
            "Default: dataset/<dataset>"
        ),
    )


    # ========================================================
    # RUN TAG
    # ========================================================

    ap.add_argument(
        "--run_tag",
        default=None,
    )


    args = ap.parse_args()


    # ========================================================
    # SEED
    # ========================================================

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )


    # ========================================================
    # ENCODER CONFIG
    # ========================================================

    enc_cfg = ENCODER_CONFIGS[
        args.encoder
    ]


    pos_dim_effective = (
        get_pos_dim(
            args.pos_method,
            args.pos_dim,
        )
    )


    in_dim = (
        enc_cfg["dim"]
        + pos_dim_effective
    )


    # ========================================================
    # DATA FOLDER
    # ========================================================

    cfg = DATASET_CONFIG[
        args.dataset
    ]


    if args.data_folder:

        data_folder = Path(
            args.data_folder
        )


        if not data_folder.is_absolute():

            data_folder = (
                BASE_DIR
                / data_folder
            )


    else:

        data_folder = (
            BASE_DIR
            / "dataset"
            / args.dataset
        )


    # ========================================================
    # SBERT EMBEDDING DIRECTORY
    # ========================================================

    sbert_dir = (
        data_folder
        / enc_cfg[
            "sbert_subdir"
        ]
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    prep_dir = (
        BASE_DIR
        / "dataset"
        / args.dataset
        / "preprocessed"
    )


    # ========================================================
    # LABEL DIRECTORY
    # ========================================================

    labels_dir = (
        data_folder
        / "labels"
        / args.label_metric
    )


    # ========================================================
    # AMR GRAPH DIRECTORY
    # ========================================================

    if args.graph_dir:

        amr_graph_dir = Path(
            args.graph_dir
        )


        if not amr_graph_dir.is_absolute():

            amr_graph_dir = (
                BASE_DIR
                / amr_graph_dir
            )


    else:

        amr_graph_dir = (
            BASE_DIR
            / "dataset"
            / args.dataset
            / "amr"
            / "graphs"
            / args.graph_name
        )


    # ========================================================
    # PRINT PATHS
    # ========================================================

    print()
    print("=" * 80)
    print("AMR-GCN TRAINING CONFIGURATION")
    print("=" * 80)

    print(
        "Base directory :",
        BASE_DIR
    )

    print(
        "Dataset        :",
        args.dataset
    )

    print(
        "SBERT encoder  :",
        args.encoder
    )

    print(
        "SBERT directory:",
        sbert_dir
    )

    print(
        "AMR graph      :",
        amr_graph_dir
    )

    print(
        "Labels         :",
        labels_dir
    )

    print(
        "Preprocessed   :",
        prep_dir
    )


    # ========================================================
    # SCENARIO NAME
    # ========================================================

    lr_str = _lr_to_scientific(
        args.lr
    )


    encoder_suffix = (
        f"_E{args.encoder}"
    )


    layers_suffix = (

        f"_L{args.num_layers}"

        if args.num_layers != 2

        else ""
    )


    residual_suffix = (

        f"_R{args.residual}"

        if args.residual != "outer"

        else ""
    )


    dropout_suffix = (

        f"_d{args.dropout}"
        .replace(
            ".",
            ""
        )

        if args.dropout != 0.3

        else ""
    )


    scoring_suffix = (

        f"_S{args.scoring}"

        if args.scoring != "linear"

        else ""
    )


    if args.pos_method == "none":

        pos_suffix = ""


    elif args.pos_method == "simple":

        pos_suffix = (
            "_Psimple"
        )


    elif args.pos_method == "sinusoidal":

        pos_suffix = (
            f"_Psin"
            f"{pos_dim_effective}"
        )


    elif args.pos_method == "learned":

        pos_suffix = (
            f"_Plearned"
            f"{pos_dim_effective}"
        )


    label_suffix = (

        f"_Y{args.label_metric}"

        if args.label_metric
        != "rouge1"

        else ""
    )


    run_tag_suffix = (

        f"_{args.run_tag}"

        if args.run_tag

        else ""
    )


    scenario_name = (

        f"amr_gcn_"

        f"{args.epochs}ep_"

        f"{lr_str}_"

        f"{args.hidden_dim}_"

        f"bs{args.batch_size}_"

        f"{args.graph_name}"

        f"{encoder_suffix}"

        f"{layers_suffix}"

        f"{residual_suffix}"

        f"{dropout_suffix}"

        f"{scoring_suffix}"

        f"{pos_suffix}"

        f"{label_suffix}"

        f"{run_tag_suffix}"
    )


    # ========================================================
    # CHECKPOINT DIR
    # ========================================================

    scenario_dir = (

        BASE_DIR
        / "checkpoints"
        / args.dataset
        / "amr"
        / scenario_name
    )


    if (
        scenario_dir.exists()
        and
        any(
            scenario_dir.iterdir()
        )
    ):

        response = input(

            f"Scenario "
            f"'{scenario_name}' "
            f"already exists. "
            f"Overwrite? [y/N]: "
        )


        if (
            response.lower()
            != "y"
        ):

            print(
                "Aborted."
            )

            return


    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # LOG
    # ========================================================

    log_path = (
        scenario_dir
        / "train.log"
    )


    log_buffer = (
        io.StringIO()
    )


    original_stdout = (
        sys.stdout
    )


    class Tee:

        def __init__(
            self,
            original,
            buffer,
        ):

            self.original = (
                original
            )

            self.buffer = (
                buffer
            )


        def write(
            self,
            text,
        ):

            self.original.write(
                text
            )

            self.buffer.write(
                text
            )


        def flush(
            self,
        ):

            self.original.flush()


    sys.stdout = Tee(
        original_stdout,
        log_buffer,
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    # ========================================================
    # LOAD TRAIN
    # ========================================================

    print()
    print(
        "Loading training data..."
    )


    (
        train_data,
        _,
        _,
    ) = _load_dataset(

        sbert_dir=sbert_dir,

        amr_graph_dir=amr_graph_dir,

        prep_dir=prep_dir,

        split=cfg[
            "train_split"
        ],

        limit=args.limit,

        labels_dir=labels_dir,

        pos_method=args.pos_method,

        pos_dim=pos_dim_effective,
    )


    print(
        f"  {len(train_data):,} "
        f"training documents"
    )


    # ========================================================
    # LOAD VALIDATION
    # ========================================================

    print()
    print(
        "Loading validation data..."
    )


    (
        val_data,
        val_sentences,
        val_summaries,
    ) = _load_dataset(

        sbert_dir=sbert_dir,

        amr_graph_dir=amr_graph_dir,

        prep_dir=prep_dir,

        split=cfg[
            "val_split"
        ],

        limit=args.limit,

        labels_dir=labels_dir,

        pos_method=args.pos_method,

        pos_dim=pos_dim_effective,
    )


    # ========================================================
    # MOVE VALIDATION TO DEVICE
    # ========================================================

    val_data = [

        data.to(
            device
        )

        for data
        in val_data
    ]


    print(
        f"  {len(val_data):,} "
        f"validation documents"
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    train_loader = DataLoader(

        train_data,

        batch_size=args.batch_size,

        shuffle=True,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = SBERTSummarizer(

        in_dim=in_dim,

        hidden_dim=args.hidden_dim,

        dropout=args.dropout,

        pos_weight=args.pos_weight,

        num_layers=args.num_layers,

        residual=args.residual,

        scoring=args.scoring,

        multi_gcn=False,

        pos_method=args.pos_method,

        pos_dim=pos_dim_effective,

    ).to(
        device
    )


    # ========================================================
    # PRINT MODEL
    # ========================================================
    
    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)
    
    model_params = sum(
        p.numel()
        for p in model.parameters()
    )
    
    print(
        "Scenario:",
        scenario_name
    )
    
    print(
        "Model parameters:",
        f"{model_params:,}"
    )
    
    print(
        "Device:",
        device
    )
    
    print(
        "Batch size:",
        args.batch_size
    )
    
    print(
        "Input dimension:",
        in_dim
    )
    
    print(
        "Hidden dimension:",
        args.hidden_dim
    )
    
    print(
        "GCN layers:",
        args.num_layers
    )
    
    print(
        "Graph source:",
        "AMR Smatch"
    )
    
    print(
        "Graph:",
        args.graph_name
    )
    
    print(
        "Feature source:",
        f"SBERT {args.encoder}"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    train(

        model=model,

        train_loader=train_loader,

        val_data=val_data,

        val_sentences=val_sentences,

        val_summaries=val_summaries,

        epochs=args.epochs,

        lr=args.lr,

        top_k=args.top_k,

        scenario_dir=scenario_dir,

        device=device,

        encoder=args.encoder,

        graph_name=args.graph_name,

        patience=args.patience,

        selection_metric=args.selection_metric,

        pos_method=args.pos_method,

        pos_dim=pos_dim_effective,

        selection=args.selection,

        lambda_mmr=args.lambda_mmr,
    )


    # ========================================================
    # SAVE LOG
    # ========================================================

    sys.stdout = (
        original_stdout
    )


    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            log_buffer.getvalue()
        )


    print(
        f"Log saved to "
        f"{log_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()