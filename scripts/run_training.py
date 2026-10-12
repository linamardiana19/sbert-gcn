import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from summarizer.encoder import ENCODER_CONFIGS
from summarizer.model import SBERTSummarizer
from summarizer.positional import make_position_encoding, make_simple_features, get_pos_dim
from summarizer.selector import select_sentences, select_sentences_mmr
from evaluation.rouge import compute_rouge, aggregate_rouge

DATASET_CONFIG = {
    "xlsum": {"train_split": "train", "val_split": "val"},
    "liputan6": {"train_split": "train", "val_split": "dev"},
}


def _lr_to_scientific(lr: float) -> str:
    if lr == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / (10 ** exp)
    if mantissa == 1.0:
        return f"1e{exp}"
    return f"{mantissa:g}e{exp}"


def _load_dataset(sbert_dir: Path, prep_dir: Path, split: str, threshold: float | str = 0.3, limit: int | None = None, labels_dir: Path | None = None, tfidf_graph_dir: Path | None = None, pos_method: str = "none", pos_dim: int = 0) -> tuple[list[Data], dict[str, list[str]], dict[str, str]]:
    embeddings_list = torch.load(sbert_dir / "embeddings" / f"{split}.pt", weights_only=False)
    # if threshold in ("adaptive", "tfidf"):
    #     threshold_str = threshold
    # elif isinstance(threshold, str) and threshold.startswith("hybrid"):
    #     threshold_str = threshold
    # else:
    #     threshold_str = str(float(threshold)).replace(".", "")

    if threshold in ("adaptive", "tfidf", "fully_connected"):
        threshold_str = threshold
    
    elif isinstance(threshold, str) and threshold.startswith("hybrid"):
        threshold_str = threshold
    
    else:
        threshold_str = str(float(threshold)).replace(".", "")


    if threshold == "tfidf":
        graph_subfolder = "tfidf"
    elif threshold_str.startswith("hybrid"):
        graph_subfolder = threshold_str
    else:
        graph_subfolder = f"t{threshold_str}"
    graphs_list = torch.load(sbert_dir / "graphs" / graph_subfolder / f"{split}.pt", weights_only=False)

    tfidf_graph_map = None
    if tfidf_graph_dir:
        tfidf_graphs_list = torch.load(tfidf_graph_dir / f"{split}.pt", weights_only=False)
        tfidf_graph_map = {item["id"]: item for item in tfidf_graphs_list}

    labels_map: dict[str, list[int]] = {}
    labels_base = labels_dir if labels_dir else sbert_dir / "labels"
    labels_path = labels_base / f"{split}.jsonl"
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                labels_map[row["id"]] = row["labels"]

    emb_map = {item["id"]: item["embeddings"] for item in embeddings_list}
    graph_map = {item["id"]: item for item in graphs_list}

    sentences_map: dict[str, list[str]] = {}
    summary_map: dict[str, str] = {}
    with open(prep_dir / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            sentences_map[doc["id"]] = doc["sentences"]
            summary_map[doc["id"]] = doc["summary"]

    common_ids = set(emb_map) & set(graph_map) & set(labels_map)
    if tfidf_graph_map:
        common_ids &= set(tfidf_graph_map)
    if limit is not None:
        common_ids = set(list(common_ids)[:limit])

    dataset: list[Data] = []
    for doc_id in common_ids:
        labels = torch.tensor(labels_map[doc_id], dtype=torch.float)
        x = emb_map[doc_id]
        n_sents = x.size(0)
        if pos_method == "simple":
            x = torch.cat([x, make_simple_features(n_sents)], dim=-1)
        elif pos_method == "sinusoidal":
            x = torch.cat([x, make_position_encoding(n_sents, pos_dim)], dim=-1)
        data = Data(
            x=x,
            edge_index=graph_map[doc_id]["edge_index"],
            edge_weight=graph_map[doc_id]["edge_weight"],
            y=labels,
        )
        if pos_method == "learned":
            data.pos_idx = torch.arange(n_sents)
        if tfidf_graph_map:
            data.edge_index_tfidf = tfidf_graph_map[doc_id]["edge_index"]
            data.edge_weight_tfidf = tfidf_graph_map[doc_id]["edge_weight"]
        data.doc_id = doc_id
        if data.edge_index.numel() > 0:
            dataset.append(data)

    return dataset, sentences_map, summary_map


def validate(
    model: SBERTSummarizer,
    val_data: list[Data],
    sentences_map: dict[str, list[str]],
    summary_map: dict[str, str],
    top_k: int,
    selection: str = "topk",
    lambda_mmr: float = 0.7,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_docs = 0

    all_rouge = []
    with torch.no_grad():
        for data in val_data:
            logits = model(data.x, data.edge_index, data.edge_weight,
                           getattr(data, 'edge_index_tfidf', None),
                           getattr(data, 'edge_weight_tfidf', None),
                           pos_idx=getattr(data, 'pos_idx', None))
            loss = model.compute_loss(logits, data.y)
            total_loss += loss.item()
            n_docs += 1

            scores = torch.sigmoid(logits)
            sentences = sentences_map.get(data.doc_id, [])
            reference = summary_map.get(data.doc_id, "")

            if sentences and reference:
                if selection == "mmr":
                    result = select_sentences_mmr(sentences, scores.cpu(), data.x.cpu(), k=top_k, lambda_=lambda_mmr)
                else:
                    result = select_sentences(sentences, scores.cpu(), k=top_k)
                rouge = compute_rouge(reference, result.summary)
                all_rouge.append(rouge)

    avg_loss = total_loss / n_docs if n_docs > 0 else 0.0
    agg = aggregate_rouge(all_rouge)
    return avg_loss, agg


def _save_checkpoint(model, optimizer, epoch, in_dim, hidden_dim, num_layers, residual, scoring, multi_gcn, best_rouge_f1, pos_method, pos_dim, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "residual": residual,
        "scoring": scoring,
        "multi_gcn": multi_gcn,
        "best_rouge_f1": best_rouge_f1,
        "pos_method": pos_method,
        "pos_dim": pos_dim,
    }, path)


def train(
    model: SBERTSummarizer,
    train_loader: DataLoader,
    val_data: list[Data],
    val_sentences: dict[str, list[str]],
    val_summaries: dict[str, str],
    epochs: int,
    lr: float,
    top_k: int,
    scenario_dir: Path,
    device: torch.device,
    patience: int = 3,
    selection_metric: str = "rouge1",
    pos_method: str = "none",
    pos_dim: int = 0,
    selection: str = "topk",
    lambda_mmr: float = 0.7,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_rouge = 0.0
    epochs_without_improvement = 0
    in_dim = model.gcn.in_dim
    hidden_dim = model.gcn.hidden_dim
    num_layers = model.gcn.num_layers
    residual = model.gcn.residual
    scoring = model.scoring
    multi_gcn = model.multi_gcn

    metric_label = selection_metric.upper().replace("ROUGE", "ROUGE-")

    best_path = scenario_dir / "best.pt"
    last_path = scenario_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_weight,
                           getattr(batch, 'edge_index_tfidf', None),
                           getattr(batch, 'edge_weight_tfidf', None),
                           pos_idx=getattr(batch, 'pos_idx', None))
            loss = model.compute_loss(logits, batch.y)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches if n_batches > 0 else 0.0

        val_loss, val_rouge = validate(model, val_data, val_sentences, val_summaries, top_k, selection=selection, lambda_mmr=lambda_mmr)

        r1 = val_rouge["rouge1"]["f1"]
        r2 = val_rouge["rouge2"]["f1"]
        rL = val_rouge["rougeL"]["f1"]
        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} R1={r1:.4f} R2={r2:.4f} RL={rL:.4f}")

        r_sel = val_rouge[selection_metric]["f1"]
        if r_sel > best_rouge:
            best_rouge = r_sel
            epochs_without_improvement = 0
            _save_checkpoint(model, optimizer, epoch, in_dim, hidden_dim, num_layers, residual, scoring, multi_gcn, r_sel, pos_method, pos_dim, best_path)
            print(f"  Saved best model ({metric_label}: {r_sel:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0:
                print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping triggered at epoch {epoch}")
                    break

    _save_checkpoint(model, optimizer, epoch, in_dim, hidden_dim, num_layers, residual, scoring, multi_gcn, r_sel, pos_method, pos_dim, last_path)
    print(f"  Saved last model (epoch {epoch})")
    print(f"\nTraining complete. Best {metric_label}: {best_rouge:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SBERT+GCN extractive summarizer.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--threshold", default="0.3", help="Graph similarity threshold, or 'adaptive' (must match run_sbert_graph.py).")
    ap.add_argument("--num_layers", type=int, default=2, help="Number of GCN layers (default: 2).")
    ap.add_argument("--residual", choices=["outer", "none"], default="outer", help="Residual connection type (default: outer).")
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before scoring head (default: 0.3).")
    ap.add_argument("--scoring", choices=["linear", "mlp", "context", "multigras"], default="linear", help="Scoring head type (default: linear).")
    ap.add_argument("--encoder", default="denaya", choices=list(ENCODER_CONFIGS.keys()),
                     help="Encoder preset (default: denaya). Determines embeddings directory and input dimension.")
    ap.add_argument("--multi_gcn", action="store_true",
                     help="Use dual GCN (semantic + TF-IDF graphs). Requires cosine and TF-IDF graphs to be built.")
    ap.add_argument("--pos_weight", type=float, default=14.0,
                     help="Positive class weight for BCEWithLogitsLoss (default: 14.0). Use analyze_dataset.py to compute.")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience. 0 = disabled (run all epochs).")
    ap.add_argument("--selection_metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                     help="ROUGE metric for best-model selection and early stopping (default: rouge1).")
    ap.add_argument("--pos_method", default="none", choices=["none", "simple", "sinusoidal", "learned"],
                     help="Positional encoding method (default: none).")
    ap.add_argument("--pos_dim", type=int, default=0,
                     help="Positional encoding dimension for sinusoidal/learned (default: 0). Ignored for none/simple.")
    # ap.add_argument("--label_metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                     # help="ROUGE metric used for labeling (default: rouge1). Must match run_labeling.py --metric.")
    ap.add_argument("--label_metric", default="rouge1", help=("Label folder name under dataset/<dataset>/labels/. "
        "Examples: rouge1, rouge2, rougeL, best_combine_rouge1, "
        "best_combine_rouge1_tie_rouge2."),)
    ap.add_argument("--selection", choices=["topk", "mmr"], default="topk",
                     help="Sentence selection method for validation (default: topk).")
    ap.add_argument("--lambda_mmr", type=float, default=0.7,
                     help="MMR lambda trade-off: 1.0=pure relevance, 0.0=pure diversity (default: 0.7).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data_folder", default=None, help=(
        "Custom dataset folder containing labels, embeddings, and graphs. "
        "If omitted, defaults to dataset/<dataset>. "
        "Example: dataset/faktualitas/liputan6"),)
    ap.add_argument(
    "--run_tag", default=None, help=(
        "Optional tag appended to checkpoint folder name. "
        "Example: factuality"),)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    enc_cfg = ENCODER_CONFIGS[args.encoder]
    pos_dim_effective = get_pos_dim(args.pos_method, args.pos_dim)
    in_dim = enc_cfg["dim"] + pos_dim_effective

    # cfg = DATASET_CONFIG[args.dataset]
    # sbert_dir = BASE_DIR / "dataset" / args.dataset / enc_cfg["sbert_subdir"]
    # prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"

    cfg = DATASET_CONFIG[args.dataset]
    if args.data_folder:
        data_folder = Path(args.data_folder)
        if not data_folder.is_absolute():
            data_folder = BASE_DIR / data_folder
    else:
        data_folder = BASE_DIR / "dataset" / args.dataset

    sbert_dir = data_folder / enc_cfg["sbert_subdir"]
    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"

    lr_str = _lr_to_scientific(args.lr)
    # if args.threshold in ("adaptive", "tfidf") or args.threshold.startswith("hybrid"):
    #     threshold = args.threshold
    #     threshold_str = args.threshold
    # else:
    #     threshold = float(args.threshold)
    #     threshold_str = str(threshold).replace(".", "")

    if (
        args.threshold in ("adaptive", "tfidf", "fully_connected")
        or args.threshold.startswith("hybrid")
    ):
        threshold = args.threshold
        threshold_str = args.threshold
    else:
        threshold = float(args.threshold)
        threshold_str = str(threshold).replace(".", "")
    encoder_suffix = f"_E{args.encoder}" if args.encoder != "denaya" else ""
    multi_gcn_suffix = "_multigcn" if args.multi_gcn else ""
    layers_suffix = f"_L{args.num_layers}" if args.num_layers != 2 else ""
    residual_suffix = f"_R{args.residual}" if args.residual != "outer" else ""
    dropout_suffix = f"_d{args.dropout}".replace(".", "") if args.dropout != 0.3 else ""
    scoring_suffix = f"_S{args.scoring}" if args.scoring != "linear" else ""
    if args.pos_method == "none":
        pos_suffix = ""
    elif args.pos_method == "simple":
        pos_suffix = "_Psimple"
    elif args.pos_method == "sinusoidal":
        pos_suffix = f"_Psin{pos_dim_effective}"
    elif args.pos_method == "learned":
        pos_suffix = f"_Plearned{pos_dim_effective}"
    label_suffix = f"_Y{args.label_metric}" if args.label_metric != "rouge1" else ""
    # scenario_name = f"sbert_gcn_{args.epochs}ep_{lr_str}_{args.hidden_dim}_bs{args.batch_size}_t{threshold_str}{encoder_suffix}{multi_gcn_suffix}{layers_suffix}{residual_suffix}{dropout_suffix}{scoring_suffix}{pos_suffix}"
    # scenario_name = (
    # f"sbert_gcn_{args.epochs}ep_{lr_str}_{args.hidden_dim}_bs{args.batch_size}"
    # f"_t{threshold_str}{encoder_suffix}{multi_gcn_suffix}{layers_suffix}"
    # f"{residual_suffix}{dropout_suffix}{scoring_suffix}{pos_suffix}{label_suffix}")
    
    run_tag_suffix = f"_{args.run_tag}" if args.run_tag else ""

    scenario_name = (
    f"sbert_gcn_{args.epochs}ep_{lr_str}_{args.hidden_dim}_bs{args.batch_size}"
    f"_t{threshold_str}{encoder_suffix}{multi_gcn_suffix}{layers_suffix}"
    f"{residual_suffix}{dropout_suffix}{scoring_suffix}{pos_suffix}"
    f"{label_suffix}{run_tag_suffix}"
    )

    scenario_dir = BASE_DIR / "checkpoints" / args.dataset / scenario_name

    if scenario_dir.exists() and any(scenario_dir.iterdir()):
        response = input(f"Scenario '{scenario_name}' already exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return

    scenario_dir.mkdir(parents=True, exist_ok=True)
    log_path = scenario_dir / "train.log"
    log_buffer = io.StringIO()
    original_stdout = sys.stdout

    class Tee:
        def __init__(self, original, buffer):
            self.original = original
            self.buffer = buffer
        def write(self, text):
            self.original.write(text)
            self.buffer.write(text)
        def flush(self):
            self.original.flush()

    sys.stdout = Tee(original_stdout, log_buffer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels_dir = BASE_DIR / "dataset" / args.dataset / "labels" / args.label_metric
    labels_dir = data_folder / "labels" / args.label_metric
    tfidf_graph_dir = None
    if args.multi_gcn:
        tfidf_graph_dir = sbert_dir / "graphs" / "tfidf"

    print("Loading training data...")
    train_data, _, _ = _load_dataset(sbert_dir, prep_dir, cfg["train_split"], threshold, args.limit, labels_dir=labels_dir, tfidf_graph_dir=tfidf_graph_dir, pos_method=args.pos_method, pos_dim=pos_dim_effective)
    print(f"  {len(train_data)} training documents")

    print("Loading validation data...")
    val_data, val_sentences, val_summaries = _load_dataset(sbert_dir, prep_dir, cfg["val_split"], threshold, args.limit, labels_dir=labels_dir, tfidf_graph_dir=tfidf_graph_dir, pos_method=args.pos_method, pos_dim=pos_dim_effective)
    val_data = [d.to(device) for d in val_data]
    print(f"  {len(val_data)} validation documents")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = SBERTSummarizer(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, pos_weight=args.pos_weight, num_layers=args.num_layers, residual=args.residual, scoring=args.scoring, multi_gcn=args.multi_gcn, pos_method=args.pos_method, pos_dim=pos_dim_effective).to(device)
    print(f"Scenario: {scenario_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

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
        patience=args.patience,
        selection_metric=args.selection_metric,
        pos_method=args.pos_method,
        pos_dim=pos_dim_effective,
        selection=args.selection,
        lambda_mmr=args.lambda_mmr,
    )

    sys.stdout = original_stdout
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
