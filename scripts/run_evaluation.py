import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from summarizer.encoder import ENCODER_CONFIGS
from summarizer.model import SBERTSummarizer
from summarizer.positional import make_position_encoding, make_simple_features, get_pos_dim
from summarizer.selector import select_sentences, select_sentences_mmr
from evaluation.rouge import compute_rouge, aggregate_rouge

DATASET_CONFIG = {
    "xlsum": {"test_split": "test"},
    "liputan6": {"test_split": "test"},
}


def _load_test_data(sbert_dir: Path, prep_dir: Path, split: str, threshold: float | str = 0.3, tfidf_graph_dir: Path | None = None, pos_method: str = "none", pos_dim: int = 0) -> tuple[list[Data], dict[str, list[str]], dict[str, str]]:
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
    elif isinstance(threshold, str) and threshold_str.startswith("hybrid"):
        graph_subfolder = threshold_str
    else:
        graph_subfolder = f"t{threshold_str}"
    graphs_list = torch.load(sbert_dir / "graphs" / graph_subfolder / f"{split}.pt", weights_only=False)

    tfidf_graph_map = None
    if tfidf_graph_dir:
        tfidf_graphs_list = torch.load(tfidf_graph_dir / f"{split}.pt", weights_only=False)
        tfidf_graph_map = {item["id"]: item for item in tfidf_graphs_list}

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

    common_ids = set(emb_map) & set(graph_map) & set(sentences_map)
    if tfidf_graph_map:
        common_ids &= set(tfidf_graph_map)

    dataset: list[Data] = []
    for doc_id in common_ids:
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
        )
        if pos_method == "learned":
            data.pos_idx = torch.arange(n_sents)
        if tfidf_graph_map:
            data.edge_index_tfidf = tfidf_graph_map[doc_id]["edge_index"]
            data.edge_weight_tfidf = tfidf_graph_map[doc_id]["edge_weight"]
        data.doc_id = doc_id
        dataset.append(data)

    return dataset, sentences_map, summary_map


def _infer_pos_method(ckpt: dict) -> tuple[str, int]:
    if "pos_method" in ckpt:
        return ckpt["pos_method"], ckpt.get("pos_dim", 0)
    pos_dim = ckpt.get("pos_dim", 0)
    if pos_dim == 0:
        return "none", 0
    return "sinusoidal", pos_dim


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SBERT+GCN extractive summarizer with ROUGE.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (e.g. checkpoints/xlsum/scenario_name/best.pt)")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--split", default="test", choices=["train", "val", "dev", "test"], help="Split to evaluate. If omitted, uses test split.",) ##lina
    ap.add_argument("--selection", choices=["topk", "mmr"], default="topk", help="Sentence selection method (default: topk).")
    ap.add_argument("--lambda_mmr", type=float, default=0.7, help="MMR lambda trade-off: 1.0=pure relevance, 0.0=pure diversity (default: 0.7).")
    ap.add_argument("--threshold", default="0.3", help="Graph similarity threshold, or 'adaptive'/'fully_connected' (must match the graph used during training)."
)
    ap.add_argument("--encoder", default="denaya", choices=list(ENCODER_CONFIGS.keys()),
                     help="Encoder preset (default: denaya). Determines embeddings/graphs directory.")
    args = ap.parse_args()

    # if args.threshold in ("adaptive", "tfidf") or args.threshold.startswith("hybrid"):
    #     threshold = args.threshold
    # else:
    #     threshold = float(args.threshold)

    if (
        args.threshold in ("adaptive", "tfidf", "fully_connected")
        or args.threshold.startswith("hybrid")
    ):
        threshold = args.threshold
    else:
        threshold = float(args.threshold)

    enc_cfg = ENCODER_CONFIGS[args.encoder]

    cfg = DATASET_CONFIG[args.dataset]
    split = args.split if args.split else cfg["test_split"] ##lina
    sbert_dir = BASE_DIR / "dataset" / args.dataset / enc_cfg["sbert_subdir"]
    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"
    results_dir = BASE_DIR / "results" / args.dataset

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading model...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    in_dim = ckpt.get("in_dim", 256)
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt.get("num_layers", 2)
    residual = ckpt.get("residual", "outer")
    scoring = ckpt.get("scoring", "linear")
    multi_gcn = ckpt.get("multi_gcn", False)
    pos_method, pos_dim = _infer_pos_method(ckpt)
    model = SBERTSummarizer(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, residual=residual, scoring=scoring, multi_gcn=multi_gcn, pos_method=pos_method, pos_dim=get_pos_dim(pos_method, pos_dim))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_rouge_f1 = ckpt.get("best_rouge_f1", ckpt.get("rouge2_f1", 0.0))
    print(f"  Loaded from epoch {ckpt['epoch']} (best ROUGE F1: {best_rouge_f1:.4f}, pos={pos_method}, pos_dim={pos_dim})")

    tfidf_graph_dir = None
    if multi_gcn:
        tfidf_graph_dir = sbert_dir / "graphs" / "tfidf"

    print(f"Loading test data ({cfg['test_split']})...")
    # test_data, sentences_map, summary_map = _load_test_data(sbert_dir, prep_dir, cfg["test_split"], threshold, tfidf_graph_dir=tfidf_graph_dir, pos_method=pos_method, pos_dim=pos_dim)
    test_data, sentences_map, summary_map = _load_test_data(sbert_dir,prep_dir,split,threshold,tfidf_graph_dir=tfidf_graph_dir,pos_method=pos_method,pos_dim=pos_dim,) ##lina
    print(f"  {len(test_data)} test documents")

    results = []
    with torch.no_grad():
        for data in tqdm(test_data, desc="Evaluating"):
            scores = model.predict(data.x, data.edge_index, data.edge_weight,
                                   getattr(data, 'edge_index_tfidf', None),
                                   getattr(data, 'edge_weight_tfidf', None),
                                   pos_idx=getattr(data, 'pos_idx', None))
            sentences = sentences_map.get(data.doc_id, [])
            reference = summary_map.get(data.doc_id, "")

            if not sentences or not reference:
                continue

            if args.selection == "mmr":
                selection = select_sentences_mmr(sentences, scores, data.x, k=args.top_k, lambda_=args.lambda_mmr)
            else:
                selection = select_sentences(sentences, scores, k=args.top_k)
            rouge_scores = compute_rouge(reference, selection.summary)

            results.append({
                "id": data.doc_id,
                "reference": reference,
                "generated": selection.summary,
                "selected_indices": selection.selected_indices,
                "rouge1": rouge_scores["rouge1"]["f1"],
                "rouge2": rouge_scores["rouge2"]["f1"],
                "rougeL": rouge_scores["rougeL"]["f1"],
                "_full_rouge": rouge_scores,
            })

    all_rouge = [r["_full_rouge"] for r in results]
    agg = aggregate_rouge(all_rouge)
    agg["num_documents"] = len(results)
    agg["checkpoint"] = str(ckpt_path)
    agg["split"] = split ##lina
    agg["top_k"] = args.top_k
    agg["selection"] = args.selection
    if args.selection == "mmr":
        agg["lambda_mmr"] = args.lambda_mmr

    scenario_name = ckpt_path.parent.name
    model_type = ckpt_path.stem
    eval_dir = results_dir / scenario_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    sel_suffix = f"_{args.selection}" if args.selection != "topk" else ""
    mmr_suffix = f"_l{args.lambda_mmr}" if args.selection == "mmr" else ""
    k_suffix = f"_k{args.top_k}" if args.top_k != 3 else ""
    # summaries_path = eval_dir / f"summaries_{model_type}{k_suffix}{sel_suffix}{mmr_suffix}.jsonl"
    summaries_path = eval_dir / f"summaries_{model_type}_{split}{k_suffix}{sel_suffix}{mmr_suffix}.jsonl" ##lina
    with open(summaries_path, "w", encoding="utf-8") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "_full_rouge"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # scores_path = eval_dir / f"scores_{model_type}{k_suffix}{sel_suffix}{mmr_suffix}.json"
    scores_path = eval_dir / f"scores_{model_type}_{split}{k_suffix}{sel_suffix}{mmr_suffix}.json" ##lina
    scores_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEvaluated {len(results)} documents")
    print(f"ROUGE-1 F1: {agg['rouge1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {agg['rouge2']['f1']:.4f}")
    print(f"ROUGE-L F1: {agg['rougeL']['f1']:.4f}")
    print(f"Results saved to {eval_dir}")


if __name__ == "__main__":
    main()
