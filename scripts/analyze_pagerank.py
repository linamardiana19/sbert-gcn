"""
Compute PageRank baseline ROUGE scores.

PageRank computes sentence importance from the graph structure alone
(unsupervised — no trained model needed). This provides a graph-based
baseline to compare against GCN.

Usage:
    uv run python scripts/analyze_pagerank.py --dataset xlsum --encoder mpnet --threshold hybrid_tadaptive
    uv run python scripts/analyze_pagerank.py --dataset xlsum --encoder mpnet --threshold hybrid_tadaptive --top_k 3
    uv run python scripts/analyze_pagerank.py --dataset xlsum --encoder mpnet --threshold hybrid_tadaptive --damping 0.85
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from tqdm import tqdm

from summarizer.encoder import ENCODER_CONFIGS
from summarizer.selector import select_sentences
from evaluation.rouge import compute_rouge, aggregate_rouge

DATASET_CONFIG = {
    "xlsum": {"splits": ["train", "val", "test"]},
    "liputan6": {"splits": ["train", "dev", "test"]},
}


def compute_pagerank(
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    if n_nodes == 0:
        return torch.tensor([])
    if n_nodes == 1:
        return torch.tensor([1.0])

    # adj[i,j] = weight of edge from i to j
    adj = torch.zeros(n_nodes, n_nodes)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = edge_weight.float()

    # Transition matrix: M[i,j] = w(j→i) / out(j)
    # = adj[j,i] / row_sum[j] = adj.T[i,j] / row_sum[j]
    out_degree = adj.sum(dim=1)
    dangling = out_degree == 0
    out_degree[dangling] = 1.0
    M = adj.t() / out_degree.unsqueeze(0)

    pr = torch.full((n_nodes,), 1.0 / n_nodes)

    for _ in range(max_iter):
        pr_new = (1 - damping) / n_nodes + damping * (M @ pr)
        if dangling.any():
            pr_new += damping * pr[dangling].sum() / n_nodes
        if (pr_new - pr).abs().sum() < tol:
            break
        pr = pr_new

    return pr


def main():
    ap = argparse.ArgumentParser(description="Compute PageRank baseline ROUGE scores")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--encoder", default="mpnet", choices=list(ENCODER_CONFIGS.keys()))
    ap.add_argument("--threshold", default="hybrid_tadaptive")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--split", default=None, help="Specific split (default: test)")
    args = ap.parse_args()

    enc_cfg = ENCODER_CONFIGS[args.encoder]
    sbert_dir = BASE_DIR / "dataset" / args.dataset / enc_cfg["sbert_subdir"]
    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"

    threshold_str = args.threshold
    if threshold_str in ("adaptive", "tfidf"):
        graph_subfolder = threshold_str if threshold_str == "tfidf" else f"t{threshold_str}"
    elif threshold_str.startswith("hybrid"):
        graph_subfolder = threshold_str
    else:
        graph_subfolder = f"t{threshold_str}"

    splits = [args.split] if args.split else ["test"]

    all_results = {}
    for split in splits:
        prep_path = prep_dir / f"{split}.jsonl"
        graph_path = sbert_dir / "graphs" / graph_subfolder / f"{split}.pt"

        if not prep_path.exists():
            print(f"SKIP: {prep_path} not found")
            continue
        if not graph_path.exists():
            print(f"SKIP: {graph_path} not found")
            continue

        print(f"\n=== {split} ===")

        sentences_map = {}
        summary_map = {}
        with open(prep_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                sentences_map[doc["id"]] = doc["sentences"]
                summary_map[doc["id"]] = doc["summary"]

        graphs_list = torch.load(graph_path, weights_only=False)
        graph_map = {item["id"]: item for item in graphs_list}

        common_ids = set(sentences_map) & set(graph_map)

        all_rouge = []
        results = []
        for doc_id in tqdm(sorted(common_ids), desc="  PageRank"):
            sentences = sentences_map[doc_id]
            reference = summary_map[doc_id]
            n = len(sentences)
            graph = graph_map[doc_id]

            pr_scores = compute_pagerank(
                n, graph["edge_index"], graph["edge_weight"], damping=args.damping,
            )

            selection = select_sentences(sentences, pr_scores, k=args.top_k)
            rouge = compute_rouge(reference, selection.summary)
            all_rouge.append(rouge)

            results.append({
                "id": doc_id,
                "reference": reference,
                "generated": selection.summary,
                "selected_indices": selection.selected_indices,
                "rouge1": rouge["rouge1"]["f1"],
                "rouge2": rouge["rouge2"]["f1"],
                "rougeL": rouge["rougeL"]["f1"],
            })

        agg = aggregate_rouge(all_rouge)
        r1 = agg["rouge1"]["f1"]
        r2 = agg["rouge2"]["f1"]
        rL = agg["rougeL"]["f1"]

        all_results[split] = {
            "num_docs": len(results),
            "top_k": args.top_k,
            "damping": args.damping,
            "threshold": args.threshold,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rL,
        }

        print(f"\n  Documents: {len(results)}")
        print(f"  ROUGE-1 F1: {r1:.4f}")
        print(f"  ROUGE-2 F1: {r2:.4f}")
        print(f"  ROUGE-L F1: {rL:.4f}")

        results_dir = BASE_DIR / "results" / args.dataset / "pagerank"
        results_dir.mkdir(parents=True, exist_ok=True)

        summaries_path = results_dir / f"summaries_{split}_k{args.top_k}_d{args.damping}.jsonl"
        with open(summaries_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores_path = results_dir / f"scores_k{args.top_k}_d{args.damping}.json"
    scores_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
