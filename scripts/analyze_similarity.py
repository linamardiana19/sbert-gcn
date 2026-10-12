"""Analyze cosine similarity distribution across documents."""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
from tqdm import tqdm

from summarizer.encoder import ENCODER_CONFIGS


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze pairwise cosine similarity distribution.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--encoder", default="mpnet", choices=list(ENCODER_CONFIGS.keys()))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_docs", type=int, default=None, help="Limit number of documents to analyze.")
    args = ap.parse_args()

    enc_cfg = ENCODER_CONFIGS[args.encoder]
    emb_path = BASE_DIR / "dataset" / args.dataset / enc_cfg["sbert_subdir"] / "embeddings" / f"{args.split}.pt"

    print(f"Loading embeddings from {emb_path}...")
    data = torch.load(emb_path, weights_only=False)
    if args.max_docs:
        data = data[:args.max_docs]
    print(f"  {len(data)} documents")

    all_sims = []
    doc_means = []
    doc_medians = []
    doc_stds = []
    doc_mins = []
    doc_maxs = []
    doc_n_sents = []

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
    edge_counts = {t: [] for t in thresholds}
    edge_ratios = {t: [] for t in thresholds}
    adaptive_thresholds = []

    for item in tqdm(data, desc="Analyzing"):
        emb = item["embeddings"]
        n = emb.size(0)
        if n < 2:
            continue

        norm = torch.nn.functional.normalize(emb, p=2, dim=1)
        sim = (norm @ norm.t()).cpu()

        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        upper = sim[mask].numpy()

        all_sims.append(upper)
        doc_means.append(upper.mean())
        doc_medians.append(np.median(upper))
        doc_stds.append(upper.std())
        doc_mins.append(upper.min())
        doc_maxs.append(upper.max())
        doc_n_sents.append(n)

        adaptive_t = max(upper.mean() - upper.std(), 0.0)
        adaptive_thresholds.append(adaptive_t)

        total_pairs = n * (n - 1)
        for t in thresholds:
            n_edges = int((upper > t).sum()) * 2
            edge_counts[t].append(n_edges)
            edge_ratios[t].append(n_edges / total_pairs if total_pairs > 0 else 0)

        n_edges_adaptive = int((upper > adaptive_t).sum()) * 2
        edge_counts.setdefault("adaptive", []).append(n_edges_adaptive)
        edge_ratios.setdefault("adaptive", []).append(n_edges_adaptive / total_pairs if total_pairs > 0 else 0)

    all_sims_flat = np.concatenate(all_sims)
    doc_means = np.array(doc_means)
    doc_medians = np.array(doc_medians)
    doc_stds = np.array(doc_stds)
    doc_n_sents = np.array(doc_n_sents)
    adaptive_thresholds = np.array(adaptive_thresholds)

    print(f"\n{'='*60}")
    print(f"COSINE SIMILARITY DISTRIBUTION ({args.split} split)")
    print(f"{'='*60}")
    print(f"Documents analyzed: {len(doc_means)}")
    print(f"Avg sentences/doc:  {doc_n_sents.mean():.1f} (median {np.median(doc_n_sents):.0f})")

    print(f"\n--- Global (all pairwise similarities pooled) ---")
    print(f"Mean:   {all_sims_flat.mean():.4f}")
    print(f"Median: {np.median(all_sims_flat):.4f}")
    print(f"Std:    {all_sims_flat.std():.4f}")
    print(f"Min:    {all_sims_flat.min():.4f}")
    print(f"Max:    {all_sims_flat.max():.4f}")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pvals = np.percentile(all_sims_flat, percentiles)
    print(f"\nPercentiles:")
    for p, v in zip(percentiles, pvals):
        print(f"  P{p:2d}: {v:.4f}")

    print(f"\n--- Per-document statistics (mean across docs) ---")
    print(f"Mean of doc means:   {doc_means.mean():.4f} (std {doc_means.std():.4f})")
    print(f"Mean of doc medians: {doc_medians.mean():.4f}")
    print(f"Mean of doc stds:    {doc_stds.mean():.4f}")

    print(f"\n--- Adaptive threshold (mean - std per doc) ---")
    print(f"Mean:   {adaptive_thresholds.mean():.4f}")
    print(f"Median: {np.median(adaptive_thresholds):.4f}")
    print(f"Std:    {adaptive_thresholds.std():.4f}")
    print(f"Min:    {adaptive_thresholds.min():.4f}")
    print(f"Max:    {adaptive_thresholds.max():.4f}")

    print(f"\n--- Edge density per threshold ---")
    print(f"{'Threshold':<12} {'Avg Edges':<12} {'Edge Ratio':<12} {'Docs w/ 0 edges'}")
    for t in thresholds + ["adaptive"]:
        counts = np.array(edge_counts[t])
        ratios = np.array(edge_ratios[t])
        zero_edge = int((counts == 0).sum())
        label = str(t) if isinstance(t, str) else f"{t:.1f}"
        print(f"{label:<12} {counts.mean():<12.1f} {ratios.mean():<12.4f} {zero_edge}")

    print(f"\n--- Pairs below each threshold (%) ---")
    for t in thresholds:
        pct = (all_sims_flat <= t).mean() * 100
        print(f"  sim <= {t:.1f}: {pct:.1f}%")


if __name__ == "__main__":
    main()
