"""
Visualize sentence graph construction result.

Output:
    results/<dataset>/graph_visualization/<graph_method>/
        graph_<split>_doc<doc_index>.png
        graph_<split>_doc<doc_index>_sentences.txt

Example:
    uv run python scripts/visualize_graph.py \
        --dataset xlsum \
        --encoder mpnet \
        --threshold hybrid_tadaptive \
        --split val \
        --doc_index 0 \
        --max_edges 80
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from summarizer.encoder import ENCODER_CONFIGS


def resolve_graph_subfolder(threshold: str) -> str:
    """
    Convert threshold argument into graph folder name.
    """
    if threshold == "tfidf":
        return "tfidf"

    if threshold == "adaptive":
        return "tadaptive"

    if threshold.startswith("hybrid"):
        return threshold

    return f"t{threshold}"


def load_preprocessed(prep_path: Path):
    """
    Load sentences and summaries from preprocessed jsonl file.
    """
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

    return sentences_map, summary_map


def build_networkx_graph(sentences, edge_index, edge_weight, max_edges: int):
    """
    Build NetworkX directed graph from edge_index and edge_weight.
    Only strongest edges are visualized to keep the graph readable.
    """
    graph = nx.DiGraph()

    for idx, sentence in enumerate(sentences):
        graph.add_node(idx, label=f"S{idx}", text=sentence)

    edges = []
    for edge_id in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        weight = float(edge_weight[edge_id].item())
        edges.append((src, dst, weight))

    edges = sorted(edges, key=lambda item: item[2], reverse=True)
    edges = edges[:max_edges]

    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight=weight)

    return graph, edges


def save_graph_image(graph, output_path: Path, title: str):
    """
    Save graph visualization as PNG.
    """
    plt.figure(figsize=(14, 9))

    pos = nx.spring_layout(graph, seed=42, weight="weight")

    node_labels = {node: f"S{node}" for node in graph.nodes()}
    edge_labels = {
        (src, dst): f"{data['weight']:.2f}"
        for src, dst, data in graph.edges(data=True)
    }

    nx.draw_networkx_nodes(graph, pos, node_size=900)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=9)
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.4)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_sentence_detail(
    output_path: Path,
    doc_id: str,
    dataset: str,
    encoder: str,
    graph_method: str,
    split: str,
    doc_index: int,
    sentences,
    reference_summary: str,
    edges,
):
    """
    Save sentence list and visualized edge list as text file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"DOC ID: {doc_id}\n")
        f.write(f"DATASET: {dataset}\n")
        f.write(f"ENCODER: {encoder}\n")
        f.write(f"GRAPH METHOD: {graph_method}\n")
        f.write(f"SPLIT: {split}\n")
        f.write(f"DOC INDEX: {doc_index}\n")
        f.write(f"NUM SENTENCES: {len(sentences)}\n")
        f.write(f"NUM EDGES VISUALIZED: {len(edges)}\n\n")

        f.write("REFERENCE SUMMARY:\n")
        f.write(reference_summary + "\n\n")

        f.write("SENTENCES:\n")
        for idx, sentence in enumerate(sentences):
            f.write(f"S{idx}: {sentence}\n")

        f.write("\nEDGES VISUALIZED:\n")
        for src, dst, weight in edges:
            f.write(f"S{src} -> S{dst} | weight={weight:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize sentence graph construction result"
    )

    parser.add_argument("--dataset", choices=["xlsum", "liputan6"], default="xlsum")
    parser.add_argument("--encoder", choices=list(ENCODER_CONFIGS.keys()), default="mpnet")
    parser.add_argument("--threshold", default="hybrid_tadaptive")
    parser.add_argument("--split", default="val")
    parser.add_argument("--doc_index", type=int, default=0)
    parser.add_argument("--max_edges", type=int, default=80)

    return parser.parse_args()


def main():
    args = parse_args()

    encoder_config = ENCODER_CONFIGS[args.encoder]
    graph_method = resolve_graph_subfolder(args.threshold)

    sbert_dir = BASE_DIR / "dataset" / args.dataset / encoder_config["sbert_subdir"]
    prep_path = BASE_DIR / "dataset" / args.dataset / "preprocessed" / f"{args.split}.jsonl"
    graph_path = sbert_dir / "graphs" / graph_method / f"{args.split}.pt"

    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {prep_path}")

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    sentences_map, summary_map = load_preprocessed(prep_path)
    graphs = torch.load(graph_path, weights_only=False)

    if args.doc_index >= len(graphs):
        raise IndexError(
            f"doc_index {args.doc_index} is out of range. "
            f"Total documents: {len(graphs)}"
        )

    graph_item = graphs[args.doc_index]
    doc_id = graph_item["id"]

    if doc_id not in sentences_map:
        raise KeyError(f"Document ID not found in preprocessed file: {doc_id}")

    sentences = sentences_map[doc_id]
    reference_summary = summary_map[doc_id]
    edge_index = graph_item["edge_index"]
    edge_weight = graph_item["edge_weight"]

    graph, edges = build_networkx_graph(
        sentences=sentences,
        edge_index=edge_index,
        edge_weight=edge_weight,
        max_edges=args.max_edges,
    )

    output_dir = (
        BASE_DIR
        / "results"
        / args.dataset
        / "graph_visualization"
        / graph_method
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"graph_{args.split}_doc{args.doc_index}.png"
    text_path = output_dir / f"graph_{args.split}_doc{args.doc_index}_sentences.txt"

    title = (
        f"Sentence Graph | {args.dataset} | {graph_method} | "
        f"{args.split} | doc_index={args.doc_index}"
    )

    save_graph_image(graph, image_path, title)

    save_sentence_detail(
        output_path=text_path,
        doc_id=doc_id,
        dataset=args.dataset,
        encoder=args.encoder,
        graph_method=graph_method,
        split=args.split,
        doc_index=args.doc_index,
        sentences=sentences,
        reference_summary=reference_summary,
        edges=edges,
    )

    print("Graph visualization saved.")
    print(f"Image file   : {image_path}")
    print(f"Detail file  : {text_path}")
    print(f"Document ID  : {doc_id}")
    print(f"Sentences    : {len(sentences)}")
    print(f"Edges shown  : {len(edges)}")


if __name__ == "__main__":
    main()