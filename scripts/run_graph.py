import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import torch
from tqdm import tqdm

from summarizer.encoder import (
    build_graph,
    build_graph_tfidf,
    build_graph_hybrid,
    build_graph_hybrid_not_normalize,
    build_graph_hybrid_intersect,
    fit_tfidf_vectorizer,
    ENCODER_CONFIGS,
)


DATASET_CONFIG = {
    "xlsum": {"splits": ["train", "val", "test"]},
    "liputan6": {"splits": ["train", "dev", "test"]},
}


def build_graphs_cosine(
    input_path: Path,
    output_path: Path,
    threshold: float | str = 0.3,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(input_path, weights_only=False)

    results = []
    for item in tqdm(data, desc="  Building graphs"):
        edge_index, edge_weight = build_graph(item["embeddings"], threshold=threshold)
        results.append({
            "id": item["id"],
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        })

    torch.save(results, output_path)
    print(f"  Saved {len(results)} graphs to {output_path}")


def build_graphs_tfidf(
    preprocessed_path: Path,
    output_path: Path,
    vectorizer,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = []
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    results = []
    for doc in tqdm(docs, desc="  Building graphs"):
        edge_index, edge_weight = build_graph_tfidf(doc["sentences"], vectorizer)
        results.append({
            "id": doc["id"],
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        })

    torch.save(results, output_path)
    print(f"  Saved {len(results)} graphs to {output_path}")


def build_graphs_hybrid(
    emb_path: Path,
    preprocessed_path: Path,
    output_path: Path,
    vectorizer,
    threshold: float = 0.3,
    graph_fn=build_graph_hybrid,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emb_data = torch.load(emb_path, weights_only=False)
    emb_map = {item["id"]: item["embeddings"] for item in emb_data}

    docs = []
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    results = []
    for doc in tqdm(docs, desc="  Building graphs"):
        doc_id = doc["id"]
        if doc_id not in emb_map:
            continue
        edge_index, edge_weight = graph_fn(
            emb_map[doc_id], doc["sentences"], vectorizer, threshold,
        )
        results.append({
            "id": doc_id,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        })

    torch.save(results, output_path)
    print(f"  Saved {len(results)} graphs to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build graphs from SBERT embeddings or TF-IDF.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--method", choices=["cosine", "tfidf", "hybrid", "hybrid_not_normalize", "hybrid_intersect"], default="cosine")
    ap.add_argument("--threshold", default="0.3",
                     help="Cosine similarity threshold (ignored for tfidf method). Use 'adaptive' for mean-1*std.")
    ap.add_argument("--split", default=None, help="Specific split to process. If omitted, processes all.")
    ap.add_argument("--min_df", type=int, default=100,
                     help="Minimum document frequency for TF-IDF vocabulary (default: 100).")
    ap.add_argument("--encoder", default="denaya", choices=list(ENCODER_CONFIGS.keys()),
                     help="Encoder preset (default: denaya). Determines embeddings directory.")
    args = ap.parse_args()

    enc_cfg = ENCODER_CONFIGS[args.encoder]
    cfg = DATASET_CONFIG[args.dataset]
    splits = [args.split] if args.split else cfg["splits"]
    dataset_dir = BASE_DIR / "dataset" / args.dataset

    sbert_subdir = enc_cfg["sbert_subdir"]


    if args.method == "cosine":
        if args.threshold in ("adaptive", "fully_connected"):
            threshold = args.threshold
            threshold_str = args.threshold
        else:
            threshold = float(args.threshold)
            threshold_str = str(threshold).replace(".", "")

    # if args.method == "cosine":
    #     if args.threshold == "adaptive":
    #         threshold = "adaptive"
    #         threshold_str = "adaptive"
    #     else:
    #         threshold = float(args.threshold)
    #         threshold_str = str(threshold).replace(".", "")

        emb_dir = dataset_dir / sbert_subdir / "embeddings"
        out_dir = dataset_dir / sbert_subdir / "graphs" / f"t{threshold_str}"

        for split in splits:
            input_path = emb_dir / f"{split}.pt"
            if not input_path.exists():
                print(f"Skipping {split}: {input_path} not found")
                continue
            print(f"\n=== {split} ===")
            build_graphs_cosine(input_path, out_dir / f"{split}.pt", threshold)

    elif args.method == "tfidf":
        preprocessed_dir = dataset_dir / "preprocessed"
        out_dir = dataset_dir / sbert_subdir / "graphs" / "tfidf"
        vectorizer_path = dataset_dir / "vectorizers" / "tfidf_vectorizer.joblib"

        if vectorizer_path.exists():
            print(f"Loading saved TF-IDF vectorizer from {vectorizer_path}")
            vectorizer = joblib.load(vectorizer_path)
        else:
            print("Fitting TF-IDF vectorizer on training documents...")
            train_path = preprocessed_dir / "train.jsonl"
            documents = []
            with open(train_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="  Reading documents"):
                    doc = json.loads(line)
                    documents.append(doc["sentences"])
            print(f"  Total documents: {len(documents)}")

            vectorizer = fit_tfidf_vectorizer(documents, min_df=args.min_df)
            print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(vectorizer, vectorizer_path)
            print(f"  Saved vectorizer to {vectorizer_path}")

        for split in splits:
            input_path = preprocessed_dir / f"{split}.jsonl"
            if not input_path.exists():
                print(f"Skipping {split}: {input_path} not found")
                continue
            print(f"\n=== {split} ===")
            build_graphs_tfidf(input_path, out_dir / f"{split}.pt", vectorizer)

    elif args.method in ("hybrid", "hybrid_not_normalize", "hybrid_intersect"):
        preprocessed_dir = dataset_dir / "preprocessed"
        emb_dir = dataset_dir / sbert_subdir / "embeddings"
        if args.threshold in ("adaptive", "fully_connected"):
            threshold = args.threshold
            threshold_str = args.threshold
        else:
            threshold = float(args.threshold)

        # if args.threshold == "adaptive":
        #     threshold = "adaptive"
        #     threshold_str = "adaptive"
        # else:
        #     threshold = float(args.threshold)
            threshold_str = str(threshold).replace(".", "")
        # method_prefix = "hybrid_intersect" if args.method == "hybrid_intersect" else "hybrid"
        if args.method == "hybrid_intersect":
            method_prefix = "hybrid_intersect"
        elif args.method == "hybrid_not_normalize":
            method_prefix = "hybrid_not_normalize"
        else:
            method_prefix = "hybrid"
        out_dir = dataset_dir / sbert_subdir / "graphs" / f"{method_prefix}_t{threshold_str}"
        vectorizer_path = dataset_dir / "vectorizers" / "tfidf_vectorizer.joblib"

        if vectorizer_path.exists():
            print(f"Loading saved TF-IDF vectorizer from {vectorizer_path}")
            vectorizer = joblib.load(vectorizer_path)
        else:
            print("Fitting TF-IDF vectorizer on training documents...")
            train_path = preprocessed_dir / "train.jsonl"
            documents = []
            with open(train_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="  Reading documents"):
                    doc = json.loads(line)
                    documents.append(doc["sentences"])
            print(f"  Total documents: {len(documents)}")

            vectorizer = fit_tfidf_vectorizer(documents, min_df=args.min_df)
            print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(vectorizer, vectorizer_path)
            print(f"  Saved vectorizer to {vectorizer_path}")

        # graph_fn = build_graph_hybrid_intersect if args.method == "hybrid_intersect" else build_graph_hybrid

        if args.method == "hybrid_intersect":
            graph_fn = build_graph_hybrid_intersect
        elif args.method == "hybrid_not_normalize":
            graph_fn = build_graph_hybrid_not_normalize
        else:
            graph_fn = build_graph_hybrid
    

        for split in splits:
            emb_path = emb_dir / f"{split}.pt"
            prep_path = preprocessed_dir / f"{split}.jsonl"
            if not emb_path.exists():
                print(f"Skipping {split}: {emb_path} not found")
                continue
            if not prep_path.exists():
                print(f"Skipping {split}: {prep_path} not found")
                continue
            print(f"\n=== {split} ===")
            build_graphs_hybrid(emb_path, prep_path, out_dir / f"{split}.pt", vectorizer, threshold, graph_fn)


if __name__ == "__main__":
    main()
