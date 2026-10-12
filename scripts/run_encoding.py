import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from tqdm import tqdm

from summarizer.encoder import SBERTEncoder, ENCODER_CONFIGS

DATASET_CONFIG = {
    "xlsum": {"splits": ["train", "val", "test"]},
    "liputan6": {"splits": ["train", "dev", "test"]},
}

CHECKPOINT_INTERVAL = 1000


def encode_split(
    encoder: SBERTEncoder,
    input_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".ckpt")

    # Resume: load checkpoint jika ada
    done_ids: set[str] = set()
    results: list[dict] = []
    if checkpoint_path.exists():
        results = torch.load(checkpoint_path, weights_only=False)
        done_ids = {r["id"] for r in results}
        print(f"  Resumed from checkpoint: {len(results)} documents")

    with open(input_path, encoding="utf-8") as f:
        docs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    # Filter dokumen yang sudah di-encode
    remaining = [d for d in docs if d["id"] not in done_ids]

    if not remaining:
        print(f"  All {len(docs)} documents already encoded")
        if checkpoint_path.exists():
            torch.save(results, output_path)
            checkpoint_path.unlink()
            print(f"  Finalized: {output_path}")
        return

    print(f"  Encoding {len(remaining)} documents ({len(done_ids)} already done)")

    for i, doc in enumerate(tqdm(remaining, desc="  Encoding")):
        embeddings = encoder.encode(doc["sentences"])
        results.append({
            "id": doc["id"],
            "embeddings": embeddings.cpu(),
        })

        # Save checkpoint setiap CHECKPOINT_INTERVAL dokumen
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(results, checkpoint_path)

    # Save final
    torch.save(results, output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"  Saved {len(results)} documents to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode sentences with SBERT.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--encoder", default="denaya", choices=list(ENCODER_CONFIGS.keys()),
                     help="Encoder preset (default: denaya). Determines model and output directory.")
    ap.add_argument("--split", default=None, help="Specific split to process. If omitted, processes all.")
    ap.add_argument("--device", default=None, help="Device to use, e.g. cpu, cuda, cuda:0, cuda:1, cuda:2, cuda:3.")
    args = ap.parse_args()

    enc_cfg = ENCODER_CONFIGS[args.encoder]
    cfg = DATASET_CONFIG[args.dataset]
    splits = [args.split] if args.split else cfg["splits"]
    prep_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"
    out_dir = BASE_DIR / "dataset" / args.dataset / enc_cfg["sbert_subdir"] / "embeddings"

    print(f"Loading SBERT model: {enc_cfg['model']} (dim={enc_cfg['dim']})...")
    # encoder = SBERTEncoder(model_name=enc_cfg["model"])
    encoder = SBERTEncoder(model_name=enc_cfg["model"], device=args.device)

    print(f"Device: {encoder.device}")

    for split in splits:
        input_path = prep_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue

        print(f"\n=== {split} ===")
        encode_split(encoder, input_path, out_dir / f"{split}.pt")


if __name__ == "__main__":
    main()
