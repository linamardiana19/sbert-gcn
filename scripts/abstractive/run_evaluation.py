"""
Evaluasi IndoT5 (+GCN) pada split test/val dengan ROUGE.

    uv run python scripts/abstractive/run_evaluation.py \
        --checkpoint checkpoints/xlsum/abstractive/<scenario>/best

Konfigurasi data (graf, prefix, panjang token, dll.) dibaca dari
<checkpoint>/abstractive_config.json, jadi tidak perlu diulang di CLI.

Output (format sama dengan scripts/run_evaluation.py):
    results/<ds>/abstractive/<scenario>/summaries_<best|last>_<split>[_bN].jsonl
    results/<ds>/abstractive/<scenario>/scores_<best|last>_<split>[_bN].json
"""

import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="1")
_pre_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import torch
from torch.utils.data import DataLoader

from abstractive import paths
from abstractive.collator import GraphSeq2SeqCollator
from abstractive.engine import build_split_dataset, generate_and_score
from abstractive.model import IndoT5GCN


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate IndoT5 (+GCN) abstractive summarizer with ROUGE.")
    ap.add_argument("--checkpoint", required=True, help="Folder checkpoint, mis. checkpoints/xlsum/abstractive/<scenario>/best")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--split", default="test", choices=["train", "val", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    # override parameter generate (default: yang dipakai saat training)
    ap.add_argument("--num_beams", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=None)
    ap.add_argument("--length_penalty", type=float, default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_absolute():
        ckpt_dir = BASE_DIR / ckpt_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model, tokenizer, cfg = IndoT5GCN.load(ckpt_dir, device)
    dataset = cfg["dataset"]
    print(f"  Scenario: {cfg.get('scenario')} | epoch {cfg.get('epoch')} | use_gcn={cfg['use_gcn']}")

    gen_kwargs = dict(cfg.get("generation", {}))
    for key in ("num_beams", "max_new_tokens", "no_repeat_ngram_size", "length_penalty"):
        if getattr(args, key) is not None:
            gen_kwargs[key] = getattr(args, key)
    print(f"  Generation: {gen_kwargs}")

    split = args.split
    if split == "val" and dataset == "liputan6":
        split = "dev"
    print(f"Loading {split} data...")
    ds = build_split_dataset(dataset, split, tokenizer, cfg["data"], args.limit)
    print(f"  {len(ds)} documents")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=GraphSeq2SeqCollator(tokenizer.pad_token_id), num_workers=args.num_workers)

    rows, agg = generate_and_score(model, loader, tokenizer, device, gen_kwargs, args.precision, desc="Evaluating")

    agg["num_documents"] = len(rows)
    agg["checkpoint"] = str(ckpt_dir)
    agg["split"] = split
    agg["use_gcn"] = cfg["use_gcn"]
    agg["generation"] = gen_kwargs
    agg["avg_generated_words"] = sum(len(r["generated"].split()) for r in rows) / max(len(rows), 1)

    scenario_name = ckpt_dir.parent.name
    model_type = ckpt_dir.name
    eval_dir = paths.results_root(dataset) / scenario_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    beam_suffix = f"_b{gen_kwargs['num_beams']}" if gen_kwargs.get("num_beams", 4) != 4 else ""
    stem = f"{model_type}_{split}{beam_suffix}"

    summaries_path = eval_dir / f"summaries_{stem}.jsonl"
    with open(summaries_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_full_rouge"}, ensure_ascii=False) + "\n")
    (eval_dir / f"scores_{stem}.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEvaluated {len(rows)} documents")
    print(f"ROUGE-1 F1: {agg['rouge1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {agg['rouge2']['f1']:.4f}")
    print(f"ROUGE-L F1: {agg['rougeL']['f1']:.4f}")
    print(f"Avg generated words: {agg['avg_generated_words']:.1f}")
    print(f"Results saved to {eval_dir}")


if __name__ == "__main__":
    main()
