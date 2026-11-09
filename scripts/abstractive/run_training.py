"""
Training IndoT5 abstractive summarization (baseline atau + GCN).

Baseline IndoT5 (tanpa graf):
    uv run python scripts/abstractive/run_training.py --dataset xlsum

IndoT5 + GCN (memakai graf hybrid_tadaptive dari pipeline extractive):
    uv run python scripts/abstractive/run_training.py --dataset xlsum --use_gcn \
        --encoder mpnet --graph_name hybrid_tadaptive

Prasyarat:
    scripts/run_preprocessing.py --dataset xlsum            (selalu)
    scripts/run_encoding.py + scripts/run_graph.py          (hanya untuk --use_gcn)

Output:
    models/Wikidepia-IndoT5-base/                           (model dasar, sekali unduh)
    dataset/xlsum/abstractive/tokenized/<tag>/              (cache tokenisasi)
    checkpoints/xlsum/abstractive/<scenario>/best/ last/ train.log history.jsonl
"""

# ------------------------------------------------------------
# GPU harus diset sebelum import torch (sama seperti run_training.py lama)
# ------------------------------------------------------------
import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="1")
_pre_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import json
import math
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from abstractive import paths
from abstractive.collator import GraphSeq2SeqCollator, move_batch
from abstractive.engine import autocast_ctx, build_split_dataset, compute_loss, generate_and_score
from abstractive.model import IndoT5GCN, load_base_model_and_tokenizer
from summarizer.encoder import ENCODER_CONFIGS


def _lr_to_scientific(lr: float) -> str:
    if lr == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / (10 ** exp)
    return f"1e{exp}" if mantissa == 1.0 else f"{mantissa:g}e{exp}"


def build_scenario_name(args) -> str:
    name = (
        f"indot5_{'gcn' if args.use_gcn else 'base'}_{args.epochs}ep_{_lr_to_scientific(args.lr)}"
        f"_bs{args.batch_size}x{args.grad_accum}_src{args.max_source_len}"
    )
    if args.model_name != paths.DEFAULT_MODEL:
        name += f"_M{paths.model_slug(args.model_name)}"
    if args.use_gcn:
        name += f"_E{args.encoder}_g{args.graph_name}_L{args.gcn_layers}_h{args.gcn_hidden}"
        if args.use_sbert:
            name += "_sbert"
        if args.pos_dim > 0:
            name += f"_Psin{args.pos_dim}"
    if args.run_tag:
        name += f"_{args.run_tag}"
    return name


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train IndoT5 (+GCN) abstractive summarizer.")
    ap.add_argument("--dataset", choices=list(paths.DATASET_SPLITS), required=True)
    ap.add_argument("--gpu", default="1", help="Nilai CUDA_VISIBLE_DEVICES (default: 1, sama dengan script lama).")
    ap.add_argument("--model_name", default=paths.DEFAULT_MODEL, help="Nama HF atau path lokal. Disimpan ke models/ saat pertama kali.")

    # graf / GCN
    ap.add_argument("--use_gcn", action="store_true", help="Aktifkan sentence-graph fusion. Tanpa flag = baseline IndoT5.")
    ap.add_argument("--encoder", default="mpnet", choices=list(ENCODER_CONFIGS), help="Menentukan folder graf/embedding (sbert_<encoder>).")
    ap.add_argument("--graph_name", default="hybrid_tadaptive", help="Subfolder di dataset/<ds>/sbert_<encoder>/graphs/.")
    ap.add_argument("--use_sbert", action="store_true", help="Tempel embedding SBERT ke fitur node GCN.")
    ap.add_argument("--pos_dim", type=int, default=0, help="Positional encoding sinusoidal kalimat untuk node GCN (0 = mati).")
    ap.add_argument("--gcn_hidden", type=int, default=256)
    ap.add_argument("--gcn_layers", type=int, default=2)
    ap.add_argument("--gcn_dropout", type=float, default=0.1)
    ap.add_argument("--gcn_residual", choices=["outer", "none"], default="outer")

    # data
    ap.add_argument("--prefix", default="ringkas: ")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah dokumen (untuk uji coba cepat).")
    ap.add_argument("--num_workers", type=int, default=2)

    # optimisasi
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate IndoT5.")
    ap.add_argument("--gcn_lr", type=float, default=1e-3, help="Learning rate modul GCN/fusion (diinisialisasi acak).")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32", help="bf16 lebih hemat VRAM (GPU Ampere+). Hindari fp16 untuk T5.")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # validasi / generate
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--val_gen_limit", type=int, default=1000, help="Jumlah dokumen val untuk ROUGE tiap epoch (0 = semua).")
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--selection_metric", choices=["rouge1", "rouge2", "rougeL"], default="rouge1")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping. 0 = nonaktif.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Timpa skenario yang sudah ada tanpa bertanya.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    splits = paths.DATASET_SPLITS[args.dataset]
    scenario_name = build_scenario_name(args)
    scenario_dir = paths.checkpoints_root(args.dataset) / scenario_name

    if scenario_dir.exists() and any(scenario_dir.iterdir()) and not args.overwrite:
        response = input(f"Scenario '{scenario_name}' already exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return
    scenario_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(scenario_dir / "train.log", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Scenario: {scenario_name}")
        print(f"Device: {device} | precision: {args.precision}")
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))

        # ---------------- model dasar ----------------
        print("\nLoading IndoT5...")
        t5, tokenizer = load_base_model_and_tokenizer(args.model_name)
        if args.gradient_checkpointing:
            t5.gradient_checkpointing_enable()
            t5.config.use_cache = False

        sbert_dim = ENCODER_CONFIGS[args.encoder]["dim"]
        node_extra_dim = (sbert_dim if args.use_sbert else 0) + args.pos_dim
        fusion_cfg = {
            "node_extra_dim": node_extra_dim,
            "gcn_hidden": args.gcn_hidden,
            "num_layers": args.gcn_layers,
            "dropout": args.gcn_dropout,
            "residual": args.gcn_residual,
        }
        model = IndoT5GCN(t5, use_gcn=args.use_gcn, fusion_cfg=fusion_cfg).to(device)

        # ---------------- data ----------------
        data_cfg = {
            "model_name": args.model_name,
            "prefix": args.prefix,
            "max_source_len": args.max_source_len,
            "max_target_len": args.max_target_len,
            "use_gcn": args.use_gcn,
            "sbert_subdir": ENCODER_CONFIGS[args.encoder]["sbert_subdir"],
            "graph_name": args.graph_name,
            "use_sbert": args.use_sbert,
            "sbert_dim": sbert_dim,
            "pos_dim": args.pos_dim,
        }
        print("\nLoading training data...")
        train_ds = build_split_dataset(args.dataset, splits["train"], tokenizer, data_cfg, args.limit)
        print(f"  {len(train_ds)} training documents")
        print("Loading validation data...")
        val_ds = build_split_dataset(args.dataset, splits["val"], tokenizer, data_cfg, args.limit)
        print(f"  {len(val_ds)} validation documents")

        if args.val_gen_limit and args.val_gen_limit < len(val_ds):
            idx = sorted(random.Random(args.seed).sample(range(len(val_ds)), args.val_gen_limit))
            val_gen_ds = Subset(val_ds, idx)
        else:
            val_gen_ds = val_ds
        print(f"  {len(val_gen_ds)} validation documents dipakai untuk ROUGE per epoch")

        collator = GraphSeq2SeqCollator(pad_token_id=tokenizer.pad_token_id)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collator, num_workers=args.num_workers)
        val_gen_loader = DataLoader(val_gen_ds, batch_size=args.eval_batch_size, shuffle=False,
                                    collate_fn=collator, num_workers=args.num_workers)

        # ---------------- optimizer ----------------
        param_groups = [{"params": [p for p in model.t5.parameters() if p.requires_grad], "lr": args.lr}]
        if args.use_gcn:
            param_groups.append({"params": list(model.fusion.parameters()), "lr": args.gcn_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

        n_t5 = sum(p.numel() for p in model.t5.parameters())
        n_fusion = sum(p.numel() for p in model.fusion.parameters()) if args.use_gcn else 0
        print(f"\nParameters: IndoT5={n_t5:,} | GCN+fusion={n_fusion:,}")
        print(f"Optimizer steps: {total_steps} ({steps_per_epoch}/epoch)")

        gen_kwargs = {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "length_penalty": args.length_penalty,
            "early_stopping": True,
        }
        save_cfg = {
            "dataset": args.dataset,
            "data": data_cfg,
            "generation": gen_kwargs,
            "train_args": vars(args),
            "scenario": scenario_name,
        }

        # ---------------- training loop ----------------
        best_score = -1.0
        no_improve = 0
        history_path = scenario_dir / "history.jsonl"
        history_path.write_text("", encoding="utf-8")
        metric_label = args.selection_metric.upper().replace("ROUGE", "ROUGE-")

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n_batches, gate_sum = 0.0, 0, 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch}/{args.epochs}")):
                batch = move_batch(batch, device)
                with autocast_ctx(device, args.precision):
                    out = model(batch)
                loss = out.loss
                if torch.isnan(loss):
                    continue
                (loss / args.grad_accum).backward()
                total_loss += loss.item()
                n_batches += 1
                if args.use_gcn:
                    gate_sum += model.fusion.last_gate_mean

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            train_loss = total_loss / max(n_batches, 1)
            val_loss = compute_loss(model, val_loader, device, args.precision)
            _, val_rouge = generate_and_score(model, val_gen_loader, tokenizer, device, gen_kwargs, args.precision, desc="  Val ROUGE")

            r1, r2, rl = (val_rouge[k]["f1"] for k in ("rouge1", "rouge2", "rougeL"))
            gate_info = f" gate={gate_sum / max(n_batches, 1):.3f}" if args.use_gcn else ""
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"R1={r1:.4f} R2={r2:.4f} RL={rl:.4f}{gate_info}")

            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "rouge1": r1, "rouge2": r2, "rougeL": rl,
                    "gate_mean": gate_sum / max(n_batches, 1) if args.use_gcn else None,
                }) + "\n")

            score = val_rouge[args.selection_metric]["f1"]
            if score > best_score:
                best_score = score
                no_improve = 0
                model.save(scenario_dir / "best", tokenizer, {**save_cfg, "epoch": epoch, "best_score": score})
                print(f"  Saved best model ({metric_label}: {score:.4f})")
            else:
                no_improve += 1
                if args.patience > 0:
                    print(f"  No improvement for {no_improve}/{args.patience} epochs")
                    if no_improve >= args.patience:
                        print(f"  Early stopping triggered at epoch {epoch}")
                        break

        model.save(scenario_dir / "last", tokenizer, {**save_cfg, "epoch": epoch, "best_score": best_score})
        print(f"  Saved last model (epoch {epoch})")
        print(f"\nTraining complete. Best {metric_label}: {best_score:.4f}")
        print(f"Checkpoints: {scenario_dir}")
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()
