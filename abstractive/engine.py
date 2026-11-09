"""Bagian yang dipakai bersama oleh training & evaluasi.

- build_split_dataset : preprocessed + cache token + (opsional) graf/SBERT -> AbstractiveDataset
- autocast_ctx        : bf16 opsional
- compute_loss        : rata-rata loss pada satu loader
- generate_and_score  : generate ringkasan + ROUGE (evaluation.rouge milik project)
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from abstractive import paths
from abstractive.collator import move_batch
from abstractive.data import (
    AbstractiveDataset,
    load_graph_map,
    load_sbert_map,
    load_tokenized_split,
)
from evaluation.rouge import aggregate_rouge, compute_rouge


def build_split_dataset(dataset: str, split: str, tokenizer, data_cfg: dict, limit: int | None = None) -> AbstractiveDataset:
    """data_cfg berisi: model_name, prefix, max_source_len, max_target_len,
    use_gcn, sbert_subdir, graph_name, use_sbert, sbert_dim, pos_dim."""
    prep_path = paths.preprocessed_dir(dataset) / f"{split}.jsonl"
    if not prep_path.exists():
        raise FileNotFoundError(f"{prep_path} tidak ada. Jalankan scripts/run_preprocessing.py --dataset {dataset}")

    cache_path = paths.token_cache_dir(
        dataset, data_cfg["model_name"], data_cfg["max_source_len"], data_cfg["max_target_len"], data_cfg["prefix"]
    ) / f"{split}.pt"
    examples = load_tokenized_split(
        prep_path, cache_path, tokenizer, data_cfg["prefix"], data_cfg["max_source_len"], data_cfg["max_target_len"]
    )
    if limit is not None:
        examples = examples[:limit]

    graph_map = sbert_map = None
    if data_cfg["use_gcn"]:
        gdir = paths.graph_dir(dataset, data_cfg["sbert_subdir"], data_cfg["graph_name"])
        graph_map = load_graph_map(gdir / f"{split}.pt")
        if data_cfg.get("use_sbert"):
            sbert_map = load_sbert_map(paths.embeddings_dir(dataset, data_cfg["sbert_subdir"]) / f"{split}.pt")

    ds = AbstractiveDataset(
        examples,
        graph_map=graph_map,
        sbert_map=sbert_map,
        sbert_dim=data_cfg.get("sbert_dim", 768),
        pos_dim=data_cfg.get("pos_dim", 0) if data_cfg["use_gcn"] else 0,
    )
    if graph_map is not None and ds.n_missing_graph:
        print(f"  Catatan: {ds.n_missing_graph} dokumen {split} tidak punya graf (dipakai tanpa edge).")
    return ds


def autocast_ctx(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def compute_loss(model, loader, device: torch.device, precision: str = "fp32") -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  Val loss", leave=False):
        batch = move_batch(batch, device)
        with autocast_ctx(device, precision):
            out = model(batch)
        total += out.loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def generate_and_score(model, loader, tokenizer, device: torch.device, gen_kwargs: dict, precision: str = "fp32", desc: str = "  Generating") -> tuple[list[dict], dict]:
    model.eval()
    rows: list[dict] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch(batch, device)
        with autocast_ctx(device, precision):
            out_ids = model.generate(batch, **gen_kwargs)
        preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        for doc_id, ref, pred in zip(batch["ids"], batch["references"], preds):
            pred = pred.strip()
            scores = compute_rouge(ref, pred)
            rows.append({
                "id": doc_id,
                "reference": ref,
                "generated": pred,
                "rouge1": scores["rouge1"]["f1"],
                "rouge2": scores["rouge2"]["f1"],
                "rougeL": scores["rougeL"]["f1"],
                "_full_rouge": scores,
            })
    agg = aggregate_rouge([r["_full_rouge"] for r in rows])
    return rows, agg
