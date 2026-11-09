"""Data untuk IndoT5: tokenisasi per kalimat + pemetaan token -> kalimat.

Ide utama: teks di ``preprocessed/*.jsonl`` sudah dipecah per kalimat.
Setiap kalimat ditokenisasi terpisah lalu disambung, sehingga untuk setiap
token kita tahu ia berasal dari kalimat ke berapa (``sent_ids``).
Indeks kalimat ini SAMA dengan indeks node pada graf hasil scripts/run_graph.py,
jadi graf yang sudah ada bisa langsung dipakai.

Format input encoder:
    [prefix tokens] [kalimat 0] [kalimat 1] ... [kalimat k] </s>
    sent_ids:  -1 ...  0 0 0     1 1 1       ...  k k        -1

Kalimat yang tidak muat di ``max_source_len`` dibuang, begitu juga node dan
edge-nya di graf (subgraf), supaya GCN hanya melihat teks yang dibaca T5.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from summarizer.positional import make_position_encoding


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# TOKENISASI
# ============================================================

def tokenize_document(
    sentences: list[str],
    summary: str,
    tokenizer,
    prefix: str,
    max_source_len: int,
    max_target_len: int,
    min_partial_tokens: int = 4,
) -> dict:
    eos_id = tokenizer.eos_token_id
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []

    input_ids: list[int] = list(prefix_ids)
    sent_ids: list[int] = [-1] * len(prefix_ids)
    limit = max_source_len - 1  # sisakan 1 slot untuk </s>

    sent_token_ids = tokenizer(sentences, add_special_tokens=False)["input_ids"] if sentences else []

    n_nodes = 0
    truncated = False
    for ids in sent_token_ids:
        room = limit - len(input_ids)
        if room <= 0:
            truncated = True
            break
        if len(ids) > room:
            # kalimat terakhir terpotong: tetap dipakai kalau sisa token cukup
            if room >= min_partial_tokens:
                input_ids.extend(ids[:room])
                sent_ids.extend([n_nodes] * room)
                n_nodes += 1
            truncated = True
            break
        input_ids.extend(ids)
        sent_ids.extend([n_nodes] * len(ids))
        n_nodes += 1

    input_ids.append(eos_id)
    sent_ids.append(-1)

    labels = tokenizer(text_target=summary, max_length=max_target_len, truncation=True)["input_ids"]

    return {
        "input_ids": input_ids,
        "sent_ids": sent_ids,
        "n_nodes": n_nodes,
        "n_sentences": len(sentences),
        "truncated": truncated,
        "labels": labels,
    }


def load_tokenized_split(
    prep_path: Path,
    cache_path: Path,
    tokenizer,
    prefix: str,
    max_source_len: int,
    max_target_len: int,
) -> list[dict]:
    """Tokenisasi satu split, dengan cache ke disk (sekali saja per konfigurasi)."""
    if cache_path.exists():
        print(f"  Loading tokenized cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    docs = read_jsonl(prep_path)
    examples = []
    for doc in tqdm(docs, desc=f"  Tokenizing {prep_path.stem}"):
        sentences = doc.get("sentences", [])
        if not sentences or not doc.get("summary"):
            continue
        ex = tokenize_document(sentences, doc["summary"], tokenizer, prefix, max_source_len, max_target_len)
        if ex["n_nodes"] == 0:
            continue
        ex["id"] = str(doc["id"])
        ex["summary"] = doc["summary"]
        examples.append(ex)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, cache_path)
    n_trunc = sum(e["truncated"] for e in examples)
    print(f"  Saved {len(examples)} examples to {cache_path} ({n_trunc} terpotong oleh max_source_len)")
    return examples


# ============================================================
# GRAF & EMBEDDING (hasil pipeline extractive)
# ============================================================

def load_graph_map(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}. Jalankan scripts/run_graph.py terlebih dahulu.")
    items = torch.load(path, weights_only=False)
    return {str(it["id"]): (it["edge_index"], it.get("edge_weight")) for it in items}


def load_sbert_map(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}. Jalankan scripts/run_encoding.py terlebih dahulu.")
    items = torch.load(path, weights_only=False)
    return {str(it["id"]): it["embeddings"] for it in items}


def restrict_graph(edge_index: torch.Tensor, edge_weight: torch.Tensor | None, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ambil subgraf untuk node 0..n_nodes-1 (kalimat yang muat di input T5)."""
    edge_index = edge_index.long()
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    mask = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
    return edge_index[:, mask], edge_weight[mask].float()


# ============================================================
# DATASET
# ============================================================

class AbstractiveDataset(Dataset):
    """Satu item = token dokumen + (opsional) subgraf kalimat dan fitur node tambahan.

    - graph_map None  -> mode baseline IndoT5 (tanpa graf)
    - sbert_map       -> embedding SBERT ditempel ke fitur node GCN
    - pos_dim > 0     -> positional encoding sinusoidal per kalimat (summarizer.positional)
    """

    def __init__(
        self,
        examples: list[dict],
        graph_map: dict | None = None,
        sbert_map: dict | None = None,
        sbert_dim: int = 768,
        pos_dim: int = 0,
    ):
        self.examples = examples
        self.graph_map = graph_map
        self.sbert_map = sbert_map
        self.sbert_dim = sbert_dim
        self.pos_dim = pos_dim
        self.n_missing_graph = 0
        if graph_map is not None:
            self.n_missing_graph = sum(1 for e in examples if e["id"] not in graph_map)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        n = ex["n_nodes"]
        item = {
            "id": ex["id"],
            "summary": ex["summary"],
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "sent_ids": torch.tensor(ex["sent_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "n_nodes": n,
        }

        if self.graph_map is not None:
            # dokumen tanpa graf -> tanpa edge (GCNConv tetap menambah self-loop)
            ei, ew = self.graph_map.get(ex["id"], (torch.zeros(2, 0, dtype=torch.long), None))
            item["edge_index"], item["edge_weight"] = restrict_graph(ei, ew, n)

            extras = []
            if self.sbert_map is not None:
                emb = self.sbert_map.get(ex["id"])
                feat = torch.zeros(n, self.sbert_dim)
                if emb is not None:
                    m = min(n, emb.size(0))
                    feat[:m] = emb[:m].float()
                extras.append(feat)
            if self.pos_dim > 0:
                extras.append(make_position_encoding(n, self.pos_dim))
            if extras:
                item["node_extra"] = torch.cat(extras, dim=-1)

        return item
