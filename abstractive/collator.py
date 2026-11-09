"""Batching token + graf kalimat.

Graf beberapa dokumen digabung menjadi satu graf besar (gaya PyTorch Geometric):
node dokumen ke-b digeser sebesar ``node_offsets[b]``. ``sent_index`` menyimpan
indeks kalimat lokal untuk setiap token (-1 = prefix, </s>, atau padding).
"""

from __future__ import annotations

import torch


class GraphSeq2SeqCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, items: list[dict]) -> dict:
        bsz = len(items)
        max_src = max(it["input_ids"].size(0) for it in items)
        max_tgt = max(it["labels"].size(0) for it in items)

        input_ids = torch.full((bsz, max_src), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_src), dtype=torch.long)
        sent_index = torch.full((bsz, max_src), -1, dtype=torch.long)
        labels = torch.full((bsz, max_tgt), self.label_pad_id, dtype=torch.long)

        for b, it in enumerate(items):
            n_src = it["input_ids"].size(0)
            n_tgt = it["labels"].size(0)
            input_ids[b, :n_src] = it["input_ids"]
            attention_mask[b, :n_src] = 1
            sent_index[b, :n_src] = it["sent_ids"]
            labels[b, :n_tgt] = it["labels"]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sent_index": sent_index,
            "ids": [it["id"] for it in items],
            "references": [it["summary"] for it in items],
        }

        if "edge_index" in items[0]:
            n_nodes = torch.tensor([it["n_nodes"] for it in items], dtype=torch.long)
            node_offsets = torch.zeros(bsz, dtype=torch.long)
            if bsz > 1:
                node_offsets[1:] = torch.cumsum(n_nodes, dim=0)[:-1]

            edge_index = torch.cat(
                [it["edge_index"] + node_offsets[b] for b, it in enumerate(items)], dim=1
            )
            edge_weight = torch.cat([it["edge_weight"] for it in items], dim=0)

            batch["node_offsets"] = node_offsets
            batch["num_nodes"] = int(n_nodes.sum().item())
            batch["edge_index"] = edge_index
            batch["edge_weight"] = edge_weight
            if "node_extra" in items[0]:
                batch["node_extra"] = torch.cat([it["node_extra"] for it in items], dim=0)

        return batch


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
