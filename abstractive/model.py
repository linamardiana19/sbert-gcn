"""IndoT5GCN: T5ForConditionalGeneration + (opsional) SentenceGraphFusion.

- use_gcn=False -> baseline IndoT5 murni (fusion tidak dibuat)
- use_gcn=True  -> output encoder diperkaya graf kalimat sebelum masuk decoder

Model dasar dimuat dari ``models/<org>-<nama>`` bila ada; bila belum, diunduh
dari Hugging Face lalu disimpan ke sana (konvensi sama dengan summarizer/encoder.py).

Struktur checkpoint (folder):
    <ckpt>/t5/                     -> save_pretrained (model + tokenizer)
    <ckpt>/graph_fusion.pt         -> state_dict fusion (hanya jika use_gcn)
    <ckpt>/abstractive_config.json -> konfigurasi model + data
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from abstractive.fusion import SentenceGraphFusion
from abstractive.paths import model_cache_dir

CONFIG_NAME = "abstractive_config.json"
FUSION_NAME = "graph_fusion.pt"


def load_base_model_and_tokenizer(model_name: str):
    """Muat T5 + tokenizer; simpan ke models/ saat pertama kali."""
    local = Path(model_name).expanduser()
    if local.is_dir():
        src = local
    else:
        cache = model_cache_dir(model_name)
        if (cache / "config.json").exists():
            src = cache
        else:
            print(f"  Downloading {model_name} -> {cache}")
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            cache.mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(cache)
            mdl.save_pretrained(cache)
            return mdl, tok
    print(f"  Loading base model from {src}")
    return AutoModelForSeq2SeqLM.from_pretrained(src), AutoTokenizer.from_pretrained(src)


class IndoT5GCN(nn.Module):
    def __init__(self, t5: nn.Module, use_gcn: bool = False, fusion_cfg: dict | None = None):
        super().__init__()
        self.t5 = t5
        self.use_gcn = use_gcn
        self.fusion_cfg = dict(fusion_cfg or {})
        self.fusion = None
        if use_gcn:
            self.fusion = SentenceGraphFusion(d_model=t5.config.d_model, **self.fusion_cfg)

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------

    def encode(self, batch: dict) -> BaseModelOutput:
        enc = self.t5.get_encoder()(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        hidden = enc.last_hidden_state
        if self.use_gcn:
            hidden = self.fusion(
                hidden,
                sent_index=batch["sent_index"],
                node_offsets=batch["node_offsets"],
                num_nodes=batch["num_nodes"],
                edge_index=batch["edge_index"],
                edge_weight=batch["edge_weight"],
                node_extra=batch.get("node_extra"),
            )
        return BaseModelOutput(last_hidden_state=hidden)

    def forward(self, batch: dict):
        enc = self.encode(batch)
        return self.t5(
            encoder_outputs=enc,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )

    @torch.no_grad()
    def generate(self, batch: dict, **gen_kwargs) -> torch.Tensor:
        enc = self.encode(batch)
        return self.t5.generate(
            encoder_outputs=enc,
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )

    # --------------------------------------------------------
    # simpan / muat
    # --------------------------------------------------------

    def save(self, out_dir: Path, tokenizer, extra_config: dict | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.t5.save_pretrained(out_dir / "t5")
        tokenizer.save_pretrained(out_dir / "t5")
        if self.use_gcn:
            torch.save(self.fusion.state_dict(), out_dir / FUSION_NAME)
        cfg = {"use_gcn": self.use_gcn, "fusion": self.fusion_cfg}
        cfg.update(extra_config or {})
        (out_dir / CONFIG_NAME).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, ckpt_dir: Path, device: torch.device | str = "cpu"):
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / CONFIG_NAME).read_text(encoding="utf-8"))
        t5 = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir / "t5")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "t5")
        model = cls(t5, use_gcn=cfg["use_gcn"], fusion_cfg=cfg.get("fusion"))
        if cfg["use_gcn"]:
            state = torch.load(ckpt_dir / FUSION_NAME, map_location="cpu", weights_only=True)
            model.fusion.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, tokenizer, cfg
