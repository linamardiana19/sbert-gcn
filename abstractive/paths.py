"""Lokasi folder untuk pipeline abstractive.

Mengikuti konvensi project yang sudah ada:
- dataset/<ds>/preprocessed/            (hasil scripts/run_preprocessing.py)
- dataset/<ds>/<sbert_subdir>/graphs/   (hasil scripts/run_graph.py)
- models/<org>-<nama_model>/            (sama dengan summarizer.encoder._model_cache_dir)

Folder baru yang dibuat oleh pipeline abstractive:
- dataset/<ds>/abstractive/tokenized/<tag>/  (cache tokenisasi)
- checkpoints/<ds>/abstractive/<scenario>/   (best/, last/, train.log)
- results/<ds>/abstractive/<scenario>/       (summaries_*.jsonl, scores_*.json)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_MODEL = "Wikidepia/IndoT5-base"

# Nama split per dataset (liputan6 memakai "dev" untuk validasi)
DATASET_SPLITS = {
    "xlsum": {"train": "train", "val": "val", "test": "test"},
    "liputan6": {"train": "train", "val": "dev", "test": "test"},
}


def dataset_dir(dataset: str) -> Path:
    return BASE_DIR / "dataset" / dataset


def preprocessed_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "preprocessed"


def graph_dir(dataset: str, sbert_subdir: str, graph_name: str) -> Path:
    return dataset_dir(dataset) / sbert_subdir / "graphs" / graph_name


def embeddings_dir(dataset: str, sbert_subdir: str) -> Path:
    return dataset_dir(dataset) / sbert_subdir / "embeddings"


def model_cache_dir(model_name: str) -> Path:
    """models/Wikidepia-IndoT5-base untuk 'Wikidepia/IndoT5-base'."""
    return MODELS_DIR / model_name.replace("/", "-")


def model_slug(model_name: str) -> str:
    p = Path(model_name)
    if p.is_dir():
        return p.name
    return model_name.split("/")[-1]


def token_cache_dir(dataset: str, model_name: str, max_source_len: int, max_target_len: int, prefix: str) -> Path:
    prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()[:6]
    tag = f"{model_slug(model_name)}_src{max_source_len}_tgt{max_target_len}_p{prefix_hash}"
    return dataset_dir(dataset) / "abstractive" / "tokenized" / tag


def checkpoints_root(dataset: str) -> Path:
    return BASE_DIR / "checkpoints" / dataset / "abstractive"


def results_root(dataset: str) -> Path:
    return BASE_DIR / "results" / dataset / "abstractive"
