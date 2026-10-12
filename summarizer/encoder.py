from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "denaya/indoSBERT-large"
MODELS_DIR = Path(__file__).parent.parent / "models"

ENCODER_CONFIGS = {
    "denaya": {
        "model": "denaya/indoSBERT-large",
        "dim": 256,
        "sbert_subdir": "sbert_denaya",
    },
    "mpnet": {
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "sbert_subdir": "sbert_mpnet",
    },
}


def _model_cache_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name.replace("/", "-")


class SBERTEncoder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = _model_cache_dir(model_name)
        if cache_dir.exists():
            self.model = SentenceTransformer(str(cache_dir), device=self.device)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.save(str(cache_dir))

    def encode(self, sentences: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device)


def build_graph(
    node_features: torch.Tensor,
    threshold: float | str = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.nn.functional.normalize(node_features, p=2, dim=1)
    sim_matrix = norm @ norm.t()

    n = node_features.size(0)

    # if threshold == "adaptive":
    #     upper_tri = sim_matrix[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
    #     if upper_tri.numel() > 1:
    #         t = (upper_tri.mean() - upper_tri.std()).item()
    #         t = max(t, 0.0)
    #     else:
    #         t = 0.0
    # else:
    #     t = float(threshold)

    if threshold == "adaptive":
        upper_tri = sim_matrix[
            torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        ]
        if upper_tri.numel() > 1:
            t = (upper_tri.mean() - upper_tri.std()).item()
            t = max(t, 0.0)
        else:
            t = 0.0
    
    elif threshold == "fully_connected":
        t = None
    
    else:
        t = float(threshold)

    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    # mask = (src != dst) & (sim_matrix > t)
    if threshold == "fully_connected":
        mask = src != dst
    else:
        mask = (src != dst) & (sim_matrix > t)
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    edge_weight = sim_matrix[mask]

    return edge_index, edge_weight

    


def fit_tfidf_vectorizer(
    documents: list[list[str]],
    min_df: int = 100,
) -> TfidfVectorizer:
    import re
    from collections import Counter

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("indonesian"))
    token_re = re.compile(r"(?u)\b\w\w+\b")

    doc_freq: Counter[str] = Counter()
    for doc_sentences in documents:
        doc_tokens = set()
        for sent in doc_sentences:
            doc_tokens.update(token_re.findall(sent.lower()))
        doc_tokens -= stop_words
        doc_freq.update(doc_tokens)

    vocab = {w: i for i, w in enumerate(sorted(w for w, c in doc_freq.items() if c >= min_df))}

    all_sentences = [s for doc in documents for s in doc]
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        token_pattern=r"(?u)\b\w\w+\b",
        norm=None,
    )
    vectorizer.fit(all_sentences)
    return vectorizer


def build_graph_tfidf(
    sentences: list[str],
    vectorizer: TfidfVectorizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    tfidf_matrix = vectorizer.transform(sentences)
    sim_matrix = (tfidf_matrix @ tfidf_matrix.T).toarray()

    n = len(sentences)
    sim_tensor = torch.tensor(sim_matrix, dtype=torch.float32)

    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    mask = (src != dst) & (sim_tensor > 0)
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    edge_weight = sim_tensor[mask]

    return edge_index, edge_weight


def _normalize_matrix(m: torch.Tensor) -> torch.Tensor:
    max_val = m.max()
    if max_val > 0:
        return m / max_val
    return m


# def _resolve_cosine_threshold(cosine_matrix: torch.Tensor, n: int, threshold: float | str) -> float:
#     if threshold == "adaptive":
#         upper_tri = cosine_matrix[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
#         return max((upper_tri.mean() - upper_tri.std()).item(), 0.0) if upper_tri.numel() > 1 else 0.0
#     return float(threshold)


def _resolve_cosine_threshold(
    cosine_matrix,
    n,
    threshold
):

    if threshold == "fully_connected":
        return None

    if threshold == "adaptive":
        upper_tri = cosine_matrix[
            torch.triu(
                torch.ones(n, n, dtype=torch.bool),
                diagonal=1
            )
        ]
        return max(
            (upper_tri.mean() - upper_tri.std()).item(),
            0.0
        ) if upper_tri.numel() > 1 else 0.0

    return float(threshold)



def build_graph_hybrid(
    node_features: torch.Tensor,
    sentences: list[str],
    vectorizer: TfidfVectorizer,
    threshold: float | str = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = node_features.size(0)

    norm = torch.nn.functional.normalize(node_features, p=2, dim=1)
    cosine_matrix = norm @ norm.t()

    t = _resolve_cosine_threshold(cosine_matrix, n, threshold)

    tfidf_matrix = vectorizer.transform(sentences)
    tfidf_sim = torch.tensor((tfidf_matrix @ tfidf_matrix.T).toarray(), dtype=torch.float32)

    cosine_norm = _normalize_matrix(cosine_matrix)
    tfidf_norm = _normalize_matrix(tfidf_sim)

    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    
    if threshold == "fully_connected":
        mask = src != dst
    else:
        cosine_mask = cosine_matrix > t
        tfidf_mask = tfidf_sim > 0
        mask = (src != dst) & (cosine_mask | tfidf_mask)
        
    # cosine_mask = cosine_matrix > t
    # tfidf_mask = tfidf_sim > 0
    # mask = (src != dst) & (cosine_mask | tfidf_mask)

    weight_matrix = 0.5 * cosine_norm + 0.5 * tfidf_norm

    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    edge_weight = weight_matrix[mask]

    return edge_index, edge_weight



def build_graph_hybrid_not_normalize(
    node_features: torch.Tensor,
    sentences: list[str],
    vectorizer: TfidfVectorizer,
    threshold: float | str = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:

    n = node_features.size(0)

    # ===============================
    # Cosine similarity
    # tetap menggunakan cosine asli
    # ===============================
    norm = torch.nn.functional.normalize(
        node_features,
        p=2,
        dim=1
    )

    cosine_matrix = norm @ norm.t()

    t = _resolve_cosine_threshold(
        cosine_matrix,
        n,
        threshold
    )


    # ===============================
    # TF-IDF similarity
    # ===============================
    tfidf_matrix = vectorizer.transform(sentences)

    tfidf_sim = torch.tensor(
        (tfidf_matrix @ tfidf_matrix.T).toarray(),
        dtype=torch.float32
    )


    src, dst = torch.meshgrid(
        torch.arange(n),
        torch.arange(n),
        indexing="ij"
    )


    # ===============================
    # Edge construction
    # sama dengan hybrid baseline
    # ===============================
    if threshold == "fully_connected":
        mask = src != dst

    else:
        cosine_mask = cosine_matrix > t
        tfidf_mask = tfidf_sim > 0

        mask = (
            (src != dst)
            &
            (cosine_mask | tfidf_mask)
        )


    # ===============================
    # PERBEDAAN EKSPERIMEN:
    # tidak ada normalisasi
    # ===============================
    weight_matrix = (
        0.5 * cosine_matrix
        +
        0.5 * tfidf_sim
    )


    edge_index = torch.stack(
        [src[mask], dst[mask]],
        dim=0
    )

    edge_weight = weight_matrix[mask]


    return edge_index, edge_weight



def build_graph_hybrid_intersect(
    node_features: torch.Tensor,
    sentences: list[str],
    vectorizer: TfidfVectorizer,
    threshold: float | str = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = node_features.size(0)

    norm = torch.nn.functional.normalize(node_features, p=2, dim=1)
    cosine_matrix = norm @ norm.t()

    t = _resolve_cosine_threshold(cosine_matrix, n, threshold)

    tfidf_matrix = vectorizer.transform(sentences)
    tfidf_sim = torch.tensor((tfidf_matrix @ tfidf_matrix.T).toarray(), dtype=torch.float32)

    cosine_norm = _normalize_matrix(cosine_matrix)
    tfidf_norm = _normalize_matrix(tfidf_sim)

    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    cosine_mask = cosine_matrix > t
    tfidf_mask = tfidf_sim > 0
    mask = (src != dst) & (cosine_mask & tfidf_mask)

    weight_matrix = 0.5 * cosine_norm + 0.5 * tfidf_norm

    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    edge_weight = weight_matrix[mask]

    return edge_index, edge_weight
