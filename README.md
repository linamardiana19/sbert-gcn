# SBERT+GCN Extractive Summarization

Extractive text summarization untuk berita Bahasa Indonesia menggunakan **Sentence-BERT** (SBERT) sebagai sentence encoder dan **Graph Convolutional Network** (GCN) untuk inter-sentence reasoning pada dataset **XL-Sum & Liputan6**.

## Arsitektur

```
Dokumen → Sentence Splitting → SBERT Encoder (frozen) → Graph Construction → GCN → Scoring Head → Sentence Selection
```

- **SBERT Encoder** — `paraphrase-multilingual-mpnet-base-v2` (768d, frozen). Menghasilkan embedding per kalimat secara independen.
- **Graph Construction** — Hybrid union graph (cosine + TF-IDF similarity) dengan adaptive threshold (mean − std per dokumen).
- **GCN** — 2-layer GCNConv dengan outer residual connection, hidden_dim=256, dropout=0.0. Message passing memperkaya representasi kalimat dengan konteks tetangga.
- **Positional Encoding** — Sinusoidal encoding (dim=16) di-concatenate ke embedding SBERT sebelum GCN.
- **Scoring Head** — MLP 2-layer menghasilkan skor kepentingan per kalimat.
- **Sentence Selection** — Top-K (k=2) dengan trigram blocking untuk menghindari redundansi.

## Struktur Project

```
sbert-gcn/
├── summarizer/           # Model dan komponen inti
│   ├── model.py          #   SBERTSummarizer (GCN + scoring head)
│   ├── gcn.py            #   SentenceGCN, MultiSentenceGCN
│   ├── encoder.py        #   Konfigurasi SBERT encoder
│   ├── positional.py     #   Positional encoding (simple, sinusoidal, learned)
│   ├── selector.py       #   Top-K sentence selection + trigram blocking
│   └── label_builder.py  #   Greedy ROUGE labeling
├── evaluation/           # Metrik evaluasi
│   └── rouge.py          #   ROUGE-1, ROUGE-2, ROUGE-L
├── scripts/              # Pipeline scripts
│   ├── run_preprocessing.py  #   Sentence splitting
│   ├── run_encoding.py       #   SBERT encoding (pre-compute)
│   ├── run_labeling.py       #   Oracle label generation
│   ├── run_graph.py          #   Graph construction
│   ├── run_training.py       #   Model training
│   ├── run_evaluation.py     #   Test set evaluation
│   ├── analyze_dataset.py    #   Statistik dataset
│   ├── analyze_oracle.py     #   Oracle ROUGE analysis
│   ├── analyze_lead.py       #   Lead-N baseline analysis
│   ├── analyze_labels.py     #   Label distribution analysis
│   └── analyze_similarity.py #   Similarity distribution analysis
├── dataset/              # Data (tidak di-track git)
├── checkpoints/          # Model checkpoints (tidak di-track git)
└── results/              # Hasil evaluasi (tidak di-track git)
```

## Setup

**Prasyarat**: Python 3.12+, CUDA-compatible GPU

```bash
# Install dependencies dengan uv
uv sync
```

## Pipeline

### 1. Preprocessing

```bash
# Sentence splitting
uv run python scripts/run_preprocessing.py --dataset xlsum

# Sentence splitting
uv run python scripts/run_preprocessing.py --dataset liputan6

# Oracle labeling
uv run python scripts/run_labeling.py --dataset xlsum

uv run python scripts/run_labeling.py --dataset liputan6
```

### 2. Encoding & Graph Construction

```bash
# SBERT encoding (pre-compute embeddings)
uv run python scripts/run_encoding.py --dataset xlsum --encoder mpnet
uv run python scripts/run_encoding.py --dataset liputan6 --encoder mpnet

# Cosine similarity graph
uv run python scripts/run_graph.py --dataset xlsum --encoder mpnet --method cosine --threshold adaptive
uv run python scripts/run_graph.py --dataset liputan6 --encoder mpnet --method cosine --threshold adaptive

# TF-IDF graph
uv run python scripts/run_graph.py --dataset xlsum --encoder mpnet --method tfidf
uv run python scripts/run_graph.py --dataset liputan6 --encoder mpnet --method tfidf

# Hybrid union graph (cosine + TF-IDF)
uv run python scripts/run_graph.py --dataset xlsum --encoder mpnet --method hybrid --threshold adaptive
uv run python scripts/run_graph.py --dataset liputan6 --encoder mpnet --method hybrid --threshold adaptive
```

### 3. Training

```bash
uv run python scripts/run_training.py \
    --dataset xlsum \
    --encoder mpnet \
    --threshold hybrid_tadaptive \
    --num_layers 2 \
    --hidden_dim 256 \
    --residual outer \
    --dropout 0.0 \
    --scoring mlp \
    --pos_method sinusoidal \
    --pos_dim 16 \
    --top_k 2 \
    --pos_weight 11 \
    --selection_metric rouge1 \
    --label_metric rouge1
```

### 4. Evaluation

```bash
uv run python scripts/run_evaluation.py \
    --dataset xlsum \
    --encoder mpnet \
    --threshold hybrid_tadaptive \
    --top_k 2 \
    --checkpoint checkpoints/xlsum/sbert_gcn_10ep_1e-3_256_bs16_thybrid_tadaptive_Empnet_d00_Smlp_Psin16/best.pt
```

## Eksperimen OFAT

Konfigurasi optimal ditentukan melalui 6 eksperimen One-Factor-At-a-Time (OFAT) pada validation set:

| Eksperimen | Faktor | Terpilih | Val R1 | Delta |
|------------|--------|----------|--------|-------|
| 1 | Encoder | mpnet (768d) | 0.2180 | — |
| 2 | Threshold | adaptive | 0.2209 | +0.0029 |
| 3 | Jenis graf | hybrid_union | 0.2217 | +0.0008 |
| 4a–d | GCN config | layers=2, dim=256, outer, dropout=0.0 | 0.2220 | +0.0003 |
| 5 | Scoring head | mlp | 0.2242 | +0.0022 |
| 6 | Positional encoding | sinusoidal(16) | 0.2350 | +0.0108 |


## Referensi

- Hasan et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.* ACL-IJCNLP.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Jing et al. (2021). *Multi-GraS: Multi-Graph Summarization.* EMNLP.
