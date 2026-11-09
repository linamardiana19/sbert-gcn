# IndoT5 (+GCN) Abstractive Summarization

Eksplorasi **abstractive summarization** untuk berita Bahasa Indonesia menggunakan **IndoT5**
(`Wikidepia/IndoT5-base`) pada dataset **XL-Sum**, dengan opsi menambahkan **Graph Convolutional
Network (GCN)** antar-kalimat ke encoder T5.

Modul ini adalah lanjutan dari pipeline extractive SBERT+GCN (lihat [`README.md`](../README.md) utama)
dan **tidak mengubah code extractive apa pun**. Graf kalimat, `SentenceGCN`, dan metrik ROUGE
dari pipeline extractive dipakai ulang sehingga hasilnya bisa dibandingkan langsung.

| | Extractive (SBERT+GCN) | Abstractive (IndoT5+GCN) |
|---|---|---|
| Output | memilih kalimat dari artikel | menulis kalimat ringkasan baru |
| Model | GCN + scoring head (~500K param) | IndoT5-base (ukuran T5-base, ratusan juta param) + GCN |
| Peran graf | node = kalimat, skor penting | memperkaya representasi encoder sebelum decoder |

## Arsitektur

```
Dokumen → Sentence Splitting → Tokenisasi per kalimat → Encoder IndoT5 → H
                                                                 │
   Graf kalimat (hybrid_tadaptive) ─► Pooling per kalimat → SentenceGCN → proj
                                                                 │
                                        H' = H + gate ⊙ info graf (gated fusion)
                                                                 ▼
                                              Decoder IndoT5 → Ringkasan
```

- **Encoder/Decoder** — `Wikidepia/IndoT5-base`, di-*fine-tune* penuh. Input diberi prefix `"ringkas: "`, maksimal 512 token.
- **Token → kalimat** — setiap kalimat ditokenisasi terpisah, sehingga setiap token diketahui asal kalimatnya. Indeks kalimat = indeks node pada graf hasil `scripts/run_graph.py`.
- **Graf** — default `hybrid_tadaptive` (cosine + TF-IDF, threshold adaptive), graf terbaik dari eksperimen extractive. Kalimat yang terpotong karena batas 512 token dibuang dari graf (subgraf).
- **GCN** — `SentenceGCN` dari `summarizer/gcn.py` (2 layer, hidden 256, outer residual).
- **Gated fusion** — hasil GCN disebar kembali ke token kalimatnya lalu ditambahkan ke output encoder lewat gate. Proyeksi diinisialisasi nol, jadi di awal training model identik dengan baseline IndoT5.
- **Mode baseline** — tanpa `--use_gcn`, model adalah IndoT5 murni (pembanding).

Detail rancangan: [`docs/08_abstractive_indot5_gcn.md`](../docs/08_abstractive_indot5_gcn.md).

## Struktur

```
sbert-gcn/
├── abstractive/                  # package model abstractive
│   ├── README.md                 #   dokumen ini
│   ├── paths.py                  #   lokasi dataset/models/checkpoints/results
│   ├── data.py                   #   tokenisasi per kalimat, token→kalimat, subgraf
│   ├── collator.py               #   batching token + graf
│   ├── fusion.py                 #   pooling kalimat → GCN → gated fusion
│   ├── model.py                  #   IndoT5GCN (load/save, forward, generate)
│   └── engine.py                 #   dataset builder, val loss, generate + ROUGE
├── scripts/abstractive/
│   ├── run_training.py           #   training (baseline atau --use_gcn)
│   └── run_evaluation.py         #   evaluasi ROUGE pada test/val
└── docs/08_abstractive_indot5_gcn.md
```

Folder yang dibuat otomatis saat dijalankan:

```
models/Wikidepia-IndoT5-base/                       # model dasar (sekali unduh)
dataset/xlsum/abstractive/tokenized/<tag>/          # cache tokenisasi
checkpoints/xlsum/abstractive/<scenario>/           # best/, last/, train.log, history.jsonl
results/xlsum/abstractive/<scenario>/               # summaries_*.jsonl, scores_*.json
```

## Setup

Memakai environment project yang sama (`uv sync`), ditambah `protobuf` untuk tokenizer IndoT5:

```bash
uv add protobuf
```

> Tanpa `protobuf` code tetap jalan (otomatis memakai tokenizer *slow*), hanya tokenisasi lebih lambat.

Model `Wikidepia/IndoT5-base` **diunduh otomatis** saat training pertama dan disimpan ke
`models/Wikidepia-IndoT5-base/`. Jika server tidak punya akses internet, unduh di komputer lain
lalu salin foldernya ke lokasi tersebut.

## Prasyarat Data

Dihasilkan oleh pipeline extractive (lihat README utama):

| File | Dibuat oleh | Dibutuhkan untuk |
|---|---|---|
| `dataset/xlsum/preprocessed/*.jsonl` | `scripts/run_preprocessing.py --dataset xlsum` | semua mode |
| `dataset/xlsum/sbert_mpnet/graphs/hybrid_tadaptive/*.pt` | `scripts/run_graph.py --method hybrid --threshold adaptive` | `--use_gcn` |
| `dataset/xlsum/sbert_mpnet/embeddings/*.pt` | `scripts/run_encoding.py --encoder mpnet` | `--use_sbert` (dan pembuatan graf) |

Dari `preprocessed/*.jsonl` yang dipakai:
- `sentences` → input encoder IndoT5 (disambung menjadi `ringkas: kalimat0 kalimat1 ... </s>`)
- `summary` → target decoder dan referensi ROUGE
- `id` → pencocokan dokumen dengan grafnya

Label oracle (`labels/`) tidak dipakai, karena model abstractive belajar langsung dari `summary`
tulisan manusia. Satu-satunya data baru yang dibuat adalah cache tokenisasi di
`dataset/xlsum/abstractive/tokenized/` (aman dihapus, akan dibuat ulang otomatis).

## Pipeline

Semua perintah dijalankan dari folder utama `sbert-gcn/`.

### 1. Uji cepat (smoke test)

Memastikan pipeline berjalan tanpa error (200 dokumen, 1 epoch, beberapa menit):

```bash
uv run python scripts/abstractive/run_training.py --dataset xlsum --use_gcn \
    --limit 200 --epochs 1 --val_gen_limit 50 --run_tag smoke --overwrite
```

Run pertama lebih lama karena mengunduh model dan membuat cache tokenisasi seluruh split.
Skor ROUGE dari uji cepat tidak bermakna.

### 2. Training baseline IndoT5

```bash
uv run python scripts/abstractive/run_training.py --dataset xlsum \
    --epochs 5 --batch_size 8 --grad_accum 2
```

### 3. Training IndoT5 + GCN

```bash
uv run python scripts/abstractive/run_training.py --dataset xlsum --use_gcn \
    --encoder mpnet --graph_name hybrid_tadaptive \
    --epochs 5 --batch_size 8 --grad_accum 2
```

Variasi opsional:

```bash
# + embedding SBERT sebagai fitur node GCN
... --use_gcn --use_sbert
# + positional encoding sinusoidal kalimat (konfigurasi terbaik di extractive)
... --use_gcn --pos_dim 16
```

#### Perbedaan baseline vs IndoT5 + GCN

Satu-satunya perbedaan adalah flag `--use_gcn` (`--encoder mpnet --graph_name hybrid_tadaptive`
hanya menuliskan nilai default agar eksplisit). Hyperparameter lain sengaja dibuat sama.

```
Baseline : kalimat → Encoder IndoT5 → Decoder IndoT5 → ringkasan
+ GCN    : kalimat → Encoder IndoT5 → [pooling kalimat → GCN pada graf → gated fusion] → Decoder → ringkasan
```

| | Baseline | + GCN |
|---|---|---|
| Flag | — | `--use_gcn` |
| Data yang dibaca | `preprocessed/*.jsonl` | `preprocessed/*.jsonl` + graf |
| Model | IndoT5 | IndoT5 + GCN + gated fusion |
| Hubungan antar-kalimat | hanya self-attention Transformer | self-attention + message passing GCN |
| Learning rate | `--lr` | `--lr` (IndoT5) + `--gcn_lr` (GCN/fusion) |
| `gate=` di log | — | ada |
| Folder scenario | `indot5_base_...` | `indot5_gcn_..._g<graph_name>_...` |

Keduanya dijalankan sebagai **ablation study**: selisih ROUGE antara keduanya menunjukkan
kontribusi GCN pada abstractive summarization (sejalan dengan ablation GCN pada eksperimen
extractive). Jika GCN lebih tinggi → graf kalimat membantu IndoT5; jika sama/lebih rendah →
self-attention IndoT5 sudah cukup menangkap hubungan antar-kalimat.

#### Memilih graf

Graf yang dipakai mengikuti argumen di terminal:

```
dataset/xlsum/sbert_<--encoder>/graphs/<--graph_name>/{train,val,test}.pt
```

`--graph_name` = nama folder yang dibuat `scripts/run_graph.py`:

| Graf dari `run_graph.py` | `--graph_name` |
|---|---|
| `--method hybrid --threshold adaptive` | `hybrid_tadaptive` (default) |
| `--method cosine --threshold adaptive` | `tadaptive` |
| `--method cosine --threshold 0.3` | `t03` |
| `--method tfidf` | `tfidf` |
| `--method hybrid_intersect --threshold adaptive` | `hybrid_intersect_tadaptive` |
| `--method hybrid_not_normalize --threshold adaptive` | `hybrid_not_normalize_tadaptive` |

`--encoder` (`mpnet` / `denaya` / `s3mpnet`) menentukan folder `sbert_<encoder>/`.
Graf hanya dibaca bila `--use_gcn` aktif. Saat evaluasi, pengaturan graf dibaca otomatis
dari checkpoint sehingga selalu sama dengan saat training.

### 4. Evaluasi

```bash
uv run python scripts/abstractive/run_evaluation.py \
    --checkpoint checkpoints/xlsum/abstractive/<scenario>/best
```

Konfigurasi data (graf, prefix, panjang token, parameter generate) dibaca otomatis dari
`<checkpoint>/abstractive_config.json`. Gunakan `--split val` untuk evaluasi validation set,
atau `--num_beams 1` dan sejenisnya untuk mengubah parameter generate.

### Menjalankan di latar belakang

Training penuh memakan waktu berjam-jam. Agar tidak berhenti saat terminal tertutup:

```bash
nohup uv run python scripts/abstractive/run_training.py --dataset xlsum --use_gcn \
    --epochs 5 --batch_size 8 --grad_accum 2 > train_gcn.out 2>&1 &

tail -f train_gcn.out     # pantau (Ctrl+C hanya berhenti memantau)
```

## Argumen Penting (`run_training.py`)

| Argumen | Default | Keterangan |
|---|---|---|
| `--use_gcn` | off | aktifkan GCN; tanpa flag = baseline IndoT5 |
| `--model_name` | `Wikidepia/IndoT5-base` | nama Hugging Face atau path lokal |
| `--encoder` | `mpnet` | menentukan folder graf `sbert_<encoder>` |
| `--graph_name` | `hybrid_tadaptive` | subfolder di `.../graphs/` |
| `--use_sbert` | off | tempel embedding SBERT ke fitur node |
| `--pos_dim` | 0 | positional encoding kalimat untuk node (0 = mati) |
| `--gcn_layers` / `--gcn_hidden` | 2 / 256 | konfigurasi GCN |
| `--epochs` | 5 | |
| `--batch_size` × `--grad_accum` | 8 × 2 | batch efektif 16 |
| `--lr` / `--gcn_lr` | 3e-4 / 1e-3 | LR IndoT5 / LR GCN+fusion |
| `--precision` | `fp32` | `bf16` untuk GPU Ampere+ (A100, RTX 30xx/40xx) |
| `--gradient_checkpointing` | off | hemat VRAM, training lebih lambat |
| `--max_source_len` / `--max_target_len` | 512 / 64 | panjang token input / ringkasan |
| `--val_gen_limit` | 1000 | dokumen val yang di-generate per epoch (0 = semua) |
| `--num_beams` | 4 | beam search saat generate |
| `--patience` | 0 | early stopping (0 = nonaktif) |
| `--limit` | — | batasi jumlah dokumen (uji coba) |
| `--gpu` | `1` | nilai `CUDA_VISIBLE_DEVICES` |
| `--run_tag` | — | label tambahan di nama scenario |
| `--overwrite` | off | timpa scenario lama tanpa bertanya |

Nama scenario dibentuk otomatis, contoh:
`indot5_gcn_5ep_3e-4_bs8x2_src512_Empnet_ghybrid_tadaptive_L2_h256`.

## Output

**Training** — `checkpoints/xlsum/abstractive/<scenario>/`

| File | Isi |
|---|---|
| `best/` | checkpoint ROUGE validasi terbaik (`t5/`, `graph_fusion.pt`, `abstractive_config.json`) |
| `last/` | checkpoint epoch terakhir |
| `train.log` | log lengkap |
| `history.jsonl` | loss, ROUGE, dan `gate_mean` per epoch |

`gate_mean` menunjukkan seberapa besar informasi graf dipakai model (0 = diabaikan,
mendekati 1 = dipakai penuh) — berguna untuk analisis peran GCN.

**Evaluasi** — `results/xlsum/abstractive/<scenario>/`

| File | Isi |
|---|---|
| `summaries_best_test.jsonl` | per dokumen: `reference`, `generated`, ROUGE-1/2/L |
| `scores_best_test.json` | rata-rata ROUGE (P/R/F1), jumlah dokumen, panjang ringkasan rata-rata |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `T5Converter requires the protobuf library` | `uv add protobuf` |
| `CUDA out of memory` | `--batch_size 4 --grad_accum 4`, tambah `--gradient_checkpointing`, atau `--precision bf16` |
| `Scenario '...' already exists. Overwrite?` | jawab `y`, atau tambah `--overwrite` / ganti `--run_tag` |
| `Graph file not found` | jalankan `scripts/run_graph.py` (hanya perlu untuk `--use_gcn`) |
| Peringatan `default legacy behaviour of T5Tokenizer` | aman diabaikan |
| Loss `nan` | jangan pakai fp16; gunakan `fp32` atau `bf16`, atau turunkan `--lr` |

## Hasil

Diisi setelah eksperimen (test set XL-Sum Indonesia, 4.780 dokumen):

| Metode | Tipe | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Lead-1 | extractive | 0.2119 | 0.0726 | 0.1755 |
| SBERT+GCN (k=2) | extractive | 0.2368 | 0.0749 | 0.1773 |
| IndoT5 baseline | abstractive | – | – | – |
| IndoT5 + GCN (hybrid_tadaptive) | abstractive | – | – | – |

## Referensi

- Hasan et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.* ACL-IJCNLP.
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Model: [Wikidepia/IndoT5-base](https://huggingface.co/Wikidepia/IndoT5-base)
