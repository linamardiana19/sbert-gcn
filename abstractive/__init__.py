"""Abstractive summarization dengan IndoT5 (+ GCN opsional).

Package ini berdiri sendiri di samping pipeline extractive (``summarizer/``)
dan TIDAK mengubahnya. Komponen yang dipakai ulang dari project:

- ``dataset/<ds>/preprocessed/*.jsonl``   -> kalimat + summary
- ``dataset/<ds>/<sbert_subdir>/graphs/`` -> graf kalimat (mis. hybrid_tadaptive)
- ``summarizer.gcn.SentenceGCN``          -> layer GCN
- ``evaluation.rouge``                    -> ROUGE (agar skor sebanding)

Modul:
- ``paths``    : lokasi folder dataset/models/checkpoints/results
- ``data``     : tokenisasi per kalimat + pemetaan token -> kalimat
- ``collator`` : batching token + graf
- ``fusion``   : pooling kalimat -> GCN -> gated fusion ke encoder T5
- ``model``    : IndoT5GCN (bungkus T5ForConditionalGeneration)
- ``engine``   : loop validasi, generate, dan evaluasi ROUGE
"""
