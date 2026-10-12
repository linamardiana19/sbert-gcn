"""
Statistical analysis of the dataset for SBERT + GCN extractive summarization experiments.

Produces:
1. Document statistics (number of sentences, number of words per document)
2. Sentence statistics (number of words per sentence)
3. Summary statistics (number of words, number of sentences)
4. Coverage & density (how extractive the reference summary is)
5. Novel n-gram (how abstractive the reference summary is)
6. Compression ratio
7. Sentence count distribution

Usage:
    uv run python scripts/analyze_dataset.py --dataset xlsum
    uv run python scripts/analyze_dataset.py --dataset xlsum --split train
"""

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import nltk
import numpy as np

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


BASE_DIR = Path(__file__).resolve().parent.parent


class Tee:
    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def tokenize_words(text: str) -> list[str]:
    return text.lower().split()


def compute_extractive_fragments(doc_words: list[str], summary_words: list[str]):
    if not summary_words or not doc_words:
        return 0.0, 0.0

    fragments = []
    i = 0
    while i < len(summary_words):
        best_len = 0
        for j in range(len(doc_words)):
            k = 0
            while (i + k < len(summary_words)
                   and j + k < len(doc_words)
                   and summary_words[i + k] == doc_words[j + k]):
                k += 1
            best_len = max(best_len, k)
        if best_len > 0:
            fragments.append(best_len)
            i += best_len
        else:
            i += 1

    covered_words = sum(fragments)
    coverage = covered_words / len(summary_words) if summary_words else 0.0
    density = (sum(f * f for f in fragments) / len(summary_words)) if summary_words else 0.0

    return coverage, density


def compute_novel_ngrams(doc_words: list[str], summary_words: list[str], max_n: int = 4):
    if not summary_words or not doc_words:
        return {n: 0.0 for n in range(1, max_n + 1)}

    results = {}
    for n in range(1, max_n + 1):
        doc_ngrams = set()
        for i in range(len(doc_words) - n + 1):
            doc_ngrams.add(tuple(doc_words[i:i + n]))

        summ_ngrams = []
        for i in range(len(summary_words) - n + 1):
            summ_ngrams.append(tuple(summary_words[i:i + n]))

        if not summ_ngrams:
            results[n] = 0.0
            continue

        novel = sum(1 for ng in summ_ngrams if ng not in doc_ngrams)
        results[n] = novel / len(summ_ngrams)

    return results


def percentile_stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def analyze_split(docs: list[dict], split_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Analysis: {split_name} ({len(docs)} documents)")
    print(f"{'='*60}")

    # --- 1. Document statistics ---
    doc_num_sents = []
    doc_num_words = []
    sent_num_words = []

    for doc in docs:
        sents = doc["sentences"]
        doc_num_sents.append(len(sents))
        words_in_doc = 0
        for s in sents:
            w = tokenize_words(s)
            sent_num_words.append(len(w))
            words_in_doc += len(w)
        doc_num_words.append(words_in_doc)

    doc_sents_stats = percentile_stats(doc_num_sents)
    doc_words_stats = percentile_stats(doc_num_words)
    sent_words_stats = percentile_stats(sent_num_words)

    print(f"\n--- Documents ---")
    print(f"  Sentences per doc : mean={doc_sents_stats['mean']:.1f}, "
          f"median={doc_sents_stats['median']:.0f}, "
          f"p95={doc_sents_stats['p95']:.0f}, "
          f"max={doc_sents_stats['max']:.0f}")
    print(f"  Words per doc     : mean={doc_words_stats['mean']:.1f}, "
          f"median={doc_words_stats['median']:.0f}, "
          f"p95={doc_words_stats['p95']:.0f}")

    print(f"\n--- Sentences ---")
    print(f"  Words per sentence: mean={sent_words_stats['mean']:.1f}, "
          f"median={sent_words_stats['median']:.0f}, "
          f"p95={sent_words_stats['p95']:.0f}, "
          f"max={sent_words_stats['max']:.0f}")

    # --- 2. Summary statistics ---
    summ_num_words = []
    summ_num_sents = []

    for doc in docs:
        summary = doc["summary"]
        summ_words = tokenize_words(summary)
        summ_num_words.append(len(summ_words))
        summ_sents = nltk.sent_tokenize(summary)
        summ_num_sents.append(len(summ_sents))

    summ_words_stats = percentile_stats(summ_num_words)
    summ_sents_stats = percentile_stats(summ_num_sents)

    print(f"\n--- Summary ---")
    print(f"  Words per summary    : mean={summ_words_stats['mean']:.1f}, "
          f"median={summ_words_stats['median']:.0f}, "
          f"p95={summ_words_stats['p95']:.0f}")
    print(f"  Sentences per summary: mean={summ_sents_stats['mean']:.1f}, "
          f"median={summ_sents_stats['median']:.0f}, "
          f"max={summ_sents_stats['max']:.0f}")

    # --- 3. Compression Ratio ---
    compression_ratios = []
    for dw, sw in zip(doc_num_words, summ_num_words):
        if dw > 0:
            compression_ratios.append(sw / dw)

    comp_stats = percentile_stats(compression_ratios)
    print(f"\n--- Compression Ratio (summary_words / doc_words) ---")
    print(f"  mean={comp_stats['mean']:.3f}, median={comp_stats['median']:.3f}")

    # --- 4. Coverage, Density & Novel N-gram ---
    print(f"\n--- Coverage & Density ({len(docs)} docs) ---")

    coverages = []
    densities = []
    novel_ngrams = {1: [], 2: [], 3: [], 4: []}
    for doc in docs:
        doc_words = tokenize_words(" ".join(doc["sentences"]))
        summ_words = tokenize_words(doc["summary"])
        cov, den = compute_extractive_fragments(doc_words, summ_words)
        coverages.append(cov)
        densities.append(den)
        novel = compute_novel_ngrams(doc_words, summ_words)
        for n in range(1, 5):
            novel_ngrams[n].append(novel[n])

    cov_stats = percentile_stats(coverages)
    den_stats = percentile_stats(densities)
    novel_stats = {n: percentile_stats(novel_ngrams[n]) for n in range(1, 5)}

    print(f"  Coverage : mean={cov_stats['mean']:.3f}, median={cov_stats['median']:.3f} "
          f"(fraction of summary words from the document)")
    print(f"  Density  : mean={den_stats['mean']:.3f}, median={den_stats['median']:.3f} "
          f"(average fragment length squared)")

    if cov_stats["mean"] < 0.5:
        print(f"  Warning: Low coverage — reference summaries are heavily paraphrased; "
              f"extractive approach ceiling may be limited")
    elif cov_stats["mean"] > 0.8:
        print(f"  High coverage — reference summaries are sufficiently extractive")

    print(f"\n--- Novel N-gram ({len(docs)} docs) ---")
    print(f"  (fraction of n-grams in the summary NOT present in the document)")
    for n in range(1, 5):
        s = novel_stats[n]
        print(f"  Novel {n}-gram: mean={s['mean']:.3f}, median={s['median']:.3f}")

    # --- 5. Sentence count distribution ---
    sent_counts = Counter(doc_num_sents)
    print(f"\n--- Sentence Count Distribution per Document ---")
    for n_sents in sorted(sent_counts.keys()):
        count = sent_counts[n_sents]
        pct = count / len(docs) * 100
        bar = "#" * int(pct)
        if n_sents <= 30 or count > 10:
            print(f"  {n_sents:3d} sentences: {count:5d} ({pct:5.1f}%) {bar}")
    if max(sent_counts.keys()) > 30:
        over_30 = sum(c for n, c in sent_counts.items() if n > 30)
        print(f"  >30 sentences: {over_30:5d} ({over_30/len(docs)*100:5.1f}%)")

    return {
        "split": split_name,
        "num_docs": len(docs),
        "doc_num_sents": doc_sents_stats,
        "doc_num_words": doc_words_stats,
        "sent_num_words": sent_words_stats,
        "summ_num_words": summ_words_stats,
        "summ_num_sents": summ_sents_stats,
        "compression_ratio": comp_stats,
        "coverage": cov_stats,
        "density": den_stats,
        "novel_ngrams": {str(n): novel_stats[n] for n in range(1, 5)},
    }


def main():
    ap = argparse.ArgumentParser(description="Dataset analysis for SBERT+GCN")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], required=True)
    ap.add_argument("--split", choices=["train", "val", "dev", "test"], default=None,
                    help="Analyze a specific split (default: all)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory (default: dataset/<name>/)")
    args = ap.parse_args()

    log = Tee(sys.stdout)
    sys.stdout = log

    data_dir = BASE_DIR / "dataset" / args.dataset / "preprocessed"

    if args.dataset == "xlsum":
        splits = ["train", "val", "test"]
    else:
        splits = ["train", "dev", "test"]

    if args.split:
        splits = [args.split]

    all_results = {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        docs = load_jsonl(path)
        results = analyze_split(docs, f"{args.dataset}/{split}")
        all_results[split] = results

    output_dir = Path(args.output) if args.output else BASE_DIR / "dataset" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    txt_path = output_dir / "analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(log.getvalue())
    print(f"Text saved to {txt_path}")


if __name__ == "__main__":
    main()
