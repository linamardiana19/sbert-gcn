import json
import nltk
from pathlib import Path
from tqdm import tqdm

nltk.download("punkt_tab", quiet=True)


def preprocess(split: str, input_dir: Path, output_dir: Path) -> None:
    input_file = input_dir / f"indonesian_{split}.jsonl"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split}.jsonl"
    failed_file = output_dir / f"{split}_failed.jsonl"

    failures = []

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        lines = f_in.readlines()
        for line_num, line in enumerate(
            tqdm(lines, desc=f"XLSum {split}"), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                sentences = [
                    s.strip()
                    for s in nltk.sent_tokenize(doc["text"])
                    if s.strip()
                ]
                record = {
                    "id": str(doc["id"]),
                    "sentences": sentences,
                    "summary": doc["summary"],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                failures.append({"source": str(line_num), "error": str(e)})

    if failures:
        with open(failed_file, "w", encoding="utf-8") as f_fail:
            for entry in failures:
                f_fail.write(json.dumps(entry, ensure_ascii=False) + "\n")
