import json
from pathlib import Path
from tqdm import tqdm


def preprocess(split: str, input_dir: Path, output_dir: Path) -> None:
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {split_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split}.jsonl"
    failed_file = output_dir / f"{split}_failed.jsonl"

    json_files = sorted(split_dir.glob("*.json"))
    failures = []

    with open(output_file, "w", encoding="utf-8") as f_out:
        for json_file in tqdm(json_files, desc=f"Liputan6 {split}"):
            try:
                doc = json.loads(json_file.read_text(encoding="utf-8"))
                sentences = [
                    " ".join(words) for words in doc["clean_article"]
                ]
                summary = " ".join(
                    " ".join(words) for words in doc["clean_summary"]
                )
                record = {
                    "id": str(doc["id"]),
                    "sentences": sentences,
                    "summary": summary,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                failures.append({"source": json_file.name, "error": str(e)})

    if failures:
        with open(failed_file, "w", encoding="utf-8") as f_fail:
            for entry in failures:
                f_fail.write(json.dumps(entry, ensure_ascii=False) + "\n")
