import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from preprocessing import liputan6, xlsum

XLSUM_INPUT = BASE_DIR / "dataset" / "xlsum"
XLSUM_OUTPUT = BASE_DIR / "dataset" / "xlsum" / "preprocessed"

LIPUTAN6_INPUT = BASE_DIR / "dataset" / "liputan6" / "canonical"
LIPUTAN6_OUTPUT = BASE_DIR / "dataset" / "liputan6" / "preprocessed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess datasets into sentence-tokenized JSONL.")
    ap.add_argument("--dataset", choices=["xlsum", "liputan6"], nargs="+", default=["xlsum", "liputan6"], help="Which dataset(s) to preprocess (default: both).")
    args = ap.parse_args()

    datasets = {
        "xlsum": {
            "splits": ["train", "val", "test"],
            "func": xlsum.preprocess,
            "input": XLSUM_INPUT,
            "output": XLSUM_OUTPUT,
        },
        "liputan6": {
            "splits": ["train", "dev", "test"],
            "func": liputan6.preprocess,
            "input": LIPUTAN6_INPUT,
            "output": LIPUTAN6_OUTPUT,
        },
    }

    for ds_name in args.dataset:
        ds = datasets[ds_name]
        for split in ds["splits"]:
            print(f"\n=== {ds_name} {split} ===")
            ds["func"](split, ds["input"], ds["output"])


if __name__ == "__main__":
    main()
