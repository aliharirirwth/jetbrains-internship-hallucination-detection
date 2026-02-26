#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalDataset, MedHalluDataset, MedHALTDataset
from src.datasets.utils import validate_schema
from tqdm import tqdm

DATASETS = [
    ("halueval", HaluEvalDataset()),
    ("medhallu", MedHalluDataset()),
    ("medhalt", MedHALTDataset()),
    ("medhal", MedHalDataset()),
]


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = int(seconds // 60), seconds % 60
    return f"{m}m {s:.1f}s"


def main():
    ap = argparse.ArgumentParser(description="Download and cache hallucination datasets from HuggingFace.")
    ap.add_argument("--quiet", action="store_true", help="Minimal output (no tqdm, only errors and summary).")
    args = ap.parse_args()
    total_start = time.perf_counter()
    if not args.quiet:
        print("Downloading datasets from HuggingFace (first run may take 1–5 min per dataset).")
    for name, loader in tqdm(DATASETS, desc="datasets", unit="dataset", disable=args.quiet):
        if not args.quiet:
            tqdm.write(f"Loading {name}... (downloading if needed)")
        start = time.perf_counter()
        try:
            samples = loader.load()
            elapsed = time.perf_counter() - start
            if not args.quiet:
                tqdm.write(f"  {len(samples)} samples in {_format_elapsed(elapsed)}")
            validate_schema(samples, dataset_name=name)
        except Exception as e:
            if not args.quiet:
                tqdm.write(f"  Error: {e}")
    total_elapsed = time.perf_counter() - total_start
    print(f"Done. Total time: {_format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
