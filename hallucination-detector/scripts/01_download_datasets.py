#!/usr/bin/env python3
"""Download and cache all hallucination datasets from HuggingFace."""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalDataset, MedHALTDataset, MedHalluDataset
from src.datasets.utils import validate_schema

DATASETS = [
    ("halueval", HaluEvalDataset()),
    ("medhallu", MedHalluDataset()),
    ("medhalt", MedHALTDataset()),
    ("medhal", MedHalDataset()),
]


def main():
    for name, loader in DATASETS:
        print(f"Loading {name}...")
        try:
            samples = loader.load()
            print(f"  {len(samples)} samples")
            validate_schema(samples, dataset_name=name)
        except Exception as e:
            print(f"  Error: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
