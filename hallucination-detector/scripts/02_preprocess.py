#!/usr/bin/env python3
"""Normalize all datasets to common schema; run validation; optionally save to data/processed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.datasets import HaluEvalDataset, MedHalDataset, MedHALTDataset, MedHalluDataset
from src.datasets.utils import validate_schema

DATASETS = {
    "halueval": HaluEvalDataset(),
    "medhallu": MedHalluDataset(),
    "medhalt": MedHALTDataset(),
    "medhal": MedHalDataset(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed", help="Output dir for CSV per dataset")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if name not in DATASETS:
            continue
        loader = DATASETS[name]
        try:
            samples = loader.load()
            validate_schema(samples, dataset_name=name)
            df = loader.get_dataframe()
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"{name}: {len(df)} rows -> {out / f'{name}.csv'}")
        except KeyboardInterrupt:
            print(f"{name}: interrupted, skipping.")
        except Exception as e:
            print(f"{name}: failed ({e}), skipping.")
    print("Done.")


if __name__ == "__main__":
    main()
