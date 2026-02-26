#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalDataset, MedHalluDataset, MedHALTDataset
from src.datasets.utils import validate_schema
from tqdm import tqdm

DATASETS = {
    "halueval": HaluEvalDataset(),
    "medhallu": MedHalluDataset(),
    "medhalt": MedHALTDataset(),
    "medhal": MedHalDataset(),
}


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = int(seconds // 60), seconds % 60
    return f"{m}m {s:.1f}s"


def main():
    ap = argparse.ArgumentParser(description="Normalize datasets to common schema and save CSVs.")
    ap.add_argument("--out", default="data/processed", help="Output dir for CSV per dataset")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--quiet", action="store_true", help="Minimal output (no tqdm, only summary).")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    if not args.quiet:
        print("Loading and preprocessing datasets (downloads from HuggingFace if needed)...")
    names = [n for n in args.datasets if n in DATASETS]
    for name in tqdm(names, desc="datasets", unit="dataset", disable=args.quiet):
        loader = DATASETS[name]
        start = time.perf_counter()
        try:
            samples = loader.load()
            validate_schema(samples, dataset_name=name)
            df = loader.get_dataframe()
            df.to_csv(out / f"{name}.csv", index=False)
            elapsed = time.perf_counter() - start
            if not args.quiet:
                tqdm.write(f"{name}: {len(df)} rows -> {out / f'{name}.csv'} ({_format_elapsed(elapsed)})")
        except KeyboardInterrupt:
            if not args.quiet:
                tqdm.write(f"{name}: interrupted, skipping.")
        except Exception as e:
            if not args.quiet:
                tqdm.write(f"{name}: failed ({e}), skipping.")
    total_elapsed = time.perf_counter() - total_start
    print(f"Done. Total time: {_format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
