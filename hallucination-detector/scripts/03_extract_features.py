#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalDataset, MedHalluDataset, MedHALTDataset
from src.models import HiddenStateExtractor

DATASET_LOADERS = {
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
    ap = argparse.ArgumentParser(description="Extract hidden states from LLM for each dataset.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default=None, help="Override model name (e.g. facebook/opt-125m)")
    ap.add_argument("--datasets", nargs="+", default=["halueval"], help="Which datasets to extract")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--features_dir", default=None)
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar (no ETA).")
    ap.add_argument("--quiet", action="store_true", help="Minimal output; implies --no-progress.")
    args = ap.parse_args()
    show_progress = not (args.no_progress or args.quiet)

    with open(Path(__file__).resolve().parent.parent / args.config) as f:
        config = yaml.safe_load(f)
    model_config = config.get("model", {})
    if args.model:
        model_config = {**model_config, "name": args.model}
    features_dir = args.features_dir or config.get("output", {}).get("features_dir", "data/features")
    features_dir = Path(__file__).resolve().parent.parent / features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("Loading model and tokenizer (first run may download from HuggingFace)...")
    total_start = time.perf_counter()
    extractor = HiddenStateExtractor(model_config["name"], model_config)
    layers = model_config.get("layers_to_extract", [-1])

    for ds_name in args.datasets:
        if ds_name not in DATASET_LOADERS:
            print(f"Unknown dataset: {ds_name}")
            continue
        samples = DATASET_LOADERS[ds_name].load()
        samples = samples[: args.max_samples]
        if not samples:
            print(f"Skipping {ds_name}: 0 samples (loader may not match dataset schema).")
            continue
        if not args.quiet:
            print(f"Extracting {len(samples)} samples from {ds_name}... (progress bar shows ETA)")
        start = time.perf_counter()
        extractor.extract_batch(
            samples,
            layers=layers,
            batch_size=args.batch_size,
            save_path=features_dir,
            dataset_name=ds_name,
            show_progress=show_progress,
        )
        elapsed = time.perf_counter() - start
        if not args.quiet:
            print(f"  {ds_name}: {_format_elapsed(elapsed)}")
    total_elapsed = time.perf_counter() - total_start
    print(f"Features saved to {features_dir}. Total time: {_format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
