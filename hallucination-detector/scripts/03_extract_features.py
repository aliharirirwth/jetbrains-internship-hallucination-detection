#!/usr/bin/env python3
"""Run LLM inference and save hidden states to data/features."""
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalDataset, MedHALTDataset, MedHalluDataset
from src.models import HiddenStateExtractor

DATASET_LOADERS = {
    "halueval": HaluEvalDataset(),
    "medhallu": MedHalluDataset(),
    "medhalt": MedHALTDataset(),
    "medhal": MedHalDataset(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default=None, help="Override model name (e.g. facebook/opt-125m)")
    ap.add_argument("--datasets", nargs="+", default=["halueval"], help="Which datasets to extract")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--features_dir", default=None)
    args = ap.parse_args()

    with open(Path(__file__).resolve().parent.parent / args.config) as f:
        config = yaml.safe_load(f)
    model_config = config.get("model", {})
    if args.model:
        model_config = {**model_config, "name": args.model}
    features_dir = args.features_dir or config.get("output", {}).get("features_dir", "data/features")
    features_dir = Path(__file__).resolve().parent.parent / features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    extractor = HiddenStateExtractor(model_config["name"], model_config)
    layers = model_config.get("layers_to_extract", [-1])
    pooling = model_config.get("pooling", "mean")

    for ds_name in args.datasets:
        if ds_name not in DATASET_LOADERS:
            print(f"Unknown dataset: {ds_name}")
            continue
        samples = DATASET_LOADERS[ds_name].load()
        samples = samples[: args.max_samples]
        if not samples:
            print(f"Skipping {ds_name}: 0 samples (loader may not match dataset schema).")
            continue
        print(f"Extracting {len(samples)} samples from {ds_name}...")
        extractor.extract_batch(
            samples,
            layers=layers,
            batch_size=args.batch_size,
            save_path=features_dir,
            dataset_name=ds_name,
        )
    print(f"Features saved to {features_dir}")


if __name__ == "__main__":
    main()
