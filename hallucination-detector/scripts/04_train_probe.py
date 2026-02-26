#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.evaluation.transfer import _compute_geometric_features
from src.models.probe import HallucinationProbe


def main():
    ap = argparse.ArgumentParser(description="Train probe on reference dataset and save to disk.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train", default="halueval")
    ap.add_argument("--features_dir", default=None)
    ap.add_argument("--output", default="results/probe_halueval.pkl")
    ap.add_argument("--model_short", default="Llama-3.1-8B")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    features_dir = Path(args.features_dir or root / config.get("output", {}).get("features_dir", "data/features"))
    if not features_dir.is_absolute():
        features_dir = root / features_dir

    layers = config.get("model", {}).get("layers_to_extract", [-1])
    pooling = config.get("model", {}).get("pooling", "mean")
    feats_config = config.get("features", config)

    labels = np.load(features_dir / f"{args.train}_labels.npy")
    hidden = {}
    for li in layers:
        path = features_dir / f"{args.train}_{args.model_short}_layer{li}_{pooling}.npy"
        if path.exists():
            hidden[li] = np.load(path)
    if not hidden:
        print("No feature files found. Run 03_extract_features.py first.")
        sys.exit(1)

    print("Building features and training probe...")
    start = time.perf_counter()
    X = _compute_geometric_features(hidden, labels, feats_config)
    probe = HallucinationProbe(
        probe_type=config.get("probe", {}).get("type", "logistic"),
        C=config.get("probe", {}).get("C", 1.0),
        max_iter=config.get("probe", {}).get("max_iter", 1000),
    )
    probe.fit(X, labels)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(out_path)
    elapsed = time.perf_counter() - start
    print(f"Probe saved to {out_path} (took {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
