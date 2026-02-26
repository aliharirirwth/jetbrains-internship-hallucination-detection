#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.transfer import run_transfer_experiment


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained probe on transfer datasets.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--probe", default="results/probe_halueval.pkl")
    ap.add_argument("--eval", nargs="+", default=["medhal", "medhallu"])
    ap.add_argument("--train_dataset", default="halueval")
    ap.add_argument("--features_dir", default=None)
    ap.add_argument("--model_short", default="Llama-3.1-8B")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    features_dir = Path(args.features_dir or root / config.get("output", {}).get("features_dir", "data/features"))
    if not features_dir.is_absolute():
        features_dir = root / features_dir

    print("Running transfer evaluation (train -> eval datasets)...")
    start = time.perf_counter()
    df = run_transfer_experiment(
        args.train_dataset,
        args.eval,
        config,
        features_dir,
        model_short=args.model_short,
    )
    elapsed = time.perf_counter() - start
    print(df.to_string())
    results_dir = root / config.get("output", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "transfer_results.csv", index=False)
    print(f"Done in {elapsed:.1f}s. Results saved to {results_dir / 'transfer_results.csv'}")


if __name__ == "__main__":
    main()
