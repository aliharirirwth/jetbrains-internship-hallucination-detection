#!/usr/bin/env python3
"""Layer, feature, and pooling ablations; save results to results/."""
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.transfer import _compute_geometric_features, _mahalanobis_stats, run_transfer_experiment
from src.models.probe import HallucinationProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--probe", default="results/probe_halueval.pkl")
    ap.add_argument("--features_dir", default=None)
    ap.add_argument("--train_dataset", default="halueval")
    ap.add_argument("--eval_datasets", nargs="+", default=["halueval", "medhallu"])
    ap.add_argument("--model_short", default="Llama-3.1-8B")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    features_dir = Path(args.features_dir or root / config.get("output", {}).get("features_dir", "data/features"))
    if not features_dir.is_absolute():
        features_dir = root / features_dir
    results_dir = root / config.get("output", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    layers = config.get("model", {}).get("layers_to_extract", [-1])
    pooling = config.get("pooling", "mean")
    feats_config = config.get("features", config)

    # Load train data once
    train_labels = np.load(features_dir / f"{args.train_dataset}_labels.npy")
    hidden_train = {}
    for li in layers:
        p = features_dir / f"{args.train_dataset}_{args.model_short}_layer{li}_{pooling}.npy"
        if p.exists():
            hidden_train[li] = np.load(p)
    if not hidden_train:
        print("No features found. Run 03_extract_features.py first.")
        sys.exit(1)

    # 1) Feature ablation: leave-one-out
    feature_ablation = []
    for drop in ["use_mahalanobis", "use_cosine_sim", "use_norm", "use_layer_diff"]:
        cfg = {**feats_config, drop: False}
        X = _compute_geometric_features(hidden_train, train_labels, cfg)
        probe = HallucinationProbe(probe_type="logistic", C=1.0, max_iter=1000)
        probe.fit(X, train_labels)
        for eval_ds in args.eval_datasets:
            p = features_dir / f"{eval_ds}_labels.npy"
            if not p.exists():
                continue
            hidden_eval = {li: np.load(features_dir / f"{eval_ds}_{args.model_short}_layer{li}_{pooling}.npy") for li in layers if (features_dir / f"{eval_ds}_{args.model_short}_layer{li}_{pooling}.npy").exists()}
            if not hidden_eval:
                continue
            stats = _mahalanobis_stats(hidden_train[layers[-1]], train_labels)
            X_eval = _compute_geometric_features(hidden_eval, np.load(p), cfg, stats)
            proba = probe.predict_proba(X_eval)[:, 1]
            m = compute_all_metrics(np.load(p), proba)
            feature_ablation.append({"dropped": drop, "eval_dataset": eval_ds, "auroc": m["auroc"], "f1": m["f1"]})
    if feature_ablation:
        pd.DataFrame(feature_ablation).to_csv(results_dir / "ablation_features.csv", index=False)
        print("Feature ablation -> results/ablation_features.csv")

    # 2) Layer ablation: single-layer probes (if multiple layers extracted)
    layer_ablation = []
    for li in layers:
        path = features_dir / f"{args.train_dataset}_{args.model_short}_layer{li}_{pooling}.npy"
        if not path.exists():
            continue
        h = {li: np.load(path)}
        cfg = {**feats_config, "use_layer_diff": False}
        X = _compute_geometric_features(h, train_labels, cfg)
        probe = HallucinationProbe(probe_type="logistic", C=1.0, max_iter=1000)
        probe.fit(X, train_labels)
        for eval_ds in args.eval_datasets:
            p = features_dir / f"{eval_ds}_labels.npy"
            path_li = features_dir / f"{eval_ds}_{args.model_short}_layer{li}_{pooling}.npy"
            if not p.exists() or not path_li.exists():
                continue
            hidden_eval = {li: np.load(path_li)}
            stats = _mahalanobis_stats(hidden_train.get(layers[-1], list(hidden_train.values())[0]), train_labels)
            X_eval = _compute_geometric_features(hidden_eval, np.load(p), cfg, stats)
            proba = probe.predict_proba(X_eval)[:, 1]
            m = compute_all_metrics(np.load(p), proba)
            layer_ablation.append({"layer": li, "eval_dataset": eval_ds, "auroc": m["auroc"], "f1": m["f1"]})
    if layer_ablation:
        pd.DataFrame(layer_ablation).to_csv(results_dir / "ablation_layers.csv", index=False)
        print("Layer ablation -> results/ablation_layers.csv")

    print("Ablations done.")


if __name__ == "__main__":
    main()
