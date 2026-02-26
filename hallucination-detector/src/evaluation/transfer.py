from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.geometric import (
    build_feature_vector,
    layer_difference,
    mahalanobis_features,
    representation_norm,
)
from ..models.probe import HallucinationProbe
from .metrics import compute_all_metrics


def _mahalanobis_stats(hidden: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute class means and inverse covariances for binary labels.

    Args:
        hidden: Hidden states (N, dim).
        labels: Binary labels (N,) with 0 = faithful, 1 = hallucination.

    Returns:
        Tuple (mean0, mean1, inv_cov0, inv_cov1) for the two classes.
    """
    dim = hidden.shape[1]
    mask0 = labels == 0
    mask1 = labels == 1
    mean0 = np.mean(hidden[mask0], axis=0) if mask0.any() else np.zeros(dim)
    mean1 = np.mean(hidden[mask1], axis=0) if mask1.any() else np.zeros(dim)
    cov0 = np.cov(hidden[mask0].T) if mask0.sum() > 1 else np.eye(dim) * 1e-6
    cov1 = np.cov(hidden[mask1].T) if mask1.sum() > 1 else np.eye(dim) * 1e-6
    inv_cov0 = np.linalg.pinv(cov0 + np.eye(dim) * 1e-5)
    inv_cov1 = np.linalg.pinv(cov1 + np.eye(dim) * 1e-5)
    return mean0, mean1, inv_cov0, inv_cov1


def _compute_geometric_features(
    hidden_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    config: dict[str, Any],
    mahalanobis_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Build geometric feature matrix for all samples.

    If mahalanobis_stats is None, computes them from labels (training); otherwise
    uses provided stats (evaluation/transfer).

    Args:
        hidden_by_layer: Dict layer_index -> (N, hidden_dim) per layer.
        labels: Binary labels (N,) for Mahalanobis when stats are computed.
        config: Feature flags (use_mahalanobis, use_cosine_sim, use_norm, use_layer_diff).
        mahalanobis_stats: Optional (mean0, mean1, inv_cov0, inv_cov1) for eval.

    Returns:
        Array of shape (N, n_features) with concatenated geometric features.
    """
    use_mahalanobis = config.get("use_mahalanobis", True)
    use_cosine = config.get("use_cosine_sim", True)
    use_norm = config.get("use_norm", True)
    use_layer_diff = config.get("use_layer_diff", True)
    layers = sorted(hidden_by_layer.keys())
    if not layers:
        return np.zeros((0, 0))

    N = hidden_by_layer[layers[0]].shape[0]
    if mahalanobis_stats is None:
        mean0, mean1, inv_cov0, inv_cov1 = _mahalanobis_stats(hidden_by_layer[layers[-1]], labels)
    else:
        mean0, mean1, inv_cov0, inv_cov1 = mahalanobis_stats

    rows = []
    for i in range(N):
        feats: dict[str, Any] = {}
        if use_mahalanobis:
            x = hidden_by_layer[layers[-1]][i]
            feats["mahalanobis"] = mahalanobis_features(x, mean0, mean1, inv_cov0, inv_cov1)
        if use_norm:
            feats["norm"] = representation_norm(hidden_by_layer[layers[-1]][i])
        if use_layer_diff and len(layers) >= 2:
            feats["layer_diff"] = layer_difference(hidden_by_layer[layers[0]][i], hidden_by_layer[layers[-1]][i])
        if use_cosine:
            feats["cosine_sim"] = np.array([0.0], dtype=np.float32)
        row = build_feature_vector(feats, config)
        rows.append(row)
    return np.stack(rows, axis=0)


def run_transfer_experiment(
    train_dataset: str,
    eval_datasets: list[str],
    config: dict[str, Any],
    features_dir: str | Path,
    layers: list[int] | None = None,
    pooling: str = "mean",
    model_short: str = "Llama-3.1-8B",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a probe on one dataset and evaluate on multiple eval datasets.

    Expects pre-extracted .npy features under features_dir (per dataset and layer).

    Args:
        train_dataset: Name of dataset used for training the probe.
        eval_datasets: List of dataset names to evaluate on.
        config: Model/feature config (e.g. from config.yaml).
        features_dir: Directory containing *_labels.npy and *_layer*_*.npy files.
        layers: Layer indices to use; defaults from config.
        pooling: Pooling strategy (e.g. mean).
        model_short: Short model name for file lookup.
        random_state: Random seed for probe training.

    Returns:
        DataFrame with columns train_dataset, eval_dataset, auroc, f1, n_train, n_eval, layer, pooling.
    """
    features_dir = Path(features_dir)
    layers = layers or config.get("model", {}).get("layers_to_extract", [-1])
    results = []

    # Load train data
    train_labels = np.load(features_dir / f"{train_dataset}_labels.npy")
    if len(np.unique(train_labels)) < 2:
        warnings.warn(
            f"Train dataset '{train_dataset}' has only one class (e.g. all hallucination). "
            "Skipping probe fit; use balanced data or another train dataset.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()
    hidden_train: dict[int, np.ndarray] = {}
    for li in layers:
        path = features_dir / f"{train_dataset}_{model_short}_layer{li}_{pooling}.npy"
        if path.exists():
            hidden_train[li] = np.load(path)
    if not hidden_train:
        return pd.DataFrame()
    X_train = _compute_geometric_features(hidden_train, train_labels, config.get("features", config))
    y_train = train_labels

    feats_config = config.get("features", config)
    mahalanobis_stats = _mahalanobis_stats(hidden_train[layers[-1]], y_train)

    probe_cfg = config.get("probe", {})
    probe = HallucinationProbe(
        probe_type=probe_cfg.get("type", "logistic"),
        C=probe_cfg.get("C", 1.0),
        max_iter=probe_cfg.get("max_iter", 1000),
        random_state=random_state,
    )
    probe.fit(X_train, y_train)

    for eval_ds in eval_datasets:
        path_labels = features_dir / f"{eval_ds}_labels.npy"
        if not path_labels.exists():
            continue
        eval_labels = np.load(path_labels)
        hidden_eval: dict[int, np.ndarray] = {}
        for li in layers:
            path = features_dir / f"{eval_ds}_{model_short}_layer{li}_{pooling}.npy"
            if path.exists():
                hidden_eval[li] = np.load(path)
        if not hidden_eval:
            continue
        X_eval = _compute_geometric_features(hidden_eval, eval_labels, feats_config, mahalanobis_stats)
        y_prob = probe.predict_proba(X_eval)[:, 1]
        metrics = compute_all_metrics(eval_labels, y_prob)
        results.append({
            "train_dataset": train_dataset,
            "eval_dataset": eval_ds,
            "auroc": metrics["auroc"],
            "f1": metrics["f1"],
            "n_train": len(y_train),
            "n_eval": metrics["n_samples"],
            "layer": str(layers),
            "pooling": pooling,
        })

    return pd.DataFrame(results)


def run_full_transfer_matrix(
    all_datasets: list[str],
    config: dict[str, Any],
    features_dir: str | Path,
    results_dir: str | Path,
    model_short: str = "Llama-3.1-8B",
) -> pd.DataFrame:
    """Run full N×N transfer matrix: train on each dataset, evaluate on all.

    Saves combined results to results_dir/transfer_matrix.csv.

    Args:
        all_datasets: List of dataset names (train and eval).
        config: Model/feature config.
        features_dir: Directory with pre-extracted .npy features.
        results_dir: Directory to write transfer_matrix.csv.
        model_short: Short model name for file lookup.

    Returns:
        DataFrame with all (train_dataset, eval_dataset, auroc, f1, ...) rows.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for train_ds in all_datasets:
        df = run_transfer_experiment(
            train_ds,
            all_datasets,
            config,
            features_dir,
            model_short=model_short,
        )
        all_rows.append(df)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(results_dir / "transfer_matrix.csv", index=False)
    return out
