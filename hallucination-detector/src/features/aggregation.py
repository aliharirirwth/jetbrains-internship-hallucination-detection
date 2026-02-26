from __future__ import annotations

from pathlib import Path

import numpy as np


def load_layer_features(
    features_dir: str | Path,
    dataset_name: str,
    model_short: str,
    layers: list[int],
    pooling: str,
) -> dict[int, np.ndarray]:
    """Load per-layer .npy arrays from disk.

    Args:
        features_dir: Directory containing {dataset}_{model}_layer{li}_{pooling}.npy files.
        dataset_name: Dataset identifier (e.g. halueval).
        model_short: Model short name (e.g. opt-125m).
        layers: List of layer indices to load.
        pooling: Pooling strategy (e.g. mean).

    Returns:
        Dict mapping layer_index -> array of shape (N, hidden_dim). Missing files are skipped.
    """
    features_dir = Path(features_dir)
    out: dict[int, np.ndarray] = {}
    for li in layers:
        path = features_dir / f"{dataset_name}_{model_short}_layer{li}_{pooling}.npy"
        if path.exists():
            out[li] = np.load(path)
    return out


def load_labels(features_dir: str | Path, dataset_name: str) -> np.ndarray:
    path = Path(features_dir) / f"{dataset_name}_labels.npy"
    if not path.exists():
        return np.array([])
    return np.load(path)


def select_layers(layers_to_extract: list[int], n_layers: int) -> list[int]:
    """Resolve negative indices (from end) to concrete layer indices in [0, n_layers).

    Args:
        layers_to_extract: Layer indices; negative values count from last layer.
        n_layers: Total number of layers in the model.

    Returns:
        List of valid layer indices in [0, n_layers); out-of-range entries are dropped.
    """
    return [n_layers + li if li < 0 else li for li in layers_to_extract if 0 <= (n_layers + li if li < 0 else li) < n_layers]
