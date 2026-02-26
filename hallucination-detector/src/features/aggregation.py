"""Layer selection and pooling strategies for building feature matrices from saved hidden states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_layer_features(
    features_dir: str | Path,
    dataset_name: str,
    model_short: str,
    layers: list[int],
    pooling: str,
) -> dict[int, np.ndarray]:
    """Load per-layer .npy arrays; return dict layer_index -> (N, hidden_dim)."""
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
    """Resolve negative indices (from end) to concrete layer indices."""
    return [n_layers + li if li < 0 else li for li in layers_to_extract if 0 <= (n_layers + li if li < 0 else li) < n_layers]
