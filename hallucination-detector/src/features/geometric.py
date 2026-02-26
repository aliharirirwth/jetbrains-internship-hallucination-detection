from __future__ import annotations

from typing import Any

import numpy as np


def mahalanobis_distance(
    x: np.ndarray,
    class_mean: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """Mahalanobis distance: sqrt((x - μ)^T Σ^{-1} (x - μ)).

    Args:
        x: Query vector (shape (d,)).
        class_mean: Class mean vector (shape (d,)).
        inv_cov: Inverse covariance matrix (shape (d, d)).

    Returns:
        Non-negative scalar distance.
    """
    d = x - class_mean
    return float(np.sqrt(np.maximum(0.0, d @ inv_cov @ d)))


def mahalanobis_features(
    x: np.ndarray,
    mean_faithful: np.ndarray,
    mean_hallucination: np.ndarray,
    inv_cov_faithful: np.ndarray,
    inv_cov_hallucination: np.ndarray,
) -> np.ndarray:
    """Build Mahalanobis-based feature vector for a single sample.

    Args:
        x: Query vector (shape (d,)).
        mean_faithful: Mean of faithful (non-hallucination) class.
        mean_hallucination: Mean of hallucination class.
        inv_cov_faithful: Inverse covariance for faithful class.
        inv_cov_hallucination: Inverse covariance for hallucination class.

    Returns:
        Array of shape (3,) with [MD_hallucination, MD_faithful, MD_hallucination - MD_faithful].
    """
    md_h = mahalanobis_distance(x, mean_hallucination, inv_cov_hallucination)
    md_f = mahalanobis_distance(x, mean_faithful, inv_cov_faithful)
    return np.array([md_h, md_f, md_h - md_f], dtype=np.float32)


def cosine_similarity_features(q_hidden: np.ndarray, a_hidden: np.ndarray) -> np.ndarray:
    """Cosine similarity between question and answer hidden states.

    Args:
        q_hidden: Question representation vector.
        a_hidden: Answer representation vector.

    Returns:
        Array of shape (1,) with the cosine similarity in [-1, 1].
    """
    nq = np.linalg.norm(q_hidden) + 1e-12
    na = np.linalg.norm(a_hidden) + 1e-12
    sim = float(np.dot(q_hidden, a_hidden) / (nq * na))
    return np.array([sim], dtype=np.float32)


def representation_norm(hidden: np.ndarray) -> float:
    """L2 norm of the hidden state vector.

    Args:
        hidden: Hidden state vector.

    Returns:
        Non-negative scalar norm.
    """
    return float(np.linalg.norm(hidden))


def layer_difference(layer_i: np.ndarray, layer_j: np.ndarray) -> np.ndarray:
    """Difference vector between two layer representations.

    Captures representation change through depth (e.g. layer_j - layer_i).

    Args:
        layer_i: Hidden state from earlier layer (shape (d,)).
        layer_j: Hidden state from later layer (shape (d,)).

    Returns:
        Difference vector (layer_j - layer_i) as float32.
    """
    return (layer_j - layer_i).astype(np.float32)


def build_feature_vector(
    sample_features: dict[str, np.ndarray | float],
    config: dict[str, Any],
) -> np.ndarray:
    """Concatenate all enabled geometric features into a single vector.

    Args:
        sample_features: Dict with keys such as "mahalanobis", "cosine_sim", "norm", "layer_diff".
        config: Dict with flags use_mahalanobis, use_cosine_sim, use_norm, use_layer_diff.

    Returns:
        One-dimensional float32 array of concatenated features, or empty array if none enabled.
    """
    parts: list[np.ndarray] = []
    if config.get("use_mahalanobis", True) and "mahalanobis" in sample_features:
        parts.append(np.atleast_1d(sample_features["mahalanobis"]))
    if config.get("use_cosine_sim", True) and "cosine_sim" in sample_features:
        parts.append(np.atleast_1d(sample_features["cosine_sim"]))
    if config.get("use_norm", True) and "norm" in sample_features:
        parts.append(np.array([sample_features["norm"]], dtype=np.float32))
    if config.get("use_layer_diff", True) and "layer_diff" in sample_features:
        parts.append(np.atleast_1d(sample_features["layer_diff"]))
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)
