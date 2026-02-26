"""Geometric features from hidden states: Mahalanobis, cosine, norm, layer difference."""

from __future__ import annotations

from typing import Any

import numpy as np


def mahalanobis_distance(
    x: np.ndarray,
    class_mean: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """MD = sqrt((x - μ)^T Σ^{-1} (x - μ))."""
    d = x - class_mean
    return float(np.sqrt(np.maximum(0.0, d @ inv_cov @ d)))


def mahalanobis_features(
    x: np.ndarray,
    mean_faithful: np.ndarray,
    mean_hallucination: np.ndarray,
    inv_cov_faithful: np.ndarray,
    inv_cov_hallucination: np.ndarray,
) -> np.ndarray:
    """Return [MD_hallucination, MD_faithful, MD_hallucination - MD_faithful]."""
    md_h = mahalanobis_distance(x, mean_hallucination, inv_cov_hallucination)
    md_f = mahalanobis_distance(x, mean_faithful, inv_cov_faithful)
    return np.array([md_h, md_f, md_h - md_f], dtype=np.float32)


def cosine_similarity_features(q_hidden: np.ndarray, a_hidden: np.ndarray) -> np.ndarray:
    """Cosine similarity between question and answer representations (scalar)."""
    nq = np.linalg.norm(q_hidden) + 1e-12
    na = np.linalg.norm(a_hidden) + 1e-12
    sim = float(np.dot(q_hidden, a_hidden) / (nq * na))
    return np.array([sim], dtype=np.float32)


def representation_norm(hidden: np.ndarray) -> float:
    """L2 norm of hidden state."""
    return float(np.linalg.norm(hidden))


def layer_difference(layer_i: np.ndarray, layer_j: np.ndarray) -> np.ndarray:
    """Difference vector between two layers (captures representation change through depth)."""
    return (layer_j - layer_i).astype(np.float32)


def build_feature_vector(
    sample_features: dict[str, np.ndarray | float],
    config: dict[str, Any],
) -> np.ndarray:
    """Concatenate all enabled geometric features into a single vector."""
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
