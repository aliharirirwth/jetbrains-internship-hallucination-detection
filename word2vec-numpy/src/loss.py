"""Negative sampling loss: -J where J = log σ(v_c·v_o) + Σ_k log σ(-v_c·v_k). We minimize loss."""

from __future__ import annotations

import numpy as np

from .model import sigmoid


def loss(pos_score: float, neg_scores: np.ndarray) -> float:
    """Loss = -J (minimize this). J = log σ(pos) + Σ log σ(-neg)."""
    eps = 1e-12
    sig_pos = sigmoid(pos_score)
    sig_neg = sigmoid(neg_scores)
    return float(-np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps)))
