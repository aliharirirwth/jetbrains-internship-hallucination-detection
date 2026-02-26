from __future__ import annotations

import numpy as np

from .model import sigmoid


def loss(pos_score: float, neg_scores: np.ndarray) -> float:
    """Negative sampling loss (minimize this).

    Loss = -J where J = log σ(pos_score) + Σ log σ(-neg_scores).

    Args:
        pos_score: Scalar positive dot-product (center·context).
        neg_scores: Array of negative dot-products.

    Returns:
        Scalar loss (non-negative).
    """
    eps = 1e-12
    sig_pos = sigmoid(pos_score)
    sig_neg = sigmoid(neg_scores)
    return float(-np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps)))
